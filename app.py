import streamlit as st
import pandas as pd
import duckdb
import tempfile
from concurrent.futures import ProcessPoolExecutor
import re
from datetime import datetime
import io
from functools import lru_cache
import multiprocessing as mp
import os
from pathlib import Path
import openpyxl
import xlrd

# Configure Streamlit
st.set_page_config(
    page_title="Grammar Check Analytics",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'parquet_path' not in st.session_state:
    st.session_state.parquet_path = None
if 'analytics' not in st.session_state:
    st.session_state.analytics = None
if 'original_columns' not in st.session_state:
    st.session_state.original_columns = []

# Pre-compiled patterns
AGENT_PATTERN_1 = re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\+\d{4})\s+Agent:(.*?)(?=\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\Z)', re.DOTALL)
AGENT_PATTERN_2 = re.compile(r'\[\d{2}:\d{2}:\d{2}\s+AGENT\]:(.*?)(?=\[\d{2}:\d{2}:\d{2}|\Z)', re.DOTALL | re.IGNORECASE)

# Grammar patterns
SPELLING_PATTERN = re.compile(r'\b(recieve|occured|seperate|definately|accomodate|occassion|wierd|untill|thier|wich|becuase|alot|tommorow|existance|appearence|begining|beleive|calender|cemetary|changable|collegue|concious|occured|embarrass|enviroment|excercise|existance|fourty|goverment|guarentee|harrass|higeine|ignorence|imediately|independant|instresting|liason|libary|maintainance|mispell|neccessary|noticable|occassion|occurance|persistant|pharoah|playwrite|posession|prefered|priviledge|reccomend|refered|religous|rythm|sieze|succesful|supercede|suprise|temperture|tendancy|truely|unforseen|unnecesary|untill|wierd)\b', re.IGNORECASE)
CONTRACTION_PATTERN = re.compile(r'\b(dont|wont|cant|shouldnt|wouldnt|couldnt|didnt|doesnt|isnt|arent|wasnt|werent|havent|hasnt|hadnt)\b')
DOUBLE_SPACE = re.compile(r'\s{2,}')
SPACE_BEFORE_PUNCT = re.compile(r'\s+([.!?,;:])')
NO_SPACE_AFTER_PUNCT = re.compile(r'([.!?,;:])([A-Za-z])')
MISSING_CAPITAL = re.compile(r'([.!?]\s+)([a-z])')
ARTICLE_A_VOWEL = re.compile(r'\ba\s+[aeiouAEIOU]\w+')
ARTICLE_AN_CONSONANT = re.compile(r'\ban\s+[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w+')
SUBJECT_VERB_ERROR = re.compile(r'\b(he|she|it)\s+(were|are)\b|\b(they|we|you)\s+(was|is)\b', re.IGNORECASE)

# Common misspellings dictionary
COMMON_MISSPELLINGS = {
    'recieve': 'receive', 'occured': 'occurred', 'seperate': 'separate',
    'definately': 'definitely', 'accomodate': 'accommodate', 'wierd': 'weird',
    'untill': 'until', 'thier': 'their', 'wich': 'which', 'becuase': 'because',
    'alot': 'a lot', 'occassion': 'occasion', 'tommorow': 'tomorrow',
    'existance': 'existence', 'appearence': 'appearance', 'begining': 'beginning',
    'beleive': 'believe', 'calender': 'calendar', 'collegue': 'colleague'
}

def load_file_to_dataframe(uploaded_file):
    """Load CSV, XLSX, or XLS file into pandas DataFrame"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        elif file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_extension == 'xls':
            df = pd.read_excel(uploaded_file, engine='xlrd')
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return df, file_extension
    except UnicodeDecodeError:
        # Try different encoding for CSV
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        return df, file_extension
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def save_to_parquet(df, filename_prefix="data"):
    """Save DataFrame to Parquet format"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
    parquet_path = temp_file.name
    temp_file.close()
    
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
    return parquet_path

def clean_text_fast(text):
    """Ultra-fast text cleaning"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
    text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
    text = DOUBLE_SPACE.sub(' ', text)
    return text.strip()

def extract_agent_messages(text):
    """Extract agent messages from transcript"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text)
    messages = []
    
    matches1 = AGENT_PATTERN_1.findall(text)
    for _, message in matches1:
        clean = clean_text_fast(message)
        if clean:
            messages.append(clean)
    
    matches2 = AGENT_PATTERN_2.findall(text)
    for message in matches2:
        clean = clean_text_fast(message)
        if clean:
            messages.append(clean)
    
    # If no agent pattern found, use the whole text
    if not messages:
        clean = clean_text_fast(text)
        if clean:
            return clean
    
    return ' '.join(messages)

@lru_cache(maxsize=10000)
def count_words(text):
    """Cached word counting"""
    return len(text.split())

def check_text_fast(text):
    """Lightning-fast grammar checking with detailed error tracking"""
    if not text or len(text) < 5:
        return {
            'total_errors': 0,
            'spelling_error_count': 0,
            'grammar_error_count': 0,
            'punctuation_error_count': 0,
            'spelling_errors': '',
            'grammar_errors': '',
            'punctuation_errors': '',
            'error_details': '',
            'suggested_corrections': '',
            'word_count': 0
        }
    
    all_errors = []
    error_details = []
    corrections = []
    
    # SPELLING ERRORS
    spelling_matches = SPELLING_PATTERN.findall(text)
    spelling_errors = []
    for match in spelling_matches:
        spelling_errors.append(match)
        all_errors.append(f"SPELLING: {match}")
        if match.lower() in COMMON_MISSPELLINGS:
            correction = COMMON_MISSPELLINGS[match.lower()]
            corrections.append(f"{match}→{correction}")
            error_details.append(f"Misspelled '{match}' (should be '{correction}')")
    
    # GRAMMAR ERRORS
    grammar_errors = []
    
    # Missing apostrophes in contractions
    contraction_matches = CONTRACTION_PATTERN.findall(text)
    if contraction_matches:
        for match in contraction_matches[:3]:
            grammar_errors.append(f"Missing apostrophe: {match}")
            all_errors.append(f"GRAMMAR: Missing apostrophe in '{match}'")
            error_details.append(f"Missing apostrophe in '{match}'")
    
    # Article errors
    if ARTICLE_A_VOWEL.search(text):
        grammar_errors.append("Article error: 'a' before vowel")
        all_errors.append("GRAMMAR: Use 'an' before vowel sound")
        error_details.append("Use 'an' before vowel sound")
    
    if ARTICLE_AN_CONSONANT.search(text):
        grammar_errors.append("Article error: 'an' before consonant")
        all_errors.append("GRAMMAR: Use 'a' before consonant sound")
        error_details.append("Use 'a' before consonant sound")
    
    # Subject-verb agreement
    if SUBJECT_VERB_ERROR.search(text):
        grammar_errors.append("Subject-verb disagreement")
        all_errors.append("GRAMMAR: Subject-verb disagreement")
        error_details.append("Subject-verb disagreement detected")
    
    # PUNCTUATION ERRORS
    punctuation_errors = []
    
    if SPACE_BEFORE_PUNCT.search(text):
        punctuation_errors.append("Space before punctuation")
        all_errors.append("PUNCTUATION: Unnecessary space before punctuation")
        error_details.append("Space before punctuation mark")
    
    if NO_SPACE_AFTER_PUNCT.search(text):
        punctuation_errors.append("Missing space after punctuation")
        all_errors.append("PUNCTUATION: Missing space after punctuation")
        error_details.append("Missing space after punctuation")
    
    if MISSING_CAPITAL.search(text):
        punctuation_errors.append("Missing capitalization")
        all_errors.append("PUNCTUATION: Missing capitalization after period")
        error_details.append("Missing capitalization")
    
    word_count = count_words(text)
    
    return {
        'total_errors': len(all_errors),
        'spelling_error_count': len(spelling_errors),
        'grammar_error_count': len(grammar_errors),
        'punctuation_error_count': len(punctuation_errors),
        'spelling_errors': ', '.join(spelling_errors[:10]) if spelling_errors else '',
        'grammar_errors': ' | '.join(grammar_errors[:5]) if grammar_errors else '',
        'punctuation_errors': ' | '.join(punctuation_errors[:5]) if punctuation_errors else '',
        'error_details': ' || '.join(error_details[:15]) if error_details else '',
        'suggested_corrections': '; '.join(corrections[:10]) if corrections else '',
        'word_count': word_count
    }

def process_chunk(texts):
    """Process a chunk of texts"""
    return [check_text_fast(text) for text in texts]

def process_dataframe_with_grammar(df, text_column, num_workers=4):
    """Process DataFrame with grammar checking and return enhanced data"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Store original columns
    original_columns = df.columns.tolist()
    total_rows = len(df)
    
    status_text.text(f"Extracting text from column '{text_column}'...")
    progress_bar.progress(0.1)
    
    # Extract agent messages
    df['extracted_text'] = df[text_column].apply(extract_agent_messages)
    
    # Keep rows with text
    df_with_text = df[df['extracted_text'].str.len() > 0].copy()
    rows_with_text = len(df_with_text)
    
    if rows_with_text == 0:
        raise ValueError("No valid text found in the selected column")
    
    status_text.text(f"Processing {rows_with_text} rows with text (out of {total_rows} total)...")
    progress_bar.progress(0.2)
    
    # Split into chunks for parallel processing
    texts = df_with_text['extracted_text'].tolist()
    chunk_size = max(len(texts) // (num_workers * 2), 100)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    # Process in parallel
    all_results = []
    processed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        
        for future in futures:
            results = future.result()
            all_results.extend(results)
            processed += len(results)
            progress = 0.2 + (processed / rows_with_text) * 0.7
            progress_bar.progress(min(progress, 0.9))
            status_text.text(f"Processed {processed}/{rows_with_text} messages...")
    
    # Add results to dataframe
    status_text.text("Adding grammar analysis results...")
    
    # Add all error columns
    df_with_text['total_errors'] = [r['total_errors'] for r in all_results]
    df_with_text['spelling_error_count'] = [r['spelling_error_count'] for r in all_results]
    df_with_text['grammar_error_count'] = [r['grammar_error_count'] for r in all_results]
    df_with_text['punctuation_error_count'] = [r['punctuation_error_count'] for r in all_results]
    df_with_text['spelling_errors'] = [r['spelling_errors'] for r in all_results]
    df_with_text['grammar_errors'] = [r['grammar_errors'] for r in all_results]
    df_with_text['punctuation_errors'] = [r['punctuation_errors'] for r in all_results]
    df_with_text['error_details'] = [r['error_details'] for r in all_results]
    df_with_text['suggested_corrections'] = [r['suggested_corrections'] for r in all_results]
    df_with_text['word_count'] = [r['word_count'] for r in all_results]
    
    # Calculate error rate
    df_with_text['error_rate_percent'] = (
        df_with_text['total_errors'] / df_with_text['word_count'].replace(0, 1) * 100
    ).round(2)
    
    # Add metadata
    df_with_text['processing_timestamp'] = datetime.now().isoformat()
    df_with_text['analysis_version'] = 'v1.0_fast_multiformat'
    
    progress_bar.progress(0.95)
    status_text.text("Saving to Parquet format for DuckDB...")
    
    # Save to Parquet
    parquet_path = save_to_parquet(df_with_text)
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Complete! Processed {rows_with_text} messages and saved to Parquet")
    
    return df_with_text, parquet_path, original_columns

def create_duckdb_analytics(parquet_path):
    """Create analytics using DuckDB on Parquet file"""
    conn = duckdb.connect(':memory:')
    
    # Load Parquet file
    conn.execute(f"""
        CREATE TABLE data AS 
        SELECT * FROM read_parquet('{parquet_path}')
    """)
    
    # Summary statistics
    summary = conn.execute("""
        SELECT 
            COUNT(*) as total_rows,
            SUM(total_errors) as total_errors,
            SUM(spelling_error_count) as total_spelling_errors,
            SUM(grammar_error_count) as total_grammar_errors,
            SUM(punctuation_error_count) as total_punctuation_errors,
            ROUND(AVG(total_errors), 2) as avg_errors_per_row,
            ROUND(AVG(error_rate_percent), 2) as avg_error_rate_percent,
            ROUND(AVG(word_count), 0) as avg_word_count,
            SUM(CASE WHEN total_errors = 0 THEN 1 ELSE 0 END) as error_free_rows,
            ROUND(SUM(CASE WHEN total_errors = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as error_free_percent,
            MAX(total_errors) as max_errors_in_row,
            MIN(total_errors) as min_errors_in_row
        FROM data
    """).df()
    
    # Error type breakdown
    error_breakdown = conn.execute("""
        SELECT 
            'Spelling' as error_type,
            SUM(spelling_error_count) as count,
            ROUND(SUM(spelling_error_count) * 100.0 / NULLIF(SUM(total_errors), 0), 1) as percentage
        FROM data
        UNION ALL
        SELECT 
            'Grammar' as error_type,
            SUM(grammar_error_count) as count,
            ROUND(SUM(grammar_error_count) * 100.0 / NULLIF(SUM(total_errors), 0), 1) as percentage
        FROM data
        UNION ALL
        SELECT 
            'Punctuation' as error_type,
            SUM(punctuation_error_count) as count,
            ROUND(SUM(punctuation_error_count) * 100.0 / NULLIF(SUM(total_errors), 0), 1) as percentage
        FROM data
        ORDER BY count DESC
    """).df()
    
    # Top rows with most errors
    top_errors = conn.execute("""
        SELECT *
        FROM data
        ORDER BY total_errors DESC
        LIMIT 20
    """).df()
    
    # Error distribution
    error_distribution = conn.execute("""
        SELECT 
            CASE 
                WHEN total_errors = 0 THEN '0 errors'
                WHEN total_errors <= 2 THEN '1-2 errors'
                WHEN total_errors <= 5 THEN '3-5 errors'
                WHEN total_errors <= 10 THEN '6-10 errors'
                WHEN total_errors <= 20 THEN '11-20 errors'
                ELSE '20+ errors'
            END as error_range,
            COUNT(*) as row_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
        FROM data
        GROUP BY error_range
        ORDER BY 
            CASE error_range
                WHEN '0 errors' THEN 1
                WHEN '1-2 errors' THEN 2
                WHEN '3-5 errors' THEN 3
                WHEN '6-10 errors' THEN 4
                WHEN '11-20 errors' THEN 5
                ELSE 6
            END
    """).df()
    
    # Get full data
    full_data = conn.execute("SELECT * FROM data ORDER BY total_errors DESC").df()
    
    conn.close()
    
    return {
        'summary': summary,
        'error_breakdown': error_breakdown,
        'top_errors': top_errors,
        'error_distribution': error_distribution,
        'full_data': full_data
    }

def main():
    st.title("📝 Advanced Grammar Check Analytics")
    st.markdown("### Multi-format file support with Parquet + DuckDB optimization")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.info("""
        **Supported Formats:**
        - CSV (.csv)
        - Excel 2007+ (.xlsx)
        - Excel 97-2003 (.xls)
        
        **Processing:**
        - Converts to Parquet
        - DuckDB for analytics
        - Parallel processing
        
        **Output:**
        - All original data
        - Error counts by type
        - Specific errors found
        - Suggested corrections
        - Error details
        """)
        
        st.markdown("---")
        
        num_workers = st.slider(
            "Parallel Workers",
            min_value=1,
            max_value=mp.cpu_count(),
            value=min(4, mp.cpu_count()),
            help="More workers = faster processing"
        )
        
        st.markdown("---")
        st.markdown("**Detection Coverage:**")
        st.markdown("✅ 50+ common misspellings")
        st.markdown("✅ Missing apostrophes")
        st.markdown("✅ Article errors (a/an)")
        st.markdown("✅ Subject-verb agreement")
        st.markdown("✅ Punctuation spacing")
        st.markdown("✅ Capitalization")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Analytics Dashboard", "💾 Download Results"])
    
    with tab1:
        st.header("Upload Data File")
        
        st.markdown("""
        Upload your data file in **CSV, XLSX, or XLS** format. The app will:
        1. Load your file into memory
        2. Convert to Parquet for optimal processing
        3. Analyze grammar and errors
        4. Keep ALL original columns + add error analysis columns
        """)
        
        uploaded_file = st.file_uploader(
            "Choose your file",
            type=['csv', 'xlsx', 'xls'],
            help="Supports CSV, Excel 2007+ (xlsx), and Excel 97-2003 (xls)"
        )
        
        if uploaded_file:
            try:
                # Load file
                with st.spinner("Loading file..."):
                    df_original, file_type = load_file_to_dataframe(uploaded_file)
                
                st.success(f"✅ Loaded {file_type.upper()} file: {len(df_original):,} rows × {len(df_original.columns)} columns")
                
                # Preview
                st.subheader("Data Preview")
                st.dataframe(df_original.head(10), use_container_width=True)
                
                # Show file info
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", f"{len(df_original):,}")
                col2.metric("Total Columns", len(df_original.columns))
                col3.metric("File Type", file_type.upper())
                
                # Column selection
                st.subheader("Select Text Column for Analysis")
                text_column = st.selectbox(
                    "Choose the column containing text/transcripts to analyze:",
                    options=df_original.columns.tolist(),
                    help="This column will be analyzed for grammar and spelling errors"
                )
                
                # Show sample of selected column
                if text_column:
                    st.markdown("**Sample from selected column:**")
                    sample_text = df_original[text_column].dropna().iloc[0] if not df_original[text_column].dropna().empty else "No data"
                    st.text_area("Sample text", str(sample_text)[:500] + "...", height=100, disabled=True)
                
                # Process button
                st.markdown("---")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if st.button("🚀 Start Grammar Analysis", type="primary", use_container_width=True):
                        start_time = datetime.now()
                        
                        try:
                            with st.spinner("Processing... This may take a few minutes for large files"):
                                df_processed, parquet_path, original_cols = process_dataframe_with_grammar(
                                    df_original,
                                    text_column,
                                    num_workers
                                )
                            
                            # Save to session
                            st.session_state.processed_data = df_processed
                            st.session_state.parquet_path = parquet_path
                            st.session_state.original_columns = original_cols
                            
                            # Generate analytics
                            with st.spinner("Generating analytics with DuckDB..."):
                                analytics = create_duckdb_analytics(parquet_path)
                                st.session_state.analytics = analytics
                            
                            # Show results
                            processing_time = (datetime.now() - start_time).total_seconds()
                            
                            st.success(f"✅ Successfully processed {len(df_processed):,} rows in {processing_time:.1f} seconds!")
                            st.balloons()
                            
                            # Quick stats
                            st.subheader("Processing Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            summary = analytics['summary'].iloc[0]
                            col1.metric("Rows Processed", f"{summary['total_rows']:,}")
                            col2.metric("Total Errors Found", f"{int(summary['total_errors']):,}")
                            col3.metric("Avg Errors/Row", f"{summary['avg_errors_per_row']:.2f}")
                            col4.metric("Error-Free Rows", f"{summary['error_free_percent']:.1f}%")
                            
                            st.info("📊 Go to the **Analytics Dashboard** tab to explore detailed insights!")
                            
                        except Exception as e:
                            st.error(f"❌ Error during processing: {str(e)}")
                            st.exception(e)
                
                with col2:
                    st.metric("Processing Speed", f"~{min(4, num_workers) * 200:.0f} rows/sec")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.exception(e)
        
        else:
            st.info("👆 Upload a CSV, XLSX, or XLS file to begin")
    
    with tab2:
        st.header("Analytics Dashboard")
        
        if st.session_state.analytics is not None:
            analytics = st.session_state.analytics
            
            # Summary Statistics
            st.subheader("📈 Summary Statistics")
            summary = analytics['summary'].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{summary['total_rows']:,}")
            col2.metric("Total Errors", f"{int(summary['total_errors']):,}")
            col3.metric("Avg Errors/Row", f"{summary['avg_errors_per_row']:.2f}")
            col4.metric("Avg Word Count", f"{int(summary['avg_word_count']):,}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Spelling Errors", f"{int(summary['total_spelling_errors']):,}")
            col2.metric("Grammar Errors", f"{int(summary['total_grammar_errors']):,}")
            col3.metric("Punctuation Errors", f"{int(summary['total_punctuation_errors']):,}")
            col4.metric("Error-Free Rows", f"{summary['error_free_percent']:.1f}%")
            
            st.markdown("---")
            
            # Error Breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Error Type Breakdown")
                st.dataframe(analytics['error_breakdown'], use_container_width=True, hide_index=True)
                st.bar_chart(analytics['error_breakdown'].set_index('error_type')['count'])
            
            with col2:
                st.subheader("📊 Error Distribution")
                st.dataframe(analytics['error_distribution'], use_container_width=True, hide_index=True)
                st.bar_chart(analytics['error_distribution'].set_index('error_range')['row_count'])
            
            st.markdown("---")
            
            # Top errors
            st.subheader("🔝 Top 20 Rows with Most Errors")
            
            # Select key columns to display
            display_cols = ['total_errors', 'spelling_error_count', 'grammar_error_count', 
                          'punctuation_error_count', 'error_details', 'extracted_text']
            
            # Add original columns if they exist
            available_display_cols = [col for col in display_cols if col in analytics['top_errors'].columns]
            
            st.dataframe(
                analytics['top_errors'][available_display_cols].head(20),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            
            # Full data preview
            st.subheader("📋 Complete Analysis Data Preview")
            st.markdown(f"**Total rows:** {len(analytics['full_data']):,}")
            
            # Show first 100 rows
            st.dataframe(
                analytics['full_data'].head(100),
                use_container_width=True,
                height=400
            )
            
        else:
            st.info("📤 Please process a file in the Upload tab first")
    
    with tab3:
        st.header("💾 Download Results")
        
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            
            st.success(f"✅ Results ready for download: {len(df):,} rows with {len(df.columns)} columns")
            
            st.markdown("### What's included in the download:")
            st.markdown("""
            ✅ **All original columns from your uploaded file**
            ✅ `extracted_text` - Cleaned text that was analyzed
            ✅ `total_errors` - Total number of errors found
            ✅ `spelling_error_count` - Number of spelling errors
            ✅ `grammar_error_count` - Number of grammar errors
            ✅ `punctuation_error_count` - Number of punctuation errors
            ✅ `spelling_errors` - Specific spelling mistakes found
            ✅ `grammar_errors` - Specific grammar issues found
            ✅ `punctuation_errors` - Specific punctuation issues found
            ✅ `error_details` - Detailed description of all errors
            ✅ `suggested_corrections` - Suggested fixes for errors
            ✅ `word_count` - Number of words in the text
            ✅ `error_rate_percent` - Error rate as percentage
            ✅ `processing_timestamp` - When the analysis was performed
            """)
            
            st.markdown("---")
            
            # Download options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📄 CSV Format")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"grammar_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption("Compatible with Excel, Google Sheets, etc.")
            
            with col2:
                st.markdown("#### 📊 Excel Format")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Write main data
                    df.to_excel(writer, sheet_name='Complete Analysis', index=False)
                    
                    # Write analytics if available
                    if st.session_state.analytics:
                        analytics = st.session_state.analytics
                        analytics['summary'].to_excel(writer, sheet_name='Summary', index=False)
                        analytics['error_breakdown'].to_excel(writer, sheet_name='Error Breakdown', index=False)
                        analytics['error_distribution'].to_excel(writer, sheet_name='Error Distribution', index=False)
                        analytics['top_errors'].to_excel(writer, sheet_name='Top 20 Errors', index=False)
                    
                    # Format the main sheet
                    workbook = writer.book
                    worksheet = writer.sheets['Complete Analysis']
                    
                    # Auto-adjust column widths
                    for i, col in enumerate(df.columns):
                        max_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
                        worksheet.set_column(i, i, min(max_len, 50))
                    
                    # Add header format
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#4A90E2',
                        'font_color': 'white',
                        'border': 1
                    })
                    
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                
                st.download_button(
                    label="📥 Download as Excel",
                    data=output.getvalue(),
                    file_name=f"grammar_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.caption("Multi-sheet workbook with analytics")
            
            with col3:
                st.markdown("#### 🗄️ Parquet Format")
                if st.session_state.parquet_path:
                    with open(st.session_state.parquet_path, 'rb') as f:
                        parquet_data = f.read()
                    
                    st.download_button(
                        label="📥 Download as Parquet",
                        data=parquet_data,
                        file_name=f"grammar_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                    st.caption("Optimized for big data tools")
            
            st.markdown("---")
            
            # Data summary
            st.subheader("📋 Column Summary")
            col_info = pd.DataFrame({
                'Column Name': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
            
            # Sample data
            st.subheader("🔍 Sample of Processed Data")
            st.dataframe(df.head(10), use_container_width=True)
            
        else:
            st.info("📤 Please process a file in the Upload tab first")
            
            st.markdown("---")
            st.markdown("### 💡 Tips for Using Downloaded Data")
            st.markdown("""
            - **Excel**: Best for viewing and manual review
            - **CSV**: Universal format, works everywhere
            - **Parquet**: Best for big data analysis with Python, R, or SQL tools
            
            **Filtering suggestions:**
            - Sort by `total_errors` to find most problematic rows
            - Filter by `error_rate_percent` to find rows with high error density
            - Use `error_details` column to understand specific issues
            - Check `suggested_corrections` for quick fixes
            """)

if __name__ == "__main__":
    main()
