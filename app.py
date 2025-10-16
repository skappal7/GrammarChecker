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

# Pre-compiled patterns
AGENT_PATTERN_1 = re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\+\d{4})\s+Agent:(.*?)(?=\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\Z)', re.DOTALL)
AGENT_PATTERN_2 = re.compile(r'\[\d{2}:\d{2}:\d{2}\s+AGENT\]:(.*?)(?=\[\d{2}:\d{2}:\d{2}|\Z)', re.DOTALL | re.IGNORECASE)
SPELLING_PATTERN = re.compile(r'\b(recieve|occured|seperate|definately|accomodate|occassion|wierd|untill|thier|wich|becuase|alot|tommorow|existance|appearence|begining|beleive|calender|cemetary|changable|collegue|concious|embarrass|enviroment|excercise|fourty|goverment|guarentee|harrass|higeine|ignorence|imediately|independant|instresting|liason|libary|maintainance|mispell|neccessary|noticable|occurance|persistant|posession|prefered|priviledge|reccomend|refered|religous|rythm|sieze|succesful|supercede|suprise|temperture|tendancy|truely|unforseen|unnecesary)\b', re.IGNORECASE)
CONTRACTION_PATTERN = re.compile(r'\b(dont|wont|cant|shouldnt|wouldnt|couldnt|didnt|doesnt|isnt|arent|wasnt|werent|havent|hasnt|hadnt)\b')
DOUBLE_SPACE = re.compile(r'\s{2,}')
SPACE_BEFORE_PUNCT = re.compile(r'\s+([.!?,;:])')
NO_SPACE_AFTER_PUNCT = re.compile(r'([.!?,;:])([A-Za-z])')
MISSING_CAPITAL = re.compile(r'([.!?]\s+)([a-z])')
ARTICLE_A_VOWEL = re.compile(r'\ba\s+[aeiouAEIOU]\w+')
ARTICLE_AN_CONSONANT = re.compile(r'\ban\s+[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w+')
SUBJECT_VERB_ERROR = re.compile(r'\b(he|she|it)\s+(were|are)\b|\b(they|we|you)\s+(was|is)\b', re.IGNORECASE)

COMMON_MISSPELLINGS = {
    'recieve': 'receive', 'occured': 'occurred', 'seperate': 'separate',
    'definately': 'definitely', 'accomodate': 'accommodate', 'wierd': 'weird',
    'untill': 'until', 'thier': 'their', 'wich': 'which', 'becuase': 'because',
    'alot': 'a lot', 'occassion': 'occasion', 'tommorow': 'tomorrow',
    'existance': 'existence', 'appearence': 'appearance', 'begining': 'beginning',
    'beleive': 'believe', 'calender': 'calendar', 'collegue': 'colleague'
}

def load_file(uploaded_file):
    """Load CSV, XLSX, or XLS file"""
    ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if ext == 'csv':
            return pd.read_csv(uploaded_file, encoding='utf-8')
        elif ext == 'xlsx':
            return pd.read_excel(uploaded_file, engine='openpyxl')
        elif ext == 'xls':
            return pd.read_excel(uploaded_file, engine='xlrd')
        else:
            raise ValueError(f"Unsupported format: {ext}")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding='latin-1')

def save_parquet(df):
    """Save to Parquet"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
    tmp.close()
    df.to_parquet(tmp.name, engine='pyarrow', compression='snappy', index=False)
    return tmp.name

def clean_text(text):
    """Clean text"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
    text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
    text = DOUBLE_SPACE.sub(' ', text)
    return text.strip()

def extract_agent(text):
    """Extract agent messages"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text)
    messages = []
    
    for _, msg in AGENT_PATTERN_1.findall(text):
        clean = clean_text(msg)
        if clean:
            messages.append(clean)
    
    for msg in AGENT_PATTERN_2.findall(text):
        clean = clean_text(msg)
        if clean:
            messages.append(clean)
    
    if not messages:
        clean = clean_text(text)
        if clean:
            return clean
    
    return ' '.join(messages)

@lru_cache(maxsize=10000)
def count_words(text):
    """Count words"""
    return len(text.split())

def check_grammar(text):
    """Check grammar"""
    if not text or len(text) < 5:
        return {
            'total_errors': 0,
            'spelling_count': 0,
            'grammar_count': 0,
            'punctuation_count': 0,
            'spelling_errors': '',
            'grammar_errors': '',
            'punctuation_errors': '',
            'error_details': '',
            'corrections': '',
            'word_count': 0
        }
    
    errors = []
    details = []
    corrections = []
    
    # Spelling
    spelling = SPELLING_PATTERN.findall(text)
    for word in spelling:
        errors.append(f"SPELLING:{word}")
        if word.lower() in COMMON_MISSPELLINGS:
            fix = COMMON_MISSPELLINGS[word.lower()]
            corrections.append(f"{word}→{fix}")
            details.append(f"Misspelled '{word}' → '{fix}'")
    
    # Grammar
    grammar = []
    
    contractions = CONTRACTION_PATTERN.findall(text)
    if contractions:
        for c in contractions[:3]:
            grammar.append(f"Missing apostrophe: {c}")
            errors.append(f"GRAMMAR:apostrophe_{c}")
            details.append(f"Missing apostrophe in '{c}'")
    
    if ARTICLE_A_VOWEL.search(text):
        grammar.append("Use 'an' before vowel")
        errors.append("GRAMMAR:article_a_vowel")
        details.append("Use 'an' before vowel sound")
    
    if ARTICLE_AN_CONSONANT.search(text):
        grammar.append("Use 'a' before consonant")
        errors.append("GRAMMAR:article_an_consonant")
        details.append("Use 'a' before consonant sound")
    
    if SUBJECT_VERB_ERROR.search(text):
        grammar.append("Subject-verb disagreement")
        errors.append("GRAMMAR:subject_verb")
        details.append("Subject-verb disagreement")
    
    # Punctuation
    punctuation = []
    
    if SPACE_BEFORE_PUNCT.search(text):
        punctuation.append("Space before punctuation")
        errors.append("PUNCT:space_before")
        details.append("Space before punctuation")
    
    if NO_SPACE_AFTER_PUNCT.search(text):
        punctuation.append("No space after punctuation")
        errors.append("PUNCT:no_space_after")
        details.append("Missing space after punctuation")
    
    if MISSING_CAPITAL.search(text):
        punctuation.append("Missing capitalization")
        errors.append("PUNCT:missing_capital")
        details.append("Missing capitalization")
    
    return {
        'total_errors': len(errors),
        'spelling_count': len(spelling),
        'grammar_count': len(grammar),
        'punctuation_count': len(punctuation),
        'spelling_errors': ', '.join(spelling[:10]),
        'grammar_errors': ' | '.join(grammar[:5]),
        'punctuation_errors': ' | '.join(punctuation[:5]),
        'error_details': ' || '.join(details[:15]),
        'corrections': '; '.join(corrections[:10]),
        'word_count': count_words(text)
    }

def process_batch(texts):
    """Process batch"""
    return [check_grammar(t) for t in texts]

def process_file(df, text_col, workers=4):
    """Process file"""
    
    progress = st.progress(0)
    status = st.empty()
    
    status.text(f"Extracting text from '{text_col}'...")
    progress.progress(0.1)
    
    df['extracted_text'] = df[text_col].apply(extract_agent)
    df = df[df['extracted_text'].str.len() > 0].copy()
    
    if len(df) == 0:
        raise ValueError("No text found")
    
    total = len(df)
    status.text(f"Processing {total} rows...")
    progress.progress(0.2)
    
    texts = df['extracted_text'].tolist()
    chunk_size = max(len(texts) // (workers * 2), 100)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    results = []
    done = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_batch, chunk) for chunk in chunks]
        
        for future in futures:
            batch_results = future.result()
            results.extend(batch_results)
            done += len(batch_results)
            prog = 0.2 + (done / total) * 0.7
            progress.progress(min(prog, 0.9))
            status.text(f"Processed {done}/{total}...")
    
    status.text("Adding results...")
    
    df['total_errors'] = [r['total_errors'] for r in results]
    df['spelling_count'] = [r['spelling_count'] for r in results]
    df['grammar_count'] = [r['grammar_count'] for r in results]
    df['punctuation_count'] = [r['punctuation_count'] for r in results]
    df['spelling_errors'] = [r['spelling_errors'] for r in results]
    df['grammar_errors'] = [r['grammar_errors'] for r in results]
    df['punctuation_errors'] = [r['punctuation_errors'] for r in results]
    df['error_details'] = [r['error_details'] for r in results]
    df['corrections'] = [r['corrections'] for r in results]
    df['word_count'] = [r['word_count'] for r in results]
    
    df['error_rate'] = (df['total_errors'] / df['word_count'].replace(0, 1) * 100).round(2)
    df['timestamp'] = datetime.now().isoformat()
    
    progress.progress(0.95)
    status.text("Saving to Parquet...")
    
    parquet_path = save_parquet(df)
    
    progress.progress(1.0)
    status.text(f"✅ Complete! Processed {total} rows")
    
    return df, parquet_path

def create_analytics(parquet_path):
    """Create analytics"""
    conn = duckdb.connect(':memory:')
    
    conn.execute(f"CREATE TABLE data AS SELECT * FROM read_parquet('{parquet_path}')")
    
    summary = conn.execute("""
        SELECT 
            COUNT(*) as total_rows,
            SUM(total_errors) as total_errors,
            SUM(spelling_count) as spelling_errors,
            SUM(grammar_count) as grammar_errors,
            SUM(punctuation_count) as punctuation_errors,
            ROUND(AVG(total_errors), 2) as avg_errors,
            ROUND(AVG(error_rate), 2) as avg_rate,
            ROUND(AVG(word_count), 0) as avg_words,
            SUM(CASE WHEN total_errors = 0 THEN 1 ELSE 0 END) as error_free,
            ROUND(SUM(CASE WHEN total_errors = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as error_free_pct,
            MAX(total_errors) as max_errors
        FROM data
    """).df()
    
    breakdown = conn.execute("""
        SELECT 'Spelling' as type, SUM(spelling_count) as count,
            ROUND(SUM(spelling_count) * 100.0 / NULLIF(SUM(total_errors), 0), 1) as pct
        FROM data
        UNION ALL
        SELECT 'Grammar', SUM(grammar_count),
            ROUND(SUM(grammar_count) * 100.0 / NULLIF(SUM(total_errors), 0), 1)
        FROM data
        UNION ALL
        SELECT 'Punctuation', SUM(punctuation_count),
            ROUND(SUM(punctuation_count) * 100.0 / NULLIF(SUM(total_errors), 0), 1)
        FROM data
        ORDER BY count DESC
    """).df()
    
    distribution = conn.execute("""
        SELECT 
            CASE 
                WHEN total_errors = 0 THEN '0 errors'
                WHEN total_errors <= 2 THEN '1-2 errors'
                WHEN total_errors <= 5 THEN '3-5 errors'
                WHEN total_errors <= 10 THEN '6-10 errors'
                WHEN total_errors <= 20 THEN '11-20 errors'
                ELSE '20+ errors'
            END as range,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
        FROM data
        GROUP BY range
        ORDER BY 
            CASE range
                WHEN '0 errors' THEN 1
                WHEN '1-2 errors' THEN 2
                WHEN '3-5 errors' THEN 3
                WHEN '6-10 errors' THEN 4
                WHEN '11-20 errors' THEN 5
                ELSE 6
            END
    """).df()
    
    top_errors = conn.execute("SELECT * FROM data ORDER BY total_errors DESC LIMIT 20").df()
    full_data = conn.execute("SELECT * FROM data ORDER BY total_errors DESC").df()
    
    conn.close()
    
    return {
        'summary': summary,
        'breakdown': breakdown,
        'distribution': distribution,
        'top_errors': top_errors,
        'full_data': full_data
    }

def main():
    st.title("📝 Grammar Check Analytics")
    st.markdown("### Multi-format support with Parquet + DuckDB")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.info("""
        **Formats:** CSV, XLSX, XLS
        **Processing:** Parquet + DuckDB
        **Output:** All original data + errors
        """)
        
        workers = st.slider("Workers", 1, mp.cpu_count(), min(4, mp.cpu_count()))
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Analytics", "💾 Download"])
    
    with tab1:
        st.header("Upload File")
        
        file = st.file_uploader("Choose file", type=['csv', 'xlsx', 'xls'])
        
        if file:
            try:
                with st.spinner("Loading..."):
                    df = load_file(file)
                
                st.success(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
                
                st.subheader("Preview")
                st.dataframe(df.head(10), width='stretch')
                
                col1, col2 = st.columns(2)
                col1.metric("Rows", f"{len(df):,}")
                col2.metric("Columns", len(df.columns))
                
                st.subheader("Select Text Column")
                text_col = st.selectbox("Column to analyze:", df.columns.tolist())
                
                if text_col:
                    sample = df[text_col].dropna().iloc[0] if not df[text_col].dropna().empty else "No data"
                    st.text_area("Sample", str(sample)[:500], height=100, disabled=True)
                
                st.markdown("---")
                
                if st.button("🚀 Start Analysis", type="primary"):
                    start = datetime.now()
                    
                    try:
                        df_result, pq_path = process_file(df, text_col, workers)
                        
                        st.session_state.processed_data = df_result
                        st.session_state.parquet_path = pq_path
                        
                        with st.spinner("Generating analytics..."):
                            analytics = create_analytics(pq_path)
                            st.session_state.analytics = analytics
                        
                        elapsed = (datetime.now() - start).total_seconds()
                        
                        st.success(f"✅ Processed {len(df_result):,} rows in {elapsed:.1f}s")
                        st.balloons()
                        
                        summary = analytics['summary'].iloc[0]
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Rows", f"{summary['total_rows']:,}")
                        c2.metric("Errors", f"{int(summary['total_errors']):,}")
                        c3.metric("Avg/Row", f"{summary['avg_errors']:.2f}")
                        c4.metric("Error-Free", f"{summary['error_free_pct']:.1f}%")
                        
                        st.info("📊 Check Analytics tab")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
                        
            except Exception as e:
                st.error(f"❌ Load error: {str(e)}")
                st.exception(e)
        else:
            st.info("👆 Upload CSV, XLSX, or XLS")
    
    with tab2:
        st.header("Analytics")
        
        if st.session_state.analytics:
            a = st.session_state.analytics
            s = a['summary'].iloc[0]
            
            st.subheader("📈 Summary")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Rows", f"{s['total_rows']:,}")
            c2.metric("Total Errors", f"{int(s['total_errors']):,}")
            c3.metric("Avg/Row", f"{s['avg_errors']:.2f}")
            c4.metric("Avg Words", f"{int(s['avg_words']):,}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Spelling", f"{int(s['spelling_errors']):,}")
            c2.metric("Grammar", f"{int(s['grammar_errors']):,}")
            c3.metric("Punctuation", f"{int(s['punctuation_errors']):,}")
            c4.metric("Error-Free", f"{s['error_free_pct']:.1f}%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Error Types")
                st.dataframe(a['breakdown'], width='stretch', hide_index=True)
                st.bar_chart(a['breakdown'].set_index('type')['count'])
            
            with col2:
                st.subheader("📊 Distribution")
                st.dataframe(a['distribution'], width='stretch', hide_index=True)
                st.bar_chart(a['distribution'].set_index('range')['count'])
            
            st.markdown("---")
            
            st.subheader("🔝 Top 20 Errors")
            display_cols = ['total_errors', 'spelling_count', 'grammar_count', 
                          'punctuation_count', 'error_details', 'extracted_text']
            available = [c for c in display_cols if c in a['top_errors'].columns]
            st.dataframe(a['top_errors'][available], width='stretch', hide_index=True)
            
            st.markdown("---")
            
            st.subheader("📋 Full Data (first 100)")
            st.dataframe(a['full_data'].head(100), width='stretch', height=400)
            
        else:
            st.info("📤 Process file first")
    
    with tab3:
        st.header("💾 Download")
        
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            
            st.success(f"✅ Ready: {len(df):,} rows × {len(df.columns)} columns")
            
            st.markdown("### Includes:")
            st.markdown("""
            - All original columns
            - extracted_text
            - total_errors, spelling_count, grammar_count, punctuation_count
            - spelling_errors, grammar_errors, punctuation_errors
            - error_details, corrections
            - word_count, error_rate
            """)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### CSV")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"grammar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key='csv_btn'
                )
            
            with col2:
                st.markdown("#### Excel")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Analysis', index=False)
                    
                    if st.session_state.analytics:
                        a = st.session_state.analytics
                        a['summary'].to_excel(writer, sheet_name='Summary', index=False)
                        a['breakdown'].to_excel(writer, sheet_name='Breakdown', index=False)
                        a['distribution'].to_excel(writer, sheet_name='Distribution', index=False)
                    
                    workbook = writer.book
                    worksheet = writer.sheets['Analysis']
                    
                    for i, col in enumerate(df.columns):
                        max_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
                        worksheet.set_column(i, i, min(max_len, 50))
                    
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#4A90E2',
                        'font_color': 'white'
                    })
                    
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                
                st.download_button(
                    "📥 Download Excel",
                    output.getvalue(),
                    f"grammar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='xlsx_btn'
                )
            
            with col3:
                st.markdown("#### Parquet")
                if st.session_state.parquet_path:
                    with open(st.session_state.parquet_path, 'rb') as f:
                        pq_data = f.read()
                    
                    st.download_button(
                        "📥 Download Parquet",
                        pq_data,
                        f"grammar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                        "application/octet-stream",
                        key='pq_btn'
                    )
            
            st.markdown("---")
            
            st.subheader("📋 Columns")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str).values,
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values
            })
            st.dataframe(col_info, width='stretch', hide_index=True)
            
            st.subheader("🔍 Sample")
            st.dataframe(df.head(10), width='stretch')
            
        else:
            st.info("📤 Process file first")

if __name__ == "__main__":
    main()
