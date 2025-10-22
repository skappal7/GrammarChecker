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

# Branding
LOGO_URL = "https://raw.githubusercontent.com/skappal7/TextAnalyser/refs/heads/main/logo.png"
FOOTER = "Developed with Streamlit with 💗 by CE Team Innovation Lab 2025"

# Configure Streamlit
st.set_page_config(
    page_title="TextGuardian Pro",
    page_icon="🛡️",
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

# FIXED: Individual word boundary patterns for each misspelling
MISSPELLING_PATTERNS = {
    'recieve': re.compile(r'\brecieve\b', re.IGNORECASE),
    'occured': re.compile(r'\boccured\b', re.IGNORECASE),
    'seperate': re.compile(r'\bseperate\b', re.IGNORECASE),
    'definately': re.compile(r'\bdefinately\b', re.IGNORECASE),
    'accomodate': re.compile(r'\baccomodate\b', re.IGNORECASE),
    'occassion': re.compile(r'\boccassion\b', re.IGNORECASE),
    'wierd': re.compile(r'\bwierd\b', re.IGNORECASE),
    'untill': re.compile(r'\buntill\b', re.IGNORECASE),
    'thier': re.compile(r'\bthier\b', re.IGNORECASE),
    'wich': re.compile(r'\bwich\b', re.IGNORECASE),
    'becuase': re.compile(r'\bbecuase\b', re.IGNORECASE),
    'alot': re.compile(r'\balot\b', re.IGNORECASE),
    'tommorow': re.compile(r'\btommorow\b', re.IGNORECASE),
    'existance': re.compile(r'\bexistance\b', re.IGNORECASE),
    'appearence': re.compile(r'\bappearence\b', re.IGNORECASE),
    'begining': re.compile(r'\bbegining\b', re.IGNORECASE),
    'beleive': re.compile(r'\bbeleive\b', re.IGNORECASE),
    'calender': re.compile(r'\bcalender\b', re.IGNORECASE),
    'cemetary': re.compile(r'\bcemetary\b', re.IGNORECASE),
    'changable': re.compile(r'\bchangable\b', re.IGNORECASE),
    'collegue': re.compile(r'\bcollegue\b', re.IGNORECASE),
    'concious': re.compile(r'\bconcious\b', re.IGNORECASE),
    'embarrass': re.compile(r'\bembarrass\b', re.IGNORECASE),
    'enviroment': re.compile(r'\benviroment\b', re.IGNORECASE),
    'excercise': re.compile(r'\bexcercise\b', re.IGNORECASE),
    'fourty': re.compile(r'\bfourty\b', re.IGNORECASE),
    'goverment': re.compile(r'\bgoverment\b', re.IGNORECASE),
    'guarentee': re.compile(r'\bguarentee\b', re.IGNORECASE),
    'harrass': re.compile(r'\bharrass\b', re.IGNORECASE),
    'higeine': re.compile(r'\bhigeine\b', re.IGNORECASE),
    'ignorence': re.compile(r'\bignorence\b', re.IGNORECASE),
    'imediately': re.compile(r'\bimediately\b', re.IGNORECASE),
    'independant': re.compile(r'\bindependant\b', re.IGNORECASE),
    'instresting': re.compile(r'\binstresting\b', re.IGNORECASE),
    'liason': re.compile(r'\bliason\b', re.IGNORECASE),
    'libary': re.compile(r'\blibary\b', re.IGNORECASE),
    'maintainance': re.compile(r'\bmaintainance\b', re.IGNORECASE),
    'mispell': re.compile(r'\bmispell\b', re.IGNORECASE),
    'neccessary': re.compile(r'\bneccessary\b', re.IGNORECASE),
    'noticable': re.compile(r'\bnoticable\b', re.IGNORECASE),
    'occurance': re.compile(r'\boccurance\b', re.IGNORECASE),
    'persistant': re.compile(r'\bpersistant\b', re.IGNORECASE),
    'posession': re.compile(r'\bposession\b', re.IGNORECASE),
    'prefered': re.compile(r'\bprefered\b', re.IGNORECASE),
    'priviledge': re.compile(r'\bpriviledge\b', re.IGNORECASE),
    'reccomend': re.compile(r'\breccomend\b', re.IGNORECASE),
    'refered': re.compile(r'\brefered\b', re.IGNORECASE),
    'religous': re.compile(r'\breligous\b', re.IGNORECASE),
    'rythm': re.compile(r'\brythm\b', re.IGNORECASE),
    'sieze': re.compile(r'\bsieze\b', re.IGNORECASE),
    'succesful': re.compile(r'\bsuccesful\b', re.IGNORECASE),
    'supercede': re.compile(r'\bsupercede\b', re.IGNORECASE),
    'suprise': re.compile(r'\bsuprise\b', re.IGNORECASE),
    'temperture': re.compile(r'\btemperture\b', re.IGNORECASE),
    'tendancy': re.compile(r'\btendancy\b', re.IGNORECASE),
    'truely': re.compile(r'\btruely\b', re.IGNORECASE),
    'unforseen': re.compile(r'\bunforseen\b', re.IGNORECASE),
    'unnecesary': re.compile(r'\bunnecesary\b', re.IGNORECASE),
}

CONTRACTION_PATTERN = re.compile(r'\b(dont|wont|cant|shouldnt|wouldnt|couldnt|didnt|doesnt|isnt|arent|wasnt|werent|havent|hasnt|hadnt)\b')
DOUBLE_SPACE = re.compile(r'\s{2,}')
SPACE_BEFORE_PUNCT = re.compile(r'\s+([.!?,;:])')
NO_SPACE_AFTER_PUNCT = re.compile(r'([.!?,;:])([A-Za-z])')
MISSING_CAPITAL = re.compile(r'([.!?]\s+)([a-z])')
ARTICLE_A_VOWEL = re.compile(r'\ba\s+[aeiouAEIOU]\w+')
ARTICLE_AN_CONSONANT = re.compile(r'\ban\s+[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w+')
SUBJECT_VERB_ERROR = re.compile(r'\b(he|she|it)\s+(were|are)\b|\b(they|we|you)\s+(was|is)\b', re.IGNORECASE)

# NEW: Additional grammar patterns
DOUBLE_NEGATIVES = re.compile(r"\b(don't|didn't|doesn't|haven't|hasn't|won't|wouldn't)\s+\w+\s+(no|nothing|nobody|never|none)\b", re.IGNORECASE)
ITS_VS_ITS = re.compile(r"\bits\s+[a-z]+ing\b|\bits\s+(a|an|the)\b", re.IGNORECASE)
YOUR_VS_YOURE = re.compile(r"\byour\s+(a|an|the|very|so|really)\b", re.IGNORECASE)
THEN_VS_THAN = re.compile(r"\b(better|worse|more|less|rather)\s+then\b", re.IGNORECASE)

COMMON_MISSPELLINGS = {
    'recieve': 'receive', 'occured': 'occurred', 'seperate': 'separate',
    'definately': 'definitely', 'accomodate': 'accommodate', 'wierd': 'weird',
    'untill': 'until', 'thier': 'their', 'wich': 'which', 'becuase': 'because',
    'alot': 'a lot', 'occassion': 'occasion', 'tommorow': 'tomorrow',
    'existance': 'existence', 'appearence': 'appearance', 'begining': 'beginning',
    'beleive': 'believe', 'calender': 'calendar', 'collegue': 'colleague',
    'cemetary': 'cemetery', 'changable': 'changeable', 'concious': 'conscious',
    'embarrass': 'embarrass', 'enviroment': 'environment', 'excercise': 'exercise',
    'fourty': 'forty', 'goverment': 'government', 'guarentee': 'guarantee',
    'harrass': 'harass', 'higeine': 'hygiene', 'ignorence': 'ignorance',
    'imediately': 'immediately', 'independant': 'independent', 'instresting': 'interesting',
    'liason': 'liaison', 'libary': 'library', 'maintainance': 'maintenance',
    'mispell': 'misspell', 'neccessary': 'necessary', 'noticable': 'noticeable',
    'occurance': 'occurrence', 'persistant': 'persistent', 'posession': 'possession',
    'prefered': 'preferred', 'priviledge': 'privilege', 'reccomend': 'recommend',
    'refered': 'referred', 'religous': 'religious', 'rythm': 'rhythm',
    'sieze': 'seize', 'succesful': 'successful', 'supercede': 'supersede',
    'suprise': 'surprise', 'temperture': 'temperature', 'tendancy': 'tendency',
    'truely': 'truly', 'unforseen': 'unforeseen', 'unnecesary': 'unnecessary'
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
    """Check grammar - OPTIMIZED with fixed spelling detection"""
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
    
    # FIXED: Check each misspelling pattern individually to catch ALL occurrences
    spelling_found = []
    for misspelled, pattern in MISSPELLING_PATTERNS.items():
        matches = pattern.findall(text)
        for match in matches:
            spelling_found.append(match)
            errors.append(f"SPELLING:{match}")
            if misspelled.lower() in COMMON_MISSPELLINGS:
                fix = COMMON_MISSPELLINGS[misspelled.lower()]
                corrections.append(f"{match}→{fix}")
                details.append(f"Misspelled '{match}' → '{fix}'")
    
    # Grammar checks
    grammar = []
    
    # Contractions
    contractions = CONTRACTION_PATTERN.findall(text)
    if contractions:
        for c in contractions[:3]:
            grammar.append(f"Missing apostrophe: {c}")
            errors.append(f"GRAMMAR:apostrophe_{c}")
            details.append(f"Missing apostrophe in '{c}'")
    
    # Articles
    if ARTICLE_A_VOWEL.search(text):
        grammar.append("Use 'an' before vowel")
        errors.append("GRAMMAR:article_a_vowel")
        details.append("Use 'an' before vowel sound")
    
    if ARTICLE_AN_CONSONANT.search(text):
        grammar.append("Use 'a' before consonant")
        errors.append("GRAMMAR:article_an_consonant")
        details.append("Use 'a' before consonant sound")
    
    # Subject-verb agreement
    if SUBJECT_VERB_ERROR.search(text):
        grammar.append("Subject-verb disagreement")
        errors.append("GRAMMAR:subject_verb")
        details.append("Subject-verb disagreement")
    
    # NEW: Additional grammar checks
    if DOUBLE_NEGATIVES.search(text):
        grammar.append("Double negative")
        errors.append("GRAMMAR:double_negative")
        details.append("Avoid double negatives")
    
    if ITS_VS_ITS.search(text):
        grammar.append("Possible its/it's confusion")
        errors.append("GRAMMAR:its_its")
        details.append("Check its vs it's usage")
    
    if YOUR_VS_YOURE.search(text):
        grammar.append("Possible your/you're confusion")
        errors.append("GRAMMAR:your_youre")
        details.append("Check your vs you're usage")
    
    if THEN_VS_THAN.search(text):
        grammar.append("Then/than confusion")
        errors.append("GRAMMAR:then_than")
        details.append("Use 'than' for comparisons")
    
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
        'spelling_count': len(spelling_found),
        'grammar_count': len(grammar),
        'punctuation_count': len(punctuation),
        'spelling_errors': ', '.join(spelling_found[:10]),
        'grammar_errors': ' | '.join(grammar[:5]),
        'punctuation_errors': ' | '.join(punctuation[:5]),
        'error_details': ' || '.join(details[:15]),
        'corrections': '; '.join(corrections[:10]),
        'word_count': count_words(text)
    }

def process_batch(texts):
    """Process batch"""
    return [check_grammar(t) for t in texts]

def process_file(df, text_col, workers=None):
    """Process file with multiprocessing"""
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)
    
    with st.spinner("Extracting text..."):
        df['extracted_text'] = df[text_col].apply(extract_agent)
    
    texts = df['extracted_text'].tolist()
    batch_size = max(100, len(texts) // (workers * 4))
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    with st.spinner(f"Processing {len(texts):,} texts with {workers} workers..."):
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_batch, batches))
    
    all_results = [r for batch in results for r in batch]
    
    result_df = pd.DataFrame(all_results)
    for col in result_df.columns:
        df[col] = result_df[col]
    
    df['error_rate'] = (df['total_errors'] / df['word_count'].replace(0, 1) * 100).round(2)
    
    pq_path = save_parquet(df)
    
    return df, pq_path

def create_analytics(pq_path):
    """Create analytics with DuckDB"""
    conn = duckdb.connect(':memory:')
    
    summary = conn.execute(f"""
        SELECT
            COUNT(*) as total_rows,
            SUM(total_errors) as total_errors,
            AVG(total_errors) as avg_errors,
            SUM(spelling_count) as spelling_errors,
            SUM(grammar_count) as grammar_errors,
            SUM(punctuation_count) as punctuation_errors,
            AVG(word_count) as avg_words,
            COUNT(CASE WHEN total_errors = 0 THEN 1 END) * 100.0 / COUNT(*) as error_free_pct
        FROM read_parquet('{pq_path}')
    """).df()
    
    breakdown = conn.execute(f"""
        SELECT 'Spelling' as type, SUM(spelling_count) as count FROM read_parquet('{pq_path}')
        UNION ALL
        SELECT 'Grammar', SUM(grammar_count) FROM read_parquet('{pq_path}')
        UNION ALL
        SELECT 'Punctuation', SUM(punctuation_count) FROM read_parquet('{pq_path}')
        ORDER BY count DESC
    """).df()
    
    distribution = conn.execute(f"""
        SELECT 
            CASE 
                WHEN total_errors = 0 THEN '0'
                WHEN total_errors <= 5 THEN '1-5'
                WHEN total_errors <= 10 THEN '6-10'
                WHEN total_errors <= 20 THEN '11-20'
                ELSE '21+'
            END as range,
            COUNT(*) as count
        FROM read_parquet('{pq_path}')
        GROUP BY range
        ORDER BY 
            CASE range
                WHEN '0' THEN 0
                WHEN '1-5' THEN 1
                WHEN '6-10' THEN 2
                WHEN '11-20' THEN 3
                ELSE 4
            END
    """).df()
    
    top_errors = conn.execute(f"""
        SELECT *
        FROM read_parquet('{pq_path}')
        WHERE total_errors > 0
        ORDER BY total_errors DESC
        LIMIT 20
    """).df()
    
    full_data = conn.execute(f"""
        SELECT * FROM read_parquet('{pq_path}')
        ORDER BY total_errors DESC
    """).df()
    
    conn.close()
    
    return {
        'summary': summary,
        'breakdown': breakdown,
        'distribution': distribution,
        'top_errors': top_errors,
        'full_data': full_data
    }

def main():
    # Header with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(LOGO_URL, width=100)
    with col2:
        st.title("🛡️ TextGuardian Pro")
        st.markdown("**Enterprise-Grade Grammar & Quality Analytics Platform**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📊 Analytics", "💾 Download", "🧪 Live Test"])
    
    with tab1:
        st.header("Upload & Process")
        
        with st.sidebar:
            st.header("⚙️ Settings")
            workers = st.slider("Worker Threads", 1, mp.cpu_count(), max(1, mp.cpu_count() - 1))
            st.info(f"💻 {mp.cpu_count()} CPUs available")
            
            with st.expander("ℹ️ Features"):
                st.markdown("""
                **Checks:**
                - 50+ common misspellings
                - Missing apostrophes
                - Article errors (a/an)
                - Subject-verb agreement
                - Double negatives
                - Common confusions (its/it's, your/you're, then/than)
                - Punctuation spacing
                - Capitalization
                
                **Performance:**
                - Multi-threaded processing
                - Regex-based (no heavy NLP)
                - Parquet storage
                - DuckDB analytics
                """)
        
        uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file:
            try:
                with st.spinner("Loading..."):
                    df = load_file(uploaded_file)
                
                st.success(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
                
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
                st.dataframe(a['breakdown'], use_container_width=True, hide_index=True)
                st.bar_chart(a['breakdown'].set_index('type')['count'])
            
            with col2:
                st.subheader("📊 Distribution")
                st.dataframe(a['distribution'], use_container_width=True, hide_index=True)
                st.bar_chart(a['distribution'].set_index('range')['count'])
            
            st.markdown("---")
            
            st.subheader("🔝 Top 20 Errors")
            display_cols = ['total_errors', 'spelling_count', 'grammar_count', 
                          'punctuation_count', 'error_details', 'extracted_text']
            available = [c for c in display_cols if c in a['top_errors'].columns]
            st.dataframe(a['top_errors'][available], use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            st.subheader("📋 Full Data (first 100)")
            st.dataframe(a['full_data'].head(100), use_container_width=True, height=400)
            
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
            st.dataframe(col_info, use_container_width=True, hide_index=True)
            
            st.subheader("🔍 Sample")
            st.dataframe(df.head(10), use_container_width=True)
            
        else:
            st.info("📤 Process file first")
    
    # NEW: Live testing tab
    with tab4:
        st.header("🧪 Live Grammar Test")
        st.markdown("Test the grammar checker in real-time")
        
        test_text = st.text_area(
            "Enter text to check:",
            value="I recieve alot of emails becuase people dont know wich address to use. Its definately a problem.",
            height=150
        )
        
        if st.button("✅ Check Grammar", type="primary"):
            if test_text:
                result = check_grammar(test_text)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Errors", result['total_errors'])
                col2.metric("Spelling", result['spelling_count'])
                col3.metric("Grammar", result['grammar_count'])
                
                if result['total_errors'] > 0:
                    st.markdown("---")
                    
                    if result['spelling_errors']:
                        st.error("**Spelling Errors:**")
                        st.write(result['spelling_errors'])
                    
                    if result['corrections']:
                        st.success("**Suggested Corrections:**")
                        st.write(result['corrections'])
                    
                    if result['grammar_errors']:
                        st.warning("**Grammar Issues:**")
                        st.write(result['grammar_errors'])
                    
                    if result['punctuation_errors']:
                        st.info("**Punctuation Issues:**")
                        st.write(result['punctuation_errors'])
                    
                    st.markdown("---")
                    st.subheader("📝 All Details")
                    st.write(result['error_details'])
                else:
                    st.success("✅ No errors found!")
            else:
                st.warning("Please enter some text to check")
    
    # Footer
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #666; padding: 20px;'>{FOOTER}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
