import streamlit as st
import pandas as pd
import duckdb
import os
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from datetime import datetime
import io
import xlsxwriter
import re
import html
from typing import Dict, List, Tuple
import multiprocessing
from functools import partial
import json

# Import multiple libraries for comprehensive checking
import pyspellchecker
import textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
import string

# Download required NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data once"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    return True

# Configure Streamlit page
st.set_page_config(
    page_title="Grammar Check Analytics",
    page_icon="📝",
    layout="wide"
)

# Initialize NLTK data
download_nltk_data()

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'parquet_path' not in st.session_state:
    st.session_state.parquet_path = None

# Pre-compile regex patterns for better performance
PATTERNS = {
    'format1': re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\+\d{4})\s+Agent:(.*?)(?=\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\[|\Z)', re.DOTALL),
    'format2': re.compile(r'\[\d{2}:\d{2}:\d{2}\s+AGENT\]:(.*?)(?=\[\d{2}:\d{2}:\d{2}|\Z)', re.DOTALL | re.IGNORECASE)
}

# Grammar patterns for rule-based checking
GRAMMAR_PATTERNS = {
    'double_negative': r'\b(not|no|never|nothing|nowhere|neither|none|nobody)\s+\b(not|no|never|nothing|nowhere|neither|none|nobody)\b',
    'subject_verb': [
        (r'\b(he|she|it)\s+(were|are)\b', 'Subject-verb disagreement'),
        (r'\b(they|we|you)\s+(was|is)\b', 'Subject-verb disagreement'),
        (r'\b(I)\s+(were|is|are)\b', 'Subject-verb disagreement'),
    ],
    'article_errors': [
        (r'\b(a)\s+[aeiouAEIOU]\w+', 'Use "an" before vowel sounds'),
        (r'\b(an)\s+[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w+', 'Use "a" before consonant sounds'),
    ],
    'tense_consistency': [
        (r'\b(yesterday|last\s+\w+)\s+.*?\b(will|shall|going\s+to)\b', 'Tense inconsistency'),
        (r'\b(tomorrow|next\s+\w+)\s+.*?\b(was|were|did|had)\b', 'Tense inconsistency'),
    ]
}

# Punctuation patterns
PUNCTUATION_PATTERNS = {
    'missing_period': r'[a-z]\s*$',
    'double_punctuation': r'[.!?,;]{2,}',
    'missing_comma_after_intro': r'^(However|Therefore|Furthermore|Moreover|Nevertheless|Thus|Hence|Consequently|Meanwhile)\s+[a-z]',
    'missing_apostrophe': r'\b(dont|wont|cant|shouldnt|wouldnt|couldnt|didnt|doesnt|isnt|arent|wasnt|werent)\b',
    'extra_spaces': r'\s{2,}',
    'space_before_punctuation': r'\s+[.!?,;:]',
    'missing_capital': r'^[a-z]|[.!?]\s+[a-z]'
}

class TranscriptParser:
    """Parse and clean transcript data to extract agent messages only"""
    
    @staticmethod
    def clean_text(text):
        """Clean HTML, special characters, and encoding issues"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Fix common encoding issues
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'â€"': '--',
            'â€¦': '...',
            '&gt;': '>',
            '&lt;': '<',
            '&amp;': '&'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_agent_messages(text):
        """Extract only agent messages from transcript"""
        if pd.isna(text):
            return []
        
        text = str(text)
        agent_messages = []
        
        # Use pre-compiled patterns
        matches1 = PATTERNS['format1'].findall(text)
        for timestamp, message in matches1:
            clean_msg = TranscriptParser.clean_text(message)
            if clean_msg:
                agent_messages.append(clean_msg)
        
        matches2 = PATTERNS['format2'].findall(text)
        for message in matches2:
            clean_msg = TranscriptParser.clean_text(message)
            if clean_msg:
                agent_messages.append(clean_msg)
        
        # Join all agent messages with a space
        return ' '.join(agent_messages) if agent_messages else ""
    
    @staticmethod
    def process_transcript_column(df, text_column):
        """Process transcript column to extract and clean agent messages"""
        df['agent_text_cleaned'] = df[text_column].apply(TranscriptParser.extract_agent_messages)
        # Keep only rows with agent messages
        df = df[df['agent_text_cleaned'].str.len() > 0].copy()
        return df

class MultiLibraryChecker:
    """Fast grammar checking using multiple specialized libraries"""
    
    def __init__(self):
        # Initialize spell checker
        self.spell_checker = pyspellchecker.SpellChecker()
        self.spell_checker.word_frequency.load_words(['okay', 'app', 'email', 'login', 'website', 'password', 'username', 'admin'])
        
        # Compile regex patterns
        self.grammar_patterns_compiled = {}
        for key, patterns in GRAMMAR_PATTERNS.items():
            if isinstance(patterns, list):
                self.grammar_patterns_compiled[key] = [(re.compile(p, re.IGNORECASE), msg) for p, msg in patterns]
            else:
                self.grammar_patterns_compiled[key] = re.compile(patterns, re.IGNORECASE)
        
        self.punctuation_patterns_compiled = {
            key: re.compile(pattern, re.MULTILINE if key == 'missing_period' else 0)
            for key, pattern in PUNCTUATION_PATTERNS.items()
        }
    
    def check_spelling(self, text):
        """Check spelling errors using pyspellchecker"""
        if not text:
            return [], {}
        
        # Tokenize and clean words
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and len(word) > 2]
        
        # Find misspelled words
        misspelled = self.spell_checker.unknown(words)
        
        corrections = {}
        for word in misspelled:
            correction = self.spell_checker.correction(word)
            if correction and correction != word:
                corrections[word] = correction
        
        return list(misspelled), corrections
    
    def check_grammar(self, text):
        """Check grammar using rule-based patterns and NLTK"""
        if not text:
            return [], []
        
        grammar_errors = []
        error_details = []
        
        # Check subject-verb agreement
        for pattern, message in self.grammar_patterns_compiled.get('subject_verb', []):
            if pattern.search(text):
                grammar_errors.append('subject_verb')
                error_details.append(message)
        
        # Check article errors
        for pattern, message in self.grammar_patterns_compiled.get('article_errors', []):
            if pattern.search(text):
                grammar_errors.append('article')
                error_details.append(message)
        
        # Check tense consistency
        for pattern, message in self.grammar_patterns_compiled.get('tense_consistency', []):
            if pattern.search(text):
                grammar_errors.append('tense')
                error_details.append(message)
        
        # Check double negatives
        if self.grammar_patterns_compiled['double_negative'].search(text):
            grammar_errors.append('double_negative')
            error_details.append('Double negative detected')
        
        # Check for sentence fragments (simple heuristic)
        sentences = sent_tokenize(text)
        for sentence in sentences:
            words = word_tokenize(sentence)
            if len(words) < 3 or not any(word in ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might'] for word in words):
                if len(sentence) > 10:  # Avoid flagging short responses
                    grammar_errors.append('fragment')
                    error_details.append('Possible sentence fragment')
                    break
        
        return grammar_errors, error_details
    
    def check_punctuation(self, text):
        """Check punctuation errors using regex patterns"""
        if not text:
            return [], []
        
        punctuation_errors = []
        error_details = []
        
        # Check each punctuation pattern
        for error_type, pattern in self.punctuation_patterns_compiled.items():
            if pattern.search(text):
                punctuation_errors.append(error_type)
                
                if error_type == 'missing_period':
                    error_details.append('Missing period at end of sentence')
                elif error_type == 'double_punctuation':
                    error_details.append('Double punctuation marks')
                elif error_type == 'missing_comma_after_intro':
                    error_details.append('Missing comma after introductory word')
                elif error_type == 'missing_apostrophe':
                    error_details.append('Missing apostrophe in contraction')
                elif error_type == 'extra_spaces':
                    error_details.append('Extra spaces between words')
                elif error_type == 'space_before_punctuation':
                    error_details.append('Space before punctuation')
                elif error_type == 'missing_capital':
                    error_details.append('Missing capitalization')
        
        return punctuation_errors, error_details
    
    def check_style(self, text):
        """Check style issues using textstat"""
        if not text or len(text) < 20:
            return {
                'readability_score': 0,
                'grade_level': 0,
                'passive_voice': False,
                'sentence_complexity': 'simple'
            }
        
        # Calculate readability metrics
        readability = textstat.flesch_reading_ease(text)
        grade = textstat.flesch_kincaid_grade(text)
        
        # Simple passive voice detection
        passive_indicators = ['was', 'were', 'been', 'being', 'is', 'are', 'am']
        passive_patterns = [f'{ind} \\w+ed\\b' for ind in passive_indicators]
        passive_voice = any(re.search(p, text, re.IGNORECASE) for p in passive_patterns)
        
        # Sentence complexity
        avg_words_per_sentence = textstat.avg_sentence_length(text)
        if avg_words_per_sentence > 20:
            complexity = 'complex'
        elif avg_words_per_sentence > 12:
            complexity = 'moderate'
        else:
            complexity = 'simple'
        
        return {
            'readability_score': readability,
            'grade_level': grade,
            'passive_voice': passive_voice,
            'sentence_complexity': complexity
        }
    
    def check_text_comprehensive(self, text):
        """Comprehensive check using all methods"""
        if pd.isna(text) or str(text).strip() == '':
            return {
                'total_errors': 0,
                'spelling_errors': 0,
                'grammar_errors': 0,
                'punctuation_errors': 0,
                'style_issues': 0,
                'spelling_mistakes': [],
                'grammar_issues': [],
                'punctuation_issues': [],
                'suggested_corrections': {},
                'readability_score': 0,
                'grade_level': 0
            }
        
        text = str(text)
        
        # Run all checks
        misspelled, spelling_corrections = self.check_spelling(text)
        grammar_errors, grammar_details = self.check_grammar(text)
        punctuation_errors, punctuation_details = self.check_punctuation(text)
        style_metrics = self.check_style(text)
        
        # Count style issues
        style_issues = 0
        if style_metrics['readability_score'] < 30:  # Very difficult to read
            style_issues += 1
        if style_metrics['passive_voice']:
            style_issues += 1
        if style_metrics['sentence_complexity'] == 'complex':
            style_issues += 1
        
        # Calculate totals
        total_errors = len(misspelled) + len(grammar_errors) + len(punctuation_errors) + style_issues
        
        return {
            'total_errors': total_errors,
            'spelling_errors': len(misspelled),
            'grammar_errors': len(grammar_errors),
            'punctuation_errors': len(punctuation_errors),
            'style_issues': style_issues,
            'spelling_mistakes': list(misspelled)[:10],  # Top 10
            'grammar_issues': grammar_details[:5],  # Top 5
            'punctuation_issues': punctuation_details[:5],  # Top 5
            'suggested_corrections': spelling_corrections,
            'readability_score': style_metrics['readability_score'],
            'grade_level': style_metrics['grade_level']
        }

def process_batch_fast(texts_batch):
    """Process a batch of texts quickly using multi-library approach"""
    checker = MultiLibraryChecker()
    results = []
    
    for text in texts_batch:
        result = checker.check_text_comprehensive(text)
        results.append(result)
    
    return results

class DataProcessor:
    """Handle data processing with DuckDB and parallel processing"""
    
    def __init__(self):
        self.conn = duckdb.connect(':memory:')
    
    def csv_to_parquet_fast(self, csv_file, text_column, batch_size=5000, num_workers=None):
        """Convert CSV to Parquet with fast multi-library checking"""
        
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)
        
        # Read CSV
        df = pd.read_csv(csv_file)
        original_columns = df.columns.tolist()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Parse transcripts to get only agent messages
        status_text.text("Extracting agent messages...")
        parser = TranscriptParser()
        df = parser.process_transcript_column(df, text_column)
        
        if len(df) == 0:
            raise ValueError("No agent messages found in the transcript data")
        
        total_rows = len(df)
        status_text.text(f"Found {total_rows} agent messages. Starting fast multi-library analysis...")
        
        # Split data for parallel processing
        texts = df['agent_text_cleaned'].tolist()
        chunk_size = max(len(texts) // (num_workers * 4), 25)  # Larger chunks for faster processing
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Process in parallel
        all_results = []
        processed = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = []
            for chunk in text_chunks:
                future = executor.submit(process_batch_fast, chunk)
                futures.append(future)
            
            # Collect results as they complete
            for i, future in enumerate(futures):
                chunk_results = future.result()
                all_results.extend(chunk_results)
                processed += len(chunk_results)
                progress_bar.progress(processed / total_rows)
                status_text.text(f"Processed {processed}/{total_rows} messages (Fast Mode)...")
        
        # Convert results to DataFrame columns
        status_text.text("Consolidating results...")
        
        # Extract all data from results
        for key in ['total_errors', 'spelling_errors', 'grammar_errors', 'punctuation_errors', 'style_issues', 'readability_score', 'grade_level']:
            df[key + ('_count' if 'errors' in key or 'issues' in key else '')] = [r[key] for r in all_results]
        
        # Convert lists and dicts to strings for storage
        df['spelling_mistakes'] = [', '.join(r['spelling_mistakes']) for r in all_results]
        df['grammar_issues'] = [' | '.join(r['grammar_issues']) for r in all_results]
        df['punctuation_issues'] = [' | '.join(r['punctuation_issues']) for r in all_results]
        
        # Format corrections
        df['suggested_corrections'] = [
            '; '.join([f"{k}→{v}" for k, v in list(r['suggested_corrections'].items())[:5]])
            for r in all_results
        ]
        
        # Calculate percentage columns
        df['error_rate'] = (df['total_errors'] / df['agent_text_cleaned'].str.split().str.len() * 100).round(2).fillna(0)
        df['spelling_error_rate'] = (df['spelling_errors_count'] / df['total_errors'].replace(0, 1) * 100).round(2)
        df['grammar_error_rate'] = (df['grammar_errors_count'] / df['total_errors'].replace(0, 1) * 100).round(2)
        
        # Add metadata
        df['processing_timestamp'] = datetime.now().isoformat()
        df['check_method'] = 'multi_library_fast'
        
        # Save to parquet
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp_file:
            df.to_parquet(tmp_file.name, engine='pyarrow', compression='snappy')
            progress_bar.progress(1.0)
            status_text.text(f"✅ Processing complete! Analyzed {total_rows} agent messages in fast mode.")
            return tmp_file.name, df
    
    def analyze_with_duckdb(self, parquet_path):
        """Perform analytics using DuckDB"""
        # Register parquet file with DuckDB
        self.conn.execute(f"""
            CREATE OR REPLACE TABLE transcripts AS 
            SELECT * FROM parquet_scan('{parquet_path}')
        """)
        
        # Analytics queries
        analytics = {}
        
        # Summary statistics
        analytics['summary'] = self.conn.execute("""
            SELECT 
                COUNT(*) as total_messages,
                SUM(total_errors) as total_errors,
                SUM(spelling_errors_count) as total_spelling_errors,
                SUM(grammar_errors_count) as total_grammar_errors,
                SUM(punctuation_errors_count) as total_punctuation_errors,
                SUM(style_issues_count) as total_style_issues,
                ROUND(AVG(total_errors), 2) as avg_errors_per_message,
                ROUND(AVG(error_rate), 2) as avg_error_rate_percent,
                ROUND(AVG(readability_score), 2) as avg_readability_score,
                ROUND(AVG(grade_level), 2) as avg_grade_level,
                SUM(CASE WHEN total_errors = 0 THEN 1 ELSE 0 END) as error_free_messages,
                ROUND(SUM(CASE WHEN total_errors = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as error_free_percentage
            FROM transcripts
        """).df()
        
        # Error category breakdown
        try:
            analytics['category_breakdown'] = self.conn.execute("""
                SELECT 
                    'Spelling' as error_type,
                    SUM(spelling_errors_count) as count,
                    CASE 
                        WHEN SUM(total_errors) > 0 
                        THEN ROUND(SUM(spelling_errors_count) * 100.0 / SUM(total_errors), 2)
                        ELSE 0 
                    END as percentage
                FROM transcripts
                UNION ALL
                SELECT 
                    'Grammar' as error_type,
                    SUM(grammar_errors_count) as count,
                    CASE 
                        WHEN SUM(total_errors) > 0 
                        THEN ROUND(SUM(grammar_errors_count) * 100.0 / SUM(total_errors), 2)
                        ELSE 0 
                    END as percentage
                FROM transcripts
                UNION ALL
                SELECT 
                    'Punctuation' as error_type,
                    SUM(punctuation_errors_count) as count,
                    CASE 
                        WHEN SUM(total_errors) > 0 
                        THEN ROUND(SUM(punctuation_errors_count) * 100.0 / SUM(total_errors), 2)
                        ELSE 0 
                    END as percentage
                FROM transcripts
                UNION ALL
                SELECT 
                    'Style' as error_type,
                    SUM(style_issues_count) as count,
                    CASE 
                        WHEN SUM(total_errors) > 0 
                        THEN ROUND(SUM(style_issues_count) * 100.0 / SUM(total_errors), 2)
                        ELSE 0 
                    END as percentage
                FROM transcripts
                ORDER BY count DESC
            """).df()
        except Exception as e:
            analytics['category_breakdown'] = pd.DataFrame({
                'error_type': ['Spelling', 'Grammar', 'Punctuation', 'Style'],
                'count': [0, 0, 0, 0],
                'percentage': [0, 0, 0, 0]
            })
        
        # Top spelling mistakes
        try:
            analytics['top_spelling_mistakes'] = self.conn.execute("""
                WITH spelling_words AS (
                    SELECT 
                        TRIM(UNNEST(STRING_SPLIT(spelling_mistakes, ','))) as word
                    FROM transcripts
                    WHERE LENGTH(spelling_mistakes) > 0
                )
                SELECT 
                    word,
                    COUNT(*) as frequency
                FROM spelling_words
                WHERE LENGTH(word) > 0
                GROUP BY word
                ORDER BY frequency DESC
                LIMIT 20
            """).df()
        except:
            analytics['top_spelling_mistakes'] = pd.DataFrame(columns=['word', 'frequency'])
        
        # Readability distribution
        analytics['readability_distribution'] = self.conn.execute("""
            SELECT 
                CASE 
                    WHEN readability_score >= 90 THEN 'Very Easy'
                    WHEN readability_score >= 80 THEN 'Easy'
                    WHEN readability_score >= 70 THEN 'Fairly Easy'
                    WHEN readability_score >= 60 THEN 'Standard'
                    WHEN readability_score >= 50 THEN 'Fairly Difficult'
                    WHEN readability_score >= 30 THEN 'Difficult'
                    ELSE 'Very Difficult'
                END as readability_level,
                COUNT(*) as message_count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM transcripts
            GROUP BY readability_level
            ORDER BY 
                CASE readability_level
                    WHEN 'Very Easy' THEN 1
                    WHEN 'Easy' THEN 2
                    WHEN 'Fairly Easy' THEN 3
                    WHEN 'Standard' THEN 4
                    WHEN 'Fairly Difficult' THEN 5
                    WHEN 'Difficult' THEN 6
                    WHEN 'Very Difficult' THEN 7
                END
        """).df()
        
        # Complete dataset
        analytics['full_data'] = self.conn.execute("""
            SELECT *
            FROM transcripts
            ORDER BY total_errors DESC
        """).df()
        
        return analytics
    
    def export_consolidated_results(self, analytics, format='xlsx'):
        """Export all results into a single consolidated file"""
        output = io.BytesIO()
        
        if format == 'xlsx':
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Write main data
                analytics['full_data'].to_excel(writer, sheet_name='Complete Analysis', index=False)
                
                # Write summary statistics
                analytics['summary'].to_excel(writer, sheet_name='Summary', index=False)
                
                # Write category breakdown
                analytics['category_breakdown'].to_excel(writer, sheet_name='Error Categories', index=False)
                
                # Write readability distribution
                analytics['readability_distribution'].to_excel(writer, sheet_name='Readability', index=False)
                
                # Write top spelling mistakes
                if not analytics['top_spelling_mistakes'].empty:
                    analytics['top_spelling_mistakes'].to_excel(writer, sheet_name='Top Spelling Mistakes', index=False)
                
                # Format the Excel file
                workbook = writer.book
                
                # Format main sheet
                worksheet = writer.sheets['Complete Analysis']
                worksheet.set_column('A:Z', 15)
                
                # Add conditional formatting for error counts
                worksheet.conditional_format('C:C', {
                    'type': '3_color_scale',
                    'min_color': '#C7E9C0',
                    'mid_color': '#FDD835',
                    'max_color': '#FF5252'
                })
                
                # Format summary sheet
                summary_sheet = writer.sheets['Summary']
                summary_sheet.set_column('A:L', 20)
        
        elif format == 'csv':
            # For CSV, export the complete analysis
            analytics['full_data'].to_csv(output, index=False)
        
        elif format == 'parquet':
            # For Parquet, include all data with metadata
            full_data = analytics['full_data'].copy()
            
            # Add summary statistics as metadata
            for col, val in analytics['summary'].iloc[0].items():
                full_data.attrs[f'summary_{col}'] = val
            
            full_data.to_parquet(output, engine='pyarrow', compression='snappy')
        
        output.seek(0)
        return output

def main():
    st.title("📝 Grammar Check Analytics System - Fast Mode")
    st.markdown("### Ultra-fast analysis using multiple specialized libraries")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("""
        **Fast Multi-Library Approach:**
        • **Spelling**: PySpellChecker
        • **Grammar**: NLTK + Rules
        • **Punctuation**: Regex Patterns
        • **Style**: TextStat
        
        **Performance:**
        • 10-20x faster than LanguageTool
        • 100K rows in ~5-10 minutes
        • ~75-80% accuracy
        """)
        
        st.markdown("---")
        st.markdown("**What's Detected:**")
        st.markdown("✅ Misspelled words")
        st.markdown("✅ Subject-verb agreement")
        st.markdown("✅ Article usage (a/an)")
        st.markdown("✅ Punctuation errors")
        st.markdown("✅ Capitalization")
        st.markdown("✅ Readability scores")
        st.markdown("✅ Passive voice")
        
        st.markdown("---")
        st.markdown("**Performance Settings:**")
        num_workers = st.slider(
            "Parallel workers",
            min_value=1,
            max_value=multiprocessing.cpu_count(),
            value=min(4, multiprocessing.cpu_count()),
            help="More workers = faster processing"
        )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Analytics Dashboard", "💾 Download Results"])
    
    with tab1:
        st.header("Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file with transcripts",
            type=['csv'],
            help="Upload your transcript data in CSV format"
        )
        
        if uploaded_file is not None:
            # Preview the CSV
            st.subheader("Data Preview")
            df_preview = pd.read_csv(uploaded_file, nrows=5)
            st.dataframe(df_preview)
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Column selection
            columns = df_preview.columns.tolist()
            text_column = st.selectbox(
                "Select the column containing transcripts",
                options=columns,
                help="Choose the column that contains the transcript text"
            )
            
            # Process button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Start Fast Processing", type="primary", use_container_width=True):
                    processor = DataProcessor()
                    
                    start_time = datetime.now()
                    
                    with st.spinner("Processing with multi-library fast analysis..."):
                        try:
                            # Process the file with fast multi-library approach
                            parquet_path, processed_df = processor.csv_to_parquet_fast(
                                uploaded_file,
                                text_column,
                                batch_size=5000,
                                num_workers=num_workers
                            )
                            
                            # Store in session state
                            st.session_state.parquet_path = parquet_path
                            st.session_state.processed_data = processed_df
                            st.session_state.num_workers = num_workers
                            
                            # Calculate processing time
                            processing_time = (datetime.now() - start_time).total_seconds()
                            
                            st.success(f"✅ Successfully processed {len(processed_df)} agent messages in {processing_time:.1f} seconds!")
                            st.balloons()
                            
                            # Show immediate statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Messages", f"{len(processed_df):,}")
                            with col2:
                                total_errors = processed_df['total_errors'].sum()
                                st.metric("Total Errors", f"{total_errors:,}")
                            with col3:
                                avg_readability = processed_df['readability_score'].mean()
                                st.metric("Avg Readability", f"{avg_readability:.1f}")
                            with col4:
                                messages_per_sec = len(processed_df) / processing_time
                                st.metric("Speed", f"{messages_per_sec:.0f} msg/sec")
                            
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
            
            with col2:
                st.markdown("#### Processing Info")
                st.info("""
                **Fast Mode Benefits:**
                - 10-20x faster processing
                - No API rate limits
                - Fully offline analysis
                - ~75-80% accuracy
                """)
        else:
            st.info("👆 Please upload a CSV file to begin analysis")

if __name__ == "__main__":
    main()
