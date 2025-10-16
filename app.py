import streamlit as st
import pandas as pd
import duckdb
import os
from pathlib import Path
import tempfile
import language_tool_python
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

# Configure Streamlit page
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
if 'grammar_tool' not in st.session_state:
    # Initialize grammar tool (using LanguageTool)
    @st.cache_resource
    def get_grammar_tool():
        return language_tool_python.LanguageTool('en-US')
    st.session_state.grammar_tool = get_grammar_tool()

# Pre-compile regex patterns for better performance
PATTERNS = {
    'format1': re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\+\d{4})\s+Agent:(.*?)(?=\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\[|\Z)', re.DOTALL),
    'format2': re.compile(r'\[\d{2}:\d{2}:\d{2}\s+AGENT\]:(.*?)(?=\[\d{2}:\d{2}:\d{2}|\Z)', re.DOTALL | re.IGNORECASE)
}

# Error category mappings
ERROR_CATEGORIES = {
    'spelling': [
        'MORFOLOGIK_RULE_EN_US', 'HUNSPELL_RULE', 'SPELLING_RULE', 
        'EN_SPELLING_RULE', 'TYPOS', 'CONFUSED_WORDS'
    ],
    'grammar': [
        'AGREEMENT_ERRORS', 'VERB_FORM', 'SUBJECT_VERB_AGREEMENT',
        'SENTENCE_FRAGMENT', 'WRONG_VERB_FORM', 'HE_VERB_AGR', 
        'SINGULAR_PLURAL', 'ARTICLE_MISSING', 'DETERMINER_AGREEMENT'
    ],
    'punctuation': [
        'COMMA_PARENTHESIS_WHITESPACE', 'DOUBLE_PUNCTUATION', 
        'UPPERCASE_SENTENCE_START', 'MISSING_COMMA', 'COMMA_SPACING',
        'PUNCTUATION_PARAGRAPH', 'EXCLAMATION_MULTIPLE', 'PERIOD_MISSING'
    ],
    'style': [
        'WORDINESS', 'PASSIVE_VOICE', 'REDUNDANCY', 'CLICHES',
        'INFORMAL_TONE', 'REPETITIVE_WORD', 'SENTENCE_LENGTH'
    ]
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

class EnhancedGrammarChecker:
    """Handle grammar checking with detailed error categorization"""
    
    def __init__(self, tool):
        self.tool = tool
    
    def categorize_error(self, rule_id):
        """Categorize error based on rule ID"""
        rule_id_upper = rule_id.upper()
        for category, rules in ERROR_CATEGORIES.items():
            for rule in rules:
                if rule in rule_id_upper:
                    return category
        return 'other'
    
    def check_text_detailed(self, text):
        """Check text for grammar errors with detailed categorization"""
        if pd.isna(text) or str(text).strip() == '':
            return {
                'total_errors': 0,
                'spelling_errors': 0,
                'grammar_errors': 0,
                'punctuation_errors': 0,
                'style_errors': 0,
                'spelling_mistakes': [],
                'grammar_issues': [],
                'punctuation_issues': [],
                'all_corrections': {}
            }
        
        try:
            matches = self.tool.check(str(text))
            
            result = {
                'total_errors': len(matches),
                'spelling_errors': 0,
                'grammar_errors': 0,
                'punctuation_errors': 0,
                'style_errors': 0,
                'spelling_mistakes': [],
                'grammar_issues': [],
                'punctuation_issues': [],
                'all_corrections': {}
            }
            
            for match in matches:
                category = self.categorize_error(match.ruleId)
                error_text = match.matchedText
                suggestions = match.replacements[:3] if match.replacements else []
                
                if category == 'spelling':
                    result['spelling_errors'] += 1
                    result['spelling_mistakes'].append(error_text)
                    if error_text:
                        result['all_corrections'][error_text] = suggestions
                elif category == 'grammar':
                    result['grammar_errors'] += 1
                    result['grammar_issues'].append(f"{error_text} ({match.message[:50]}...)")
                elif category == 'punctuation':
                    result['punctuation_errors'] += 1
                    result['punctuation_issues'].append(match.message[:50])
                elif category == 'style':
                    result['style_errors'] += 1
                else:
                    # Count as grammar if uncategorized
                    result['grammar_errors'] += 1
            
            return result
            
        except Exception as e:
            return {
                'total_errors': 0,
                'spelling_errors': 0,
                'grammar_errors': 0,
                'punctuation_errors': 0,
                'style_errors': 0,
                'spelling_mistakes': [],
                'grammar_issues': [],
                'punctuation_issues': [],
                'all_corrections': {},
                'error': str(e)
            }

def process_batch_parallel(texts_batch, grammar_tool_params):
    """Process a batch of texts in parallel - function for multiprocessing"""
    # Create a new grammar tool instance for this process
    tool = language_tool_python.LanguageTool('en-US')
    checker = EnhancedGrammarChecker(tool)
    
    results = []
    for text in texts_batch:
        result = checker.check_text_detailed(text)
        results.append(result)
    
    tool.close()
    return results

class DataProcessor:
    """Handle data processing with DuckDB and parallel processing"""
    
    def __init__(self):
        self.conn = duckdb.connect(':memory:')
    
    def csv_to_parquet_parallel(self, csv_file, text_column, batch_size=5000, num_workers=None):
        """Convert CSV to Parquet with parallel grammar checking"""
        
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
        status_text.text(f"Found {total_rows} agent messages. Starting parallel grammar analysis...")
        
        # Split data for parallel processing
        texts = df['agent_text_cleaned'].tolist()
        chunk_size = max(len(texts) // (num_workers * 4), 10)  # Smaller chunks for better progress tracking
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Process in parallel using multiprocessing
        all_results = []
        processed = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = []
            for chunk in text_chunks:
                future = executor.submit(process_batch_parallel, chunk, None)
                futures.append(future)
            
            # Collect results as they complete
            for i, future in enumerate(futures):
                chunk_results = future.result()
                all_results.extend(chunk_results)
                processed += len(chunk_results)
                progress_bar.progress(processed / total_rows)
                status_text.text(f"Processed {processed}/{total_rows} messages...")
        
        # Convert results to DataFrame columns
        status_text.text("Consolidating results...")
        
        # Extract all data from results
        result_data = {
            'total_errors': [],
            'spelling_errors_count': [],
            'grammar_errors_count': [],
            'punctuation_errors_count': [],
            'style_errors_count': [],
            'spelling_mistakes': [],
            'grammar_issues': [],
            'punctuation_issues': [],
            'suggested_corrections': []
        }
        
        for result in all_results:
            result_data['total_errors'].append(result['total_errors'])
            result_data['spelling_errors_count'].append(result['spelling_errors'])
            result_data['grammar_errors_count'].append(result['grammar_errors'])
            result_data['punctuation_errors_count'].append(result['punctuation_errors'])
            result_data['style_errors_count'].append(result['style_errors'])
            
            # Convert lists to strings for storage
            result_data['spelling_mistakes'].append(', '.join(result['spelling_mistakes'][:10]))  # Top 10
            result_data['grammar_issues'].append(' | '.join(result['grammar_issues'][:5]))  # Top 5
            result_data['punctuation_issues'].append(' | '.join(result['punctuation_issues'][:5]))  # Top 5
            
            # Format corrections as string
            corrections = []
            for word, suggestions in list(result['all_corrections'].items())[:5]:
                if suggestions:
                    corrections.append(f"{word}→{suggestions[0]}")
            result_data['suggested_corrections'].append('; '.join(corrections))
        
        # Add all results to DataFrame
        for key, values in result_data.items():
            df[key] = values
        
        # Calculate percentage columns
        df['error_rate'] = (df['total_errors'] / df['agent_text_cleaned'].str.split().str.len() * 100).round(2)
        df['spelling_error_rate'] = (df['spelling_errors_count'] / df['total_errors'].replace(0, 1) * 100).round(2)
        df['grammar_error_rate'] = (df['grammar_errors_count'] / df['total_errors'].replace(0, 1) * 100).round(2)
        
        # Add metadata
        df['processing_timestamp'] = datetime.now().isoformat()
        
        # Save to parquet
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp_file:
            df.to_parquet(tmp_file.name, engine='pyarrow', compression='snappy')
            progress_bar.progress(1.0)
            status_text.text(f"✅ Processing complete! Analyzed {total_rows} agent messages.")
            return tmp_file.name, df
    
    def analyze_with_duckdb(self, parquet_path):
        """Perform analytics using DuckDB with enhanced queries"""
        # Register parquet file with DuckDB
        self.conn.execute(f"""
            CREATE OR REPLACE TABLE transcripts AS 
            SELECT * FROM parquet_scan('{parquet_path}')
        """)
        
        # Analytics queries
        analytics = {}
        
        # Enhanced total statistics - simplified to avoid errors
        analytics['summary'] = self.conn.execute("""
            SELECT 
                COUNT(*) as total_messages,
                SUM(total_errors) as total_errors,
                SUM(spelling_errors_count) as total_spelling_errors,
                SUM(grammar_errors_count) as total_grammar_errors,
                SUM(punctuation_errors_count) as total_punctuation_errors,
                SUM(style_errors_count) as total_style_errors,
                ROUND(AVG(total_errors), 2) as avg_errors_per_message,
                ROUND(AVG(error_rate), 2) as avg_error_rate_percent,
                SUM(CASE WHEN total_errors = 0 THEN 1 ELSE 0 END) as error_free_messages,
                ROUND(SUM(CASE WHEN total_errors = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as error_free_percentage
            FROM transcripts
        """).df()
        
        # Error category breakdown - simplified
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
                    SUM(style_errors_count) as count,
                    CASE 
                        WHEN SUM(total_errors) > 0 
                        THEN ROUND(SUM(style_errors_count) * 100.0 / SUM(total_errors), 2)
                        ELSE 0 
                    END as percentage
                FROM transcripts
                ORDER BY count DESC
            """).df()
        except Exception as e:
            # Fallback
            analytics['category_breakdown'] = pd.DataFrame({
                'error_type': ['Spelling', 'Grammar', 'Punctuation', 'Style'],
                'count': [0, 0, 0, 0],
                'percentage': [0, 0, 0, 0]
            })
        
        # Top spelling mistakes - simplified query
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
            # Fallback if string_split doesn't work
            analytics['top_spelling_mistakes'] = pd.DataFrame(columns=['word', 'frequency'])
        
        # Complete dataset with all details
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
                summary_sheet.set_column('A:J', 20)
        
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
    st.title("📝 Grammar Check Analytics System - Agent Transcripts")
    st.markdown("### High-performance analysis of agent transcripts with detailed error categorization")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("""
        **Features:**
        ✅ Parallel processing for speed
        ✅ Detailed error categorization
        ✅ Spelling vs Grammar breakdown
        ✅ Actual error text extraction
        ✅ Single consolidated output file
        ✅ Agent-only message analysis
        """)
        
        st.markdown("---")
        st.markdown("**Error Categories:**")
        st.markdown("• **Spelling**: Misspelled words")
        st.markdown("• **Grammar**: Verb forms, agreements")
        st.markdown("• **Punctuation**: Commas, periods")
        st.markdown("• **Style**: Wordiness, passive voice")
        
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
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                batch_size = st.number_input(
                    "Batch size for processing",
                    min_value=1000,
                    max_value=10000,
                    value=5000,
                    step=1000,
                    help="Larger batch sizes use more memory but may be faster"
                )
            
            # Process button
            if st.button("🚀 Start Processing", type="primary"):
                processor = DataProcessor()
                
                with st.spinner("Processing with parallel grammar analysis..."):
                    try:
                        # Process the file with parallel processing
                        parquet_path, processed_df = processor.csv_to_parquet_parallel(
                            uploaded_file,
                            text_column,
                            batch_size,
                            num_workers
                        )
                        
                        # Store in session state
                        st.session_state.parquet_path = parquet_path
                        st.session_state.processed_data = processed_df
                        
                        st.success(f"✅ Successfully processed {len(processed_df)} agent messages!")
                        st.balloons()
                        
                        # Show immediate statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Messages", len(processed_df))
                        with col2:
                            total_errors = processed_df['total_errors'].sum()
                            st.metric("Total Errors", f"{total_errors:,}")
                        with col3:
                            spelling_errors = processed_df['spelling_errors_count'].sum()
                            st.metric("Spelling Errors", f"{spelling_errors:,}")
                        with col4:
                            grammar_errors = processed_df['grammar_errors_count'].sum()
                            st.metric("Grammar Errors", f"{grammar_errors:,}")
                        
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.header("Analytics Dashboard")
        
        if st.session_state.parquet_path is not None:
            processor = DataProcessor()
            
            with st.spinner("Running analytics..."):
                analytics = processor.analyze_with_duckdb(st.session_state.parquet_path)
            
            # Display summary statistics
            st.subheader("📈 Summary Statistics")
            summary = analytics['summary']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", f"{int(summary['total_messages'].iloc[0]):,}")
                st.metric("Total Errors", f"{int(summary['total_errors'].iloc[0]):,}")
            with col2:
                st.metric("Average Errors/Message", f"{summary['avg_errors_per_message'].iloc[0]:.2f}")
                st.metric("Average Error Rate", f"{summary['avg_error_rate_percent'].iloc[0]:.1f}%")
            with col3:
                st.metric("Error-free Messages", f"{int(summary['error_free_messages'].iloc[0]):,}")
                st.metric("Error-free %", f"{summary['error_free_percentage'].iloc[0]:.1f}%")
            
            # Error category breakdown
            st.subheader("📊 Error Category Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(analytics['category_breakdown'])
            
            with col2:
                # Create a pie chart using Streamlit's native charting
                if not analytics['category_breakdown'].empty:
                    chart_data = analytics['category_breakdown'].set_index('error_type')['count']
                    st.bar_chart(chart_data)
            
            # Top spelling mistakes
            st.subheader("🔤 Top Spelling Mistakes")
            if not analytics['top_spelling_mistakes'].empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(analytics['top_spelling_mistakes'].head(10))
                with col2:
                    chart_data = analytics['top_spelling_mistakes'].head(10).set_index('word')['frequency']
                    st.bar_chart(chart_data)
            else:
                st.info("No spelling mistakes found")
            
            # Detailed view with filtering
            st.subheader("📋 Detailed Message Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                min_errors = st.number_input("Min errors", min_value=0, value=0)
            with col2:
                max_errors = st.number_input("Max errors", min_value=0, value=100)
            with col3:
                error_type_filter = st.selectbox(
                    "Filter by error type",
                    options=['All', 'Spelling', 'Grammar', 'Punctuation', 'Style']
                )
            
            # Apply filters
            filtered_data = analytics['full_data'].copy()
            filtered_data = filtered_data[
                (filtered_data['total_errors'] >= min_errors) &
                (filtered_data['total_errors'] <= max_errors)
            ]
            
            if error_type_filter == 'Spelling':
                filtered_data = filtered_data[filtered_data['spelling_errors_count'] > 0]
            elif error_type_filter == 'Grammar':
                filtered_data = filtered_data[filtered_data['grammar_errors_count'] > 0]
            elif error_type_filter == 'Punctuation':
                filtered_data = filtered_data[filtered_data['punctuation_errors_count'] > 0]
            elif error_type_filter == 'Style':
                filtered_data = filtered_data[filtered_data['style_errors_count'] > 0]
            
            # Display columns to show
            display_columns = [
                'agent_text_cleaned', 'total_errors', 'spelling_errors_count',
                'grammar_errors_count', 'punctuation_errors_count', 'spelling_mistakes',
                'suggested_corrections', 'error_rate'
            ]
            
            # Filter to available columns
            display_columns = [col for col in display_columns if col in filtered_data.columns]
            
            st.dataframe(
                filtered_data[display_columns],
                use_container_width=True,
                height=400
            )
            
            # Store analytics in session state for download
            st.session_state.analytics = analytics
        else:
            st.info("⚠️ Please process a CSV file first in the 'Upload & Process' tab")
    
    with tab3:
        st.header("Download Results")
        
        if st.session_state.parquet_path is not None and 'analytics' in st.session_state:
            st.success("✅ Your consolidated results are ready for download!")
            
            processor = DataProcessor()
            analytics = st.session_state.analytics
            
            st.markdown("### 📦 Download Consolidated File")
            st.markdown("All analysis results in a single file with multiple sheets/sections")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Excel Format")
                st.markdown("Best for analysis in Excel")
                excel_output = processor.export_consolidated_results(analytics, format='xlsx')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_output,
                    file_name=f"grammar_analysis_complete_{timestamp}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            
            with col2:
                st.markdown("#### CSV Format")
                st.markdown("Universal compatibility")
                csv_output = processor.export_consolidated_results(analytics, format='csv')
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_output,
                    file_name=f"grammar_analysis_complete_{timestamp}.csv",
                    mime='text/csv'
                )
            
            with col3:
                st.markdown("#### Parquet Format")
                st.markdown("Best for BI tools")
                parquet_output = processor.export_consolidated_results(analytics, format='parquet')
                
                st.download_button(
                    label="📥 Download Parquet",
                    data=parquet_output,
                    file_name=f"grammar_analysis_complete_{timestamp}.parquet",
                    mime='application/octet-stream'
                )
            
            # Summary of what's included
            st.markdown("---")
            st.markdown("### 📋 What's Included in Your Download:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Data Columns:**
                - All original input columns
                - Cleaned agent text
                - Total error counts
                - Error counts by category
                - Actual spelling mistakes
                - Grammar issues identified
                - Suggested corrections
                - Error rates and percentages
                """)
            
            with col2:
                st.markdown("""
                **Summary Statistics:**
                - Total messages analyzed
                - Error breakdown by type
                - Top spelling mistakes
                - Error-free message count
                - Average error rates
                - Processing timestamp
                - Complete row-level details
                """)
            
        else:
            st.info("⚠️ Please process a CSV file first to download results")

if __name__ == "__main__":
    main()
