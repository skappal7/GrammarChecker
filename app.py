import streamlit as st
import pandas as pd
import duckdb
import os
from pathlib import Path
import tempfile
import language_tool_python
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime
import io
import xlsxwriter
import re
import html

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
        
        # Pattern 1: YYYY-MM-DD HH:MM:SS +0000 Agent:
        pattern1 = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\+\d{4})\s+Agent:(.*?)(?=\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\[|\Z)'
        matches1 = re.findall(pattern1, text, re.DOTALL)
        for timestamp, message in matches1:
            clean_msg = TranscriptParser.clean_text(message)
            if clean_msg:
                agent_messages.append(clean_msg)
        
        # Pattern 2: [HH:MM:SS AGENT]:
        pattern2 = r'\[\d{2}:\d{2}:\d{2}\s+AGENT\]:(.*?)(?=\[\d{2}:\d{2}:\d{2}|\Z)'
        matches2 = re.findall(pattern2, text, re.DOTALL | re.IGNORECASE)
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

class GrammarChecker:
    """Handle grammar checking operations"""
    
    def __init__(self, tool):
        self.tool = tool
    
    def check_text(self, text):
        """Check text for grammar errors"""
        if pd.isna(text) or str(text).strip() == '':
            return 0, [], []
        
        try:
            matches = self.tool.check(str(text))
            error_count = len(matches)
            error_types = [match.ruleId for match in matches]
            error_messages = [match.message for match in matches]
            return error_count, error_types, error_messages
        except Exception as e:
            return 0, [f"Error: {str(e)}"], []
    
    def process_batch(self, texts):
        """Process multiple texts in batch"""
        results = []
        for text in texts:
            error_count, error_types, error_messages = self.check_text(text)
            results.append({
                'error_count': error_count,
                'error_types': '|'.join(error_types) if error_types else '',
                'error_messages': '|'.join(error_messages) if error_messages else ''
            })
        return results

class DataProcessor:
    """Handle data processing with DuckDB"""
    
    def __init__(self):
        self.conn = duckdb.connect(':memory:')
    
    def csv_to_parquet(self, csv_file, text_column, batch_size=1000):
        """Convert CSV to Parquet with grammar checking"""
        # Read CSV in chunks for memory efficiency
        chunks = []
        grammar_checker = GrammarChecker(st.session_state.grammar_tool)
        transcript_parser = TranscriptParser()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Read and process CSV
        total_rows = sum(1 for line in io.StringIO(csv_file.getvalue().decode('utf-8'))) - 1
        csv_file.seek(0)
        
        processed_row_count = 0
        
        for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=batch_size)):
            status_text.text(f"Processing chunk {i+1}...")
            
            # Extract and clean agent messages from transcripts
            if text_column in chunk.columns:
                # Parse transcripts to get only agent messages
                chunk = transcript_parser.process_transcript_column(chunk, text_column)
                
                if len(chunk) > 0:  # Only process if there are agent messages
                    grammar_results = []
                    
                    # Process grammar checking on cleaned agent text
                    for idx in range(0, len(chunk), 100):
                        batch = chunk['agent_text_cleaned'].iloc[idx:idx+100].tolist()
                        batch_results = grammar_checker.process_batch(batch)
                        grammar_results.extend(batch_results)
                        
                        # Update progress
                        current_progress = min((i * batch_size + idx + 100) / total_rows, 1.0)
                        progress_bar.progress(current_progress)
                    
                    # Add grammar check results to chunk
                    grammar_df = pd.DataFrame(grammar_results)
                    chunk['grammar_error_count'] = grammar_df['error_count']
                    chunk['grammar_error_types'] = grammar_df['error_types']
                    chunk['grammar_error_messages'] = grammar_df['error_messages']
                    
                    chunks.append(chunk)
                    processed_row_count += len(chunk)
        
        if not chunks:
            raise ValueError("No agent messages found in the transcript data")
        
        # Combine all chunks
        df_complete = pd.concat(chunks, ignore_index=True)
        
        # Save to parquet
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp_file:
            df_complete.to_parquet(tmp_file.name, engine='pyarrow', compression='snappy')
            progress_bar.progress(1.0)
            status_text.text(f"Processing complete! Found {processed_row_count} agent messages.")
            return tmp_file.name, df_complete
    
    def analyze_with_duckdb(self, parquet_path):
        """Perform analytics using DuckDB"""
        # Register parquet file with DuckDB
        self.conn.execute(f"""
            CREATE OR REPLACE TABLE transcripts AS 
            SELECT * FROM parquet_scan('{parquet_path}')
        """)
        
        # Analytics queries
        analytics = {}
        
        # Total statistics
        analytics['total_stats'] = self.conn.execute("""
            SELECT 
                COUNT(*) as total_rows,
                SUM(grammar_error_count) as total_errors,
                AVG(grammar_error_count) as avg_errors_per_row,
                MIN(grammar_error_count) as min_errors,
                MAX(grammar_error_count) as max_errors,
                COUNT(CASE WHEN grammar_error_count = 0 THEN 1 END) as error_free_rows,
                COUNT(CASE WHEN grammar_error_count > 0 THEN 1 END) as rows_with_errors
            FROM transcripts
        """).df()
        
        # Error distribution
        analytics['error_distribution'] = self.conn.execute("""
            SELECT 
                grammar_error_count as error_count,
                COUNT(*) as row_count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM transcripts
            GROUP BY grammar_error_count
            ORDER BY grammar_error_count
        """).df()
        
        # Top error types
        analytics['top_errors'] = self.conn.execute("""
            WITH error_types AS (
                SELECT 
                    UNNEST(STRING_SPLIT(grammar_error_types, '|')) as error_type
                FROM transcripts
                WHERE grammar_error_types != ''
            )
            SELECT 
                error_type,
                COUNT(*) as occurrence_count
            FROM error_types
            WHERE error_type != ''
            GROUP BY error_type
            ORDER BY occurrence_count DESC
            LIMIT 20
        """).df()
        
        # Row-level details
        analytics['row_details'] = self.conn.execute("""
            SELECT *
            FROM transcripts
            ORDER BY grammar_error_count DESC
        """).df()
        
        return analytics
    
    def export_results(self, analytics, format='xlsx'):
        """Export results to specified format"""
        output = io.BytesIO()
        
        if format == 'xlsx':
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                analytics['total_stats'].to_excel(writer, sheet_name='Summary', index=False)
                analytics['error_distribution'].to_excel(writer, sheet_name='Error Distribution', index=False)
                analytics['top_errors'].to_excel(writer, sheet_name='Top Error Types', index=False)
                analytics['row_details'].to_excel(writer, sheet_name='Row Details', index=False)
                
                # Format the Excel file
                workbook = writer.book
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D3D3D3',
                    'border': 1
                })
                
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    worksheet.set_column('A:Z', 15)
        
        elif format == 'csv':
            # For CSV, we'll export the main row details
            analytics['row_details'].to_csv(output, index=False)
        
        elif format == 'parquet':
            analytics['row_details'].to_parquet(output, engine='pyarrow', compression='snappy')
        
        output.seek(0)
        return output

def main():
    st.title("📝 Grammar Check Analytics System - Agent Transcripts")
    st.markdown("### Analyze agent transcripts for grammatical errors using DuckDB and LanguageTool")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("""
        **How it works:**
        1. Upload CSV file with transcripts
        2. Select transcript column
        3. System extracts agent messages only
        4. Cleans text and checks grammar
        5. Converts to Parquet format
        6. DuckDB performs analytics
        7. Download results in multiple formats
        """)
        
        st.markdown("---")
        st.markdown("**Supported Transcript Formats:**")
        st.markdown("- `YYYY-MM-DD HH:MM:SS Agent:`")
        st.markdown("- `[HH:MM:SS AGENT]:`")
        st.markdown("- Automatically filters customer messages")
        st.markdown("- Cleans HTML and special characters")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Analytics", "💾 Download Results"])
    
    with tab1:
        st.header("Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
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
                "Select the column containing text/transcripts",
                options=columns,
                help="Choose the column that contains the text you want to check for grammar"
            )
            
            # Batch size configuration
            batch_size = st.number_input(
                "Batch size for processing",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Larger batch sizes are faster but use more memory"
            )
            
            # Process button
            if st.button("🚀 Start Processing", type="primary"):
                processor = DataProcessor()
                
                with st.spinner("Converting CSV to Parquet and checking grammar..."):
                    try:
                        # Process the file
                        parquet_path, processed_df = processor.csv_to_parquet(
                            uploaded_file,
                            text_column,
                            batch_size
                        )
                        
                        # Store in session state
                        st.session_state.parquet_path = parquet_path
                        st.session_state.processed_data = processed_df
                        
                        st.success(f"✅ Successfully processed {len(processed_df)} rows!")
                        st.balloons()
                        
                        # Show basic stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", len(processed_df))
                        with col2:
                            total_errors = processed_df['grammar_error_count'].sum()
                            st.metric("Total Grammar Errors", total_errors)
                        with col3:
                            avg_errors = processed_df['grammar_error_count'].mean()
                            st.metric("Avg Errors per Row", f"{avg_errors:.2f}")
                        
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.header("Analytics Dashboard")
        
        if st.session_state.parquet_path is not None:
            processor = DataProcessor()
            
            with st.spinner("Running analytics with DuckDB..."):
                analytics = processor.analyze_with_duckdb(st.session_state.parquet_path)
            
            # Display analytics
            st.subheader("📈 Summary Statistics")
            st.dataframe(analytics['total_stats'])
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Error Distribution")
                st.bar_chart(
                    analytics['error_distribution'].set_index('error_count')['row_count']
                )
            
            with col2:
                st.subheader("🏷️ Top Error Types")
                if not analytics['top_errors'].empty:
                    st.bar_chart(
                        analytics['top_errors'].set_index('error_type')['occurrence_count'].head(10)
                    )
                else:
                    st.info("No error types found")
            
            # Detailed view
            st.subheader("📋 Row-Level Details")
            
            # Filtering options
            col1, col2 = st.columns(2)
            with col1:
                min_errors = st.number_input(
                    "Minimum error count",
                    min_value=0,
                    value=0
                )
            with col2:
                max_errors = st.number_input(
                    "Maximum error count",
                    min_value=0,
                    value=int(analytics['row_details']['grammar_error_count'].max())
                )
            
            # Filter the data
            filtered_data = analytics['row_details'][
                (analytics['row_details']['grammar_error_count'] >= min_errors) &
                (analytics['row_details']['grammar_error_count'] <= max_errors)
            ]
            
            st.dataframe(filtered_data, use_container_width=True)
            
            # Store analytics in session state for download
            st.session_state.analytics = analytics
        else:
            st.info("⚠️ Please process a CSV file first in the 'Upload & Process' tab")
    
    with tab3:
        st.header("Download Results")
        
        if st.session_state.parquet_path is not None and 'analytics' in st.session_state:
            st.success("✅ Results ready for download!")
            
            processor = DataProcessor()
            analytics = st.session_state.analytics
            
            # Download options
            st.subheader("Select Download Format")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Primary Export")
                export_format = st.radio(
                    "Choose format",
                    options=['xlsx', 'csv'],
                    format_func=lambda x: {
                        'xlsx': 'Excel (.xlsx)',
                        'csv': 'CSV (.csv)'
                    }[x]
                )
                
                # Generate primary export
                output = processor.export_results(analytics, format=export_format)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"grammar_check_results_{timestamp}.{export_format}"
                
                st.download_button(
                    label=f"📥 Download {export_format.upper()}",
                    data=output,
                    file_name=filename,
                    mime={
                        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        'csv': 'text/csv'
                    }[export_format]
                )
            
            with col2:
                st.markdown("### Parquet Export (Mandatory)")
                st.info("Parquet format is optimized for BI tools")
                
                # Generate parquet export
                parquet_output = processor.export_results(analytics, format='parquet')
                parquet_filename = f"grammar_check_results_{timestamp}.parquet"
                
                st.download_button(
                    label="📥 Download Parquet",
                    data=parquet_output,
                    file_name=parquet_filename,
                    mime='application/octet-stream'
                )
            
            # Additional export options
            st.markdown("---")
            st.subheader("🔧 Advanced Options")
            
            if st.checkbox("Include only rows with errors"):
                filtered_analytics = {
                    key: value for key, value in analytics.items()
                }
                filtered_analytics['row_details'] = analytics['row_details'][
                    analytics['row_details']['grammar_error_count'] > 0
                ]
                
                filtered_output = processor.export_results(filtered_analytics, format=export_format)
                
                st.download_button(
                    label=f"📥 Download Filtered {export_format.upper()}",
                    data=filtered_output,
                    file_name=f"grammar_check_errors_only_{timestamp}.{export_format}",
                    mime={
                        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        'csv': 'text/csv'
                    }[export_format]
                )
        else:
            st.info("⚠️ Please process a CSV file first to download results")

if __name__ == "__main__":
    main()
