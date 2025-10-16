# 📝 Grammar Check Analytics System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.9%2B-FFF000)](https://duckdb.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/skappal7/grammar-check-analytics/graphs/commit-activity)

A high-performance, enterprise-grade solution for analyzing agent transcript quality at scale. Built with modern data engineering practices, this system processes millions of customer service transcripts to identify grammatical errors, enabling quality assurance teams to improve agent communication standards.

## 🎯 Problem Statement

Customer service quality directly impacts brand reputation. With thousands of daily agent interactions, manually reviewing transcript quality is impossible. This solution automates grammar analysis across massive transcript datasets, providing actionable insights for training and quality improvement.

## ✨ Key Features

### 🚀 **Performance at Scale**
- **Parquet Optimization**: 50-90% storage reduction with columnar compression
- **DuckDB Analytics**: In-process OLAP queries without infrastructure overhead
- **Batch Processing**: Memory-efficient processing of multi-GB datasets
- **Streaming Architecture**: Process files larger than available RAM

### 🎨 **Intelligent Parsing**
- **Multi-format Support**: Handles various transcript timestamp formats
- **Agent-only Focus**: Automatically filters customer messages
- **HTML Cleaning**: Removes tags, fixes encoding issues
- **Smart Text Extraction**: Preserves context while cleaning noise

### 📊 **Comprehensive Analytics**
- **Error Distribution**: Statistical analysis of grammar patterns
- **Error Classification**: Categorizes issues by type and severity
- **Trend Analysis**: Track quality improvements over time
- **Row-level Details**: Drill down to specific problematic transcripts

### 💼 **Enterprise Integration**
- **Multiple Export Formats**: Excel, CSV, Parquet
- **BI-Ready Output**: Direct integration with Power BI, Tableau, Qlik
- **Batch Processing API**: Scriptable for automated workflows
- **Cloud-Compatible**: Deploy on AWS, GCP, or Azure

## 🏗️ Architecture

```
CSV Upload → Transcript Parser → Agent Text Extraction → Text Cleaning
    ↓
Export Results ← DuckDB Analytics ← Parquet Conversion ← Grammar Analysis
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **Grammar Engine** | LanguageTool | Multi-language grammar checking |
| **Data Processing** | Pandas + NumPy | Efficient data manipulation |
| **Storage Format** | Apache Parquet | Columnar storage optimization |
| **Analytics Engine** | DuckDB | SQL analytics on Parquet |
| **Export** | XlsxWriter | Multi-format data export |

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum (8GB+ recommended for large datasets)
- 2GB free disk space for LanguageTool models

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/skappal7/grammar-check-analytics.git
cd grammar-check-analytics
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run grammar_check_app.py
```

The application will open at `http://localhost:8501`

## 📖 Usage Guide

### 1. Data Preparation
Prepare your CSV with transcript data. Supported formats:

```
timestamp,transcript
"2025-01-04 02:56:31 +0000","Agent: Hello, how can I help you today?"
```

### 2. Upload & Process
- Navigate to the **Upload & Process** tab
- Select your CSV file
- Choose the column containing transcripts
- Click **Start Processing**

### 3. View Analytics
- Switch to the **Analytics** tab
- Review summary statistics
- Explore error distributions
- Filter results by error count

### 4. Export Results
- Go to **Download Results** tab
- Choose primary format (Excel/CSV)
- Download mandatory Parquet for BI tools
- Optional: Export filtered datasets

## 📊 Sample Output

### Summary Statistics
| Metric | Value |
|--------|--------|
| Total Agent Messages | 15,234 |
| Total Grammar Errors | 3,456 |
| Average Errors/Message | 0.23 |
| Error-free Messages | 12,543 (82.3%) |

### Top Grammar Issues
1. **COMMA_PARENTHESIS_WHITESPACE** - 23%
2. **UPPERCASE_SENTENCE_START** - 18%
3. **AGREEMENT_ERRORS** - 15%
4. **MISSING_PUNCTUATION** - 12%

## 🔧 Configuration

### Batch Size Optimization
```python
# Adjust based on available RAM
BATCH_SIZE = 1000  # Default
BATCH_SIZE = 5000  # For 16GB+ RAM
BATCH_SIZE = 500   # For low-memory systems
```

### Supported Transcript Formats
```python
# Format 1: ISO Timestamp
"2025-01-04 02:56:31 +0000 Agent: Message"

# Format 2: Bracket Time
"[14:23:45 AGENT]: Message"

# Custom formats can be added in TranscriptParser class
```

## 🚀 Performance Benchmarks

| Dataset Size | Processing Time | Memory Usage |
|-------------|-----------------|--------------|
| 10,000 rows | ~2 minutes | 500 MB |
| 100,000 rows | ~15 minutes | 2 GB |
| 1,000,000 rows | ~2 hours | 8 GB |

*Benchmarks on Intel i7, 16GB RAM, SSD storage*

## 🔬 Advanced Features

### Custom Grammar Rules
```python
# Add domain-specific rules
custom_rules = {
    'GREETING_FORMAT': r'^(Hi|Hello|Hey)',
    'CLOSING_FORMAT': r'(Thank you|Thanks|Regards)$'
}
```

### API Integration
```python
# Programmatic usage
from grammar_check_app import DataProcessor, GrammarChecker

processor = DataProcessor()
results = processor.analyze_with_duckdb('data.parquet')
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 grammar_check_app.py
```

## 📈 Roadmap

- [ ] Real-time streaming processing
- [ ] Multi-language support (Spanish, French, German)
- [ ] Custom grammar rule builder UI
- [ ] Cloud deployment templates
- [ ] REST API endpoint
- [ ] Sentiment analysis integration
- [ ] Automated quality scoring
- [ ] Team performance dashboards

## 🐛 Troubleshooting

### Common Issues

**LanguageTool Download Fails**
```bash
# Manual download
python -m language_tool_python.download_lt
```

**Memory Error on Large Files**
```python
# Reduce batch size in the UI or code
batch_size = 500  # Instead of 1000
```

**DuckDB Version Conflict**
```bash
pip uninstall duckdb
pip install duckdb==0.9.2
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LanguageTool](https://languagetool.org/) for grammar checking engine
- [DuckDB](https://duckdb.org/) for analytical processing
- [Streamlit](https://streamlit.io/) for the web framework
- [Apache Parquet](https://parquet.apache.org/) for columnar storage

## 📧 Contact

- **Project Lead**: Skappal7
- **GitHub**: [github.com/skappal7](https://github.com/skappal7)
- **Issues**: [GitHub Issues](https://github.com/skappal7/grammar-check-analytics/issues)

---

<p align="center">
Built with ❤️ for Quality Assurance Teams Worldwide
</p>

<p align="center">
<a href="https://github.com/skappal7/grammar-check-analytics/stargazers">⭐ Star this repo</a> • 
<a href="https://github.com/skappal7/grammar-check-analytics/fork">🍴 Fork it</a> • 
<a href="https://github.com/skappal7/grammar-check-analytics/issues">🐛 Report Bug</a> • 
<a href="https://github.com/skappal7/grammar-check-analytics/issues">✨ Request Feature</a>
</p>
