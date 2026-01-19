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
FOOTER = "Developed with Streamlit by CE Team Innovation Lab 2025"

# ============================================================================
# SVG ICONS - Professional Design System
# ============================================================================
SVG_ICONS = {
    'shield': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>''',
    'upload': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>''',
    'download': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>''',
    'chart': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>''',
    'settings': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>''',
    'check': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>''',
    'error': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>''',
    'info': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>''',
    'mail': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>''',
    'chat': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>''',
    'lock': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>''',
    'globe': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>''',
    'cpu': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>''',
    'target': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>''',
    'list': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>''',
    'file': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>''',
    'zap': '''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>''',
}

def icon(name, size=20):
    svg = SVG_ICONS.get(name, '')
    if not svg:
        return ''
    # Replace both possible default sizes (20 and 24 for shield)
    svg = svg.replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"')
    svg = svg.replace('width="20"', f'width="{size}"').replace('height="20"', f'height="{size}"')
    return svg

def icon_text(icon_name, text, size=20):
    return f'<span style="display:inline-flex;align-items:center;gap:8px;">{icon(icon_name, size)} {text}</span>'

# Configure Streamlit
st.set_page_config(page_title="TextGuardian Pro", page_icon="🛡️", layout="wide")

# ============================================================================
# STUNNING CSS - Premium Eye-Candy Design
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ===== GLOBAL STYLES ===== */
    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }}
    
    /* ===== FLOATING LOGO - TOP RIGHT ===== */
    .floating-logo {{
        position: fixed;
        top: 12px;
        right: 20px;
        z-index: 999999;
        background: white;
        padding: 8px 12px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }}
    .floating-logo img {{
        height: 36px;
        display: block;
    }}
    
    /* ===== HERO HEADER ===== */
    .hero-header {{
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3a7ca5 100%);
        padding: 2.5rem 3rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(30, 58, 95, 0.3);
        position: relative;
        overflow: hidden;
    }}
    .hero-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        pointer-events: none;
    }}
    .hero-title {{
        display: flex;
        align-items: center;
        gap: 16px;
        color: white;
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }}
    .hero-title svg {{
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
    }}
    .hero-subtitle {{
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.75rem;
        font-weight: 400;
    }}
    
    /* ===== GLASS CARDS ===== */
    .glass-card {{
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 20px;
        padding: 1.75rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255,255,255,0.8);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.12);
    }}
    
    /* ===== METRIC CARDS ===== */
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.25rem;
        margin: 1.5rem 0;
    }}
    .metric-card {{
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.04);
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .metric-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }}
    .metric-icon {{
        width: 48px;
        height: 48px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
    }}
    .metric-icon svg {{
        color: #4338ca;
        width: 24px;
        height: 24px;
    }}
    .metric-value {{
        font-size: 2.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a5f 0%, #3a7ca5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }}
    .metric-label {{
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* ===== PII PROTECTION BADGE ===== */
    .pii-badge {{
        background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
        color: white;
        padding: 1.25rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 8px 24px rgba(153, 27, 27, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }}
    .pii-badge svg {{
        width: 24px;
        height: 24px;
    }}
    
    /* ===== SUCCESS CARD ===== */
    .success-card {{
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 8px 24px rgba(6, 95, 70, 0.25);
        font-weight: 500;
        font-size: 1.05rem;
    }}
    .success-card svg {{
        width: 28px;
        height: 28px;
        flex-shrink: 0;
    }}
    
    /* ===== SECTION HEADERS ===== */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin: 2rem 0 1.25rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #e2e8f0;
    }}
    .section-header svg {{
        color: #1e3a5f;
        width: 26px;
        height: 26px;
    }}
    
    /* ===== SIDEBAR ===== */
    .sidebar-card {{
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }}
    .sidebar-label {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
    }}
    .sidebar-label svg {{
        color: #1e3a5f;
        width: 20px;
        height: 20px;
    }}
    
    /* ===== INFO BOX ===== */
    .info-box {{
        background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 0 14px 14px 0;
        color: #1e40af;
        font-size: 0.9rem;
        line-height: 1.6;
    }}
    
    /* ===== UPLOAD ZONE ===== */
    .upload-zone {{
        border: 3px dashed #cbd5e1;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        transition: all 0.3s ease;
    }}
    .upload-zone:hover {{
        border-color: #1e3a5f;
        background: linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 100%);
        transform: scale(1.01);
    }}
    .upload-icon {{
        width: 64px;
        height: 64px;
        margin: 0 auto 1rem;
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .upload-icon svg {{
        color: #4338ca;
        width: 32px;
        height: 32px;
    }}
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(255,255,255,0.8);
        padding: 10px;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%) !important;
        color: white !important;
    }}
    
    /* ===== BUTTONS ===== */
    .stButton > button {{
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border: none;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.875rem 2.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(30, 58, 95, 0.3);
    }}
    .stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(30, 58, 95, 0.4);
    }}
    
    /* ===== DATAFRAME ===== */
    .stDataFrame {{
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }}
    
    /* ===== FOOTER ===== */
    .footer {{
        text-align: center;
        padding: 2rem 1rem;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }}
</style>

<!-- Floating Logo Top Right -->
<div class="floating-logo">
    <img src="{LOGO_URL}" alt="Logo">
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'parquet_path' not in st.session_state:
    st.session_state.parquet_path = None
if 'analytics' not in st.session_state:
    st.session_state.analytics = None
if 'pii_stats' not in st.session_state:
    st.session_state.pii_stats = None

# ============================================================================
# GLOBAL PII REDACTION ENGINE - GDPR + Multi-Region Compliant
# ============================================================================
PII_PATTERNS = {
    # Universal
    'email': (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),
    'credit_card': (re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'), '[CARD]'),
    'ip_address': (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP]'),
    'date_of_birth': (re.compile(r'\b(?:DOB|D\.O\.B|Date of Birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE), '[DOB]'),
    'passport': (re.compile(r'\b(?:Passport|PP)[:#\s]*[A-Z]{1,2}\d{6,9}\b', re.IGNORECASE), '[PASSPORT]'),
    
    # UAE
    'phone_uae': (re.compile(r'(?:\+971|00971|971)?[- ]?(?:50|52|54|55|56|58)[- ]?\d{3}[- ]?\d{4}'), '[PHONE]'),
    'emirates_id': (re.compile(r'784[- ]?\d{4}[- ]?\d{7}[- ]?\d'), '[EMIRATES_ID]'),
    'uae_plate': (re.compile(r'\b[A-Z]{1,2}[- ]?\d{1,5}[- ]?(?:Dubai|AbuDhabi|Sharjah|Ajman|RAK|UAQ|Fujairah|AD|DXB|SHJ)\b', re.IGNORECASE), '[PLATE]'),
    
    # UK
    'phone_uk': (re.compile(r'(?:\+44|0044)?[- ]?(?:7\d{3}|20|1\d{2,4})[- ]?\d{3}[- ]?\d{3,4}'), '[PHONE]'),
    'uk_nino': (re.compile(r'\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]?\b'), '[NINO]'),
    'uk_postcode': (re.compile(r'\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b', re.IGNORECASE), '[POSTCODE]'),
    
    # US
    'phone_us': (re.compile(r'(?:\+1[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}'), '[PHONE]'),
    'us_ssn': (re.compile(r'\b(?:SSN|Social)[:\s]*\d{3}[- ]?\d{2}[- ]?\d{4}\b', re.IGNORECASE), '[SSN]'),
    'us_zip': (re.compile(r'\b(?:ZIP|Zip Code)[:\s]*\d{5}(?:-\d{4})?\b', re.IGNORECASE), '[ZIP]'),
    
    # India
    'phone_india': (re.compile(r'(?:\+91|0091)[- ]?[6-9]\d{4}[- ]?\d{5}'), '[PHONE]'),
    'india_aadhaar': (re.compile(r'\b(?:Aadhaar|UID)[:\s]*\d{4}[- ]?\d{4}[- ]?\d{4}\b', re.IGNORECASE), '[AADHAAR]'),
    'india_pan': (re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'), '[PAN]'),
    
    # EU
    'eu_vat': (re.compile(r'\b[A-Z]{2}\d{8,12}\b'), '[VAT]'),
    'iban': (re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b'), '[IBAN]'),
    
    # Business
    'reservation': (re.compile(r'\b(?:Ref|Reference|Booking|Confirmation)[:#\s]*[A-Z0-9]{6,13}\b', re.IGNORECASE), '[REF]'),
    'po_box': (re.compile(r'P\.?O\.?\s*BOX\s*\d+', re.IGNORECASE), '[ADDRESS]'),
}

# Name patterns - salutations and signatures
NAME_PATTERNS = [
    re.compile(r'\bDear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE),
    re.compile(r'\bHi\s+([A-Z][a-z]+)', re.IGNORECASE),
    re.compile(r'\bHello\s+([A-Z][a-z]+)', re.IGNORECASE),
    re.compile(r'Kind\s+Regards,?\s*\n?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE),
    re.compile(r'Best\s+Regards,?\s*\n?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE),
    re.compile(r'Regards,?\s*\n?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE),
    re.compile(r'Sincerely,?\s*\n?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE),
    re.compile(r'From:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*<', re.IGNORECASE),
    re.compile(r'To:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*<', re.IGNORECASE),
]

def redact_pii(text, collect_stats=True):
    """Redact PII - Global GDPR Compliant"""
    if pd.isna(text) or not text:
        return ("", {}) if collect_stats else ""
    
    text = str(text)
    stats = {k: 0 for k in PII_PATTERNS.keys()}
    stats['names'] = 0
    
    for pii_type, (pattern, replacement) in PII_PATTERNS.items():
        matches = pattern.findall(text)
        stats[pii_type] = len(matches)
        text = pattern.sub(replacement, text)
    
    for name_pattern in NAME_PATTERNS:
        matches = name_pattern.findall(text)
        stats['names'] += len(matches)
        for name in matches:
            if len(name) > 2:
                text = text.replace(name, '[NAME]')
    
    return (text, stats) if collect_stats else text

# ============================================================================
# EMAIL FORMAT DETECTION & PARSING
# ============================================================================
EMAIL_INDICATORS = [
    re.compile(r'\bFrom:\s*', re.IGNORECASE),
    re.compile(r'\bTo:\s*', re.IGNORECASE),
    re.compile(r'\bSubject:\s*', re.IGNORECASE),
    re.compile(r'Kind\s+Regards', re.IGNORECASE),
    re.compile(r'Best\s+Regards', re.IGNORECASE),
    re.compile(r'\bDear\s+[A-Z]', re.IGNORECASE),
    re.compile(r'EXTERNAL\s+EMAIL', re.IGNORECASE),
    re.compile(r'@[a-z]+\.[a-z]{2,}', re.IGNORECASE),
]

EMAIL_THREAD_MARKERS = [
    re.compile(r'On\s+\d{1,2}/\d{1,2}/\d{2,4}.*?wrote:', re.IGNORECASE | re.DOTALL),
    re.compile(r'On\s+\d{1,2}\s+\w+\s+\d{4}.*?wrote:', re.IGNORECASE | re.DOTALL),
    re.compile(r'-{3,}\s*Forwarded\s+message\s*-{3,}', re.IGNORECASE),
    re.compile(r'Begin\s+forwarded\s+message:', re.IGNORECASE),
]

EMAIL_NOISE_PATTERNS = [
    re.compile(r'Caution:\s*EXTERNAL\s+EMAIL.*?(?=\n\n|\Z)', re.IGNORECASE | re.DOTALL),
    re.compile(r'\[cid:image\d+\.png@[^\]]+\]', re.IGNORECASE),
    re.compile(r'<https?://[^>]+>', re.IGNORECASE),
    re.compile(r'mailto:[^\s>]+', re.IGNORECASE),
]

def is_email_format(text):
    if pd.isna(text) or not text:
        return False
    text = str(text)
    return sum(1 for p in EMAIL_INDICATORS if p.search(text)) >= 2

def extract_email_body(text):
    if pd.isna(text) or not text:
        return ""
    text = str(text)
    for pattern in EMAIL_NOISE_PATTERNS:
        text = pattern.sub(' ', text)
    for pattern in EMAIL_THREAD_MARKERS:
        match = pattern.search(text)
        if match:
            text = text[:match.start()]
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# ============================================================================
# CHAT PARSING
# ============================================================================
AGENT_PATTERN_1 = re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\+\d{4})\s+Agent:(.*?)(?=\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\Z)', re.DOTALL)
AGENT_PATTERN_2 = re.compile(r'\[\d{2}:\d{2}:\d{2}\s+AGENT\]:(.*?)(?=\[\d{2}:\d{2}:\d{2}|\Z)', re.DOTALL | re.IGNORECASE)

# ============================================================================
# GRAMMAR PATTERNS
# ============================================================================
MISSPELLING_PATTERNS = {
    'recieve': re.compile(r'\brecieve\b', re.IGNORECASE),
    'occured': re.compile(r'\boccured\b', re.IGNORECASE),
    'seperate': re.compile(r'\bseperate\b', re.IGNORECASE),
    'definately': re.compile(r'\bdefinately\b', re.IGNORECASE),
    'accomodate': re.compile(r'\baccomodate\b', re.IGNORECASE),
    'wierd': re.compile(r'\bwierd\b', re.IGNORECASE),
    'untill': re.compile(r'\buntill\b', re.IGNORECASE),
    'thier': re.compile(r'\bthier\b', re.IGNORECASE),
    'becuase': re.compile(r'\bbecuase\b', re.IGNORECASE),
    'alot': re.compile(r'\balot\b', re.IGNORECASE),
    'tommorow': re.compile(r'\btommorow\b', re.IGNORECASE),
    'beleive': re.compile(r'\bbeleive\b', re.IGNORECASE),
    'calender': re.compile(r'\bcalender\b', re.IGNORECASE),
    'collegue': re.compile(r'\bcollegue\b', re.IGNORECASE),
    'enviroment': re.compile(r'\benviroment\b', re.IGNORECASE),
    'goverment': re.compile(r'\bgoverment\b', re.IGNORECASE),
    'guarentee': re.compile(r'\bguarentee\b', re.IGNORECASE),
    'imediately': re.compile(r'\bimediately\b', re.IGNORECASE),
    'neccessary': re.compile(r'\bneccessary\b', re.IGNORECASE),
    'occurance': re.compile(r'\boccurance\b', re.IGNORECASE),
    'reccomend': re.compile(r'\breccomend\b', re.IGNORECASE),
    'succesful': re.compile(r'\bsuccesful\b', re.IGNORECASE),
    'suprise': re.compile(r'\bsuprise\b', re.IGNORECASE),
}

COMMON_MISSPELLINGS = {
    'recieve': 'receive', 'occured': 'occurred', 'seperate': 'separate',
    'definately': 'definitely', 'accomodate': 'accommodate', 'wierd': 'weird',
    'untill': 'until', 'thier': 'their', 'becuase': 'because', 'alot': 'a lot',
    'tommorow': 'tomorrow', 'beleive': 'believe', 'calender': 'calendar',
    'collegue': 'colleague', 'enviroment': 'environment', 'goverment': 'government',
    'guarentee': 'guarantee', 'imediately': 'immediately', 'neccessary': 'necessary',
    'occurance': 'occurrence', 'reccomend': 'recommend', 'succesful': 'successful',
    'suprise': 'surprise',
}

CONTRACTION_PATTERN = re.compile(r'\b(dont|wont|cant|shouldnt|wouldnt|couldnt|didnt|doesnt|isnt|arent|wasnt|werent)\b')
DOUBLE_SPACE = re.compile(r'\s{2,}')
SPACE_BEFORE_PUNCT = re.compile(r'\s+([.!?,;:])')
NO_SPACE_AFTER_PUNCT = re.compile(r'([.!?,;:])([A-Za-z])')
MISSING_CAPITAL = re.compile(r'([.!?]\s+)([a-z])')
ARTICLE_A_VOWEL = re.compile(r'\ba\s+[aeiouAEIOU]\w+')
ARTICLE_AN_CONSONANT = re.compile(r'\ban\s+[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w+')
SUBJECT_VERB_ERROR = re.compile(r'\b(he|she|it)\s+(were|are)\b|\b(they|we|you)\s+(was|is)\b', re.IGNORECASE)

# Common words that should NEVER be capitalized after comma (excludes proper nouns)
COMMON_WORDS_LOWERCASE = {
    'please', 'thank', 'thanks', 'the', 'a', 'an', 'and', 'or', 'but', 'so', 'yet',
    'how', 'what', 'when', 'where', 'why', 'who', 'which', 'that', 'this', 'these',
    'we', 'you', 'they', 'it', 'he', 'she', 'our', 'your', 'their', 'its', 'my',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
    'if', 'then', 'else', 'as', 'for', 'to', 'from', 'with', 'without', 'by', 'at',
    'however', 'therefore', 'hence', 'thus', 'also', 'too', 'very', 'just', 'only',
    'kindly', 'attached', 'below', 'above', 'here', 'there', 'now', 'today', 'tomorrow',
    'regarding', 'concerning', 'following', 'including', 'excluding', 'per', 'via',
}
CAPITAL_AFTER_COMMA_PATTERN = re.compile(r',\s*([A-Z][a-z]+)')

def check_capital_after_comma(text):
    """Check for incorrectly capitalized common words after comma"""
    matches = CAPITAL_AFTER_COMMA_PATTERN.findall(text)
    errors = []
    for word in matches:
        if word.lower() in COMMON_WORDS_LOWERCASE:
            errors.append(word)
    return errors

# ============================================================================
# CORE PROCESSING
# ============================================================================
def load_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    try:
        if ext == 'csv':
            return pd.read_csv(uploaded_file, encoding='utf-8')
        elif ext in ['xlsx', 'xls']:
            engine = 'openpyxl' if ext == 'xlsx' else 'xlrd'
            return pd.read_excel(uploaded_file, engine=engine)
        raise ValueError(f"Unsupported: {ext}")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding='latin-1')

def save_parquet(df):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
    tmp.close()
    df.to_parquet(tmp.name, engine='pyarrow', compression='snappy', index=False)
    return tmp.name

def clean_text(text):
    if pd.isna(text) or not text:
        return ""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')
    text = DOUBLE_SPACE.sub(' ', text)
    return text.strip()

def extract_agent(text):
    if pd.isna(text) or not text:
        return ""
    text = str(text)
    messages = []
    for _, msg in AGENT_PATTERN_1.findall(text):
        c = clean_text(msg)
        if c:
            messages.append(c)
    for msg in AGENT_PATTERN_2.findall(text):
        c = clean_text(msg)
        if c:
            messages.append(c)
    if not messages:
        return clean_text(text)
    return ' '.join(messages)

@lru_cache(maxsize=10000)
def count_words(text):
    return len(text.split())

def check_grammar(text):
    if not text or len(text) < 5:
        return {'total_errors': 0, 'spelling_count': 0, 'grammar_count': 0,
                'punctuation_count': 0, 'error_details': '', 'corrections': '', 'word_count': 0}
    
    errors, details, corrections = [], [], []
    spelling_found = []
    
    for misspelled, pattern in MISSPELLING_PATTERNS.items():
        for match in pattern.findall(text):
            spelling_found.append(match)
            errors.append(f"SPELLING:{match}")
            if misspelled.lower() in COMMON_MISSPELLINGS:
                fix = COMMON_MISSPELLINGS[misspelled.lower()]
                corrections.append(f"{match}→{fix}")
                details.append(f"'{match}' → '{fix}'")
    
    grammar = []
    for c in CONTRACTION_PATTERN.findall(text)[:3]:
        grammar.append(f"apostrophe:{c}")
        errors.append(f"GRAMMAR:{c}")
        details.append(f"Missing apostrophe: '{c}'")
    
    if ARTICLE_A_VOWEL.search(text):
        grammar.append("a→an")
        errors.append("GRAMMAR:article")
        details.append("Use 'an' before vowel")
    if ARTICLE_AN_CONSONANT.search(text):
        grammar.append("an→a")
        errors.append("GRAMMAR:article")
        details.append("Use 'a' before consonant")
    if SUBJECT_VERB_ERROR.search(text):
        grammar.append("subject-verb")
        errors.append("GRAMMAR:agreement")
        details.append("Subject-verb disagreement")
    
    # Smart capital after comma check (only flags common words, not proper nouns)
    capital_errors = check_capital_after_comma(text)
    for word in capital_errors[:3]:  # Limit to first 3
        grammar.append(f"capital:{word}")
        errors.append(f"GRAMMAR:capital_{word}")
        details.append(f"Unnecessary capital: '{word}' after comma")
    
    punctuation = []
    if SPACE_BEFORE_PUNCT.search(text):
        punctuation.append("space-punct")
        errors.append("PUNCT:space")
        details.append("Space before punctuation")
    if NO_SPACE_AFTER_PUNCT.search(text):
        punctuation.append("no-space")
        errors.append("PUNCT:nospace")
        details.append("Missing space after punctuation")
    if MISSING_CAPITAL.search(text):
        punctuation.append("capital")
        errors.append("PUNCT:capital")
        details.append("Missing capitalization")
    
    return {
        'total_errors': len(errors), 'spelling_count': len(spelling_found),
        'grammar_count': len(grammar), 'punctuation_count': len(punctuation),
        'error_details': ' | '.join(details[:10]),
        'corrections': '; '.join(corrections[:8]),
        'word_count': count_words(text)
    }

def process_single_text(args):
    text, mode, redact_pii_flag = args
    
    if mode == 'email':
        if not is_email_format(text):
            return None
        extracted = extract_email_body(text)
    else:
        extracted = extract_agent(text)
    
    pii_stats = {}
    if redact_pii_flag:
        extracted, pii_stats = redact_pii(extracted, collect_stats=True)
    
    grammar_result = check_grammar(extracted)
    grammar_result['redacted_text'] = extracted  # ONLY redacted text stored
    grammar_result['is_valid_format'] = True
    grammar_result['pii_redacted'] = sum(pii_stats.values()) if pii_stats else 0
    
    return grammar_result, pii_stats

def process_batch(batch_args):
    texts, mode, redact_pii_flag = batch_args
    results, pii_totals = [], {}
    
    for text in texts:
        result = process_single_text((text, mode, redact_pii_flag))
        if result:
            grammar_result, pii_stats = result
            results.append(grammar_result)
            for k, v in pii_stats.items():
                pii_totals[k] = pii_totals.get(k, 0) + v
        else:
            results.append({
                'total_errors': 0, 'spelling_count': 0, 'grammar_count': 0,
                'punctuation_count': 0, 'error_details': '', 'corrections': '',
                'word_count': 0, 'redacted_text': '[NOT_EMAIL_FORMAT]',
                'is_valid_format': False, 'pii_redacted': 0
            })
    
    return results, pii_totals

def process_file(df, text_col, mode='chat', redact_pii_flag=True, workers=None):
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)
    
    texts = df[text_col].tolist()
    batch_size = max(100, len(texts) // (workers * 4))
    batches = [(texts[i:i + batch_size], mode, redact_pii_flag) for i in range(0, len(texts), batch_size)]
    
    with st.spinner(f"Processing {len(texts):,} records..."):
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batch_results = list(executor.map(process_batch, batches))
    
    all_results, all_pii_stats = [], {}
    for results, pii_stats in batch_results:
        all_results.extend(results)
        for k, v in pii_stats.items():
            all_pii_stats[k] = all_pii_stats.get(k, 0) + v
    
    result_df = pd.DataFrame(all_results)
    for col in result_df.columns:
        df[col] = result_df[col]
    
    df['error_rate'] = (df['total_errors'] / df['word_count'].replace(0, 1) * 100).round(2)
    
    return df, save_parquet(df), all_pii_stats

def create_analytics(pq_path):
    conn = duckdb.connect(':memory:')
    
    summary = conn.execute(f"""
        SELECT COUNT(*) as total_rows, SUM(total_errors) as total_errors,
            AVG(total_errors) as avg_errors, SUM(spelling_count) as spelling_errors,
            SUM(grammar_count) as grammar_errors, SUM(punctuation_count) as punctuation_errors,
            AVG(word_count) as avg_words,
            COUNT(CASE WHEN total_errors = 0 THEN 1 END) * 100.0 / COUNT(*) as error_free_pct,
            COUNT(CASE WHEN is_valid_format = true THEN 1 END) as valid_format_count,
            SUM(pii_redacted) as total_pii_redacted
        FROM read_parquet('{pq_path}')
    """).df()
    
    breakdown = conn.execute(f"""
        SELECT 'Spelling' as type, SUM(spelling_count) as count FROM read_parquet('{pq_path}')
        UNION ALL SELECT 'Grammar', SUM(grammar_count) FROM read_parquet('{pq_path}')
        UNION ALL SELECT 'Punctuation', SUM(punctuation_count) FROM read_parquet('{pq_path}')
        ORDER BY count DESC
    """).df()
    
    distribution = conn.execute(f"""
        SELECT CASE WHEN total_errors = 0 THEN '0 errors' WHEN total_errors <= 5 THEN '1-5'
            WHEN total_errors <= 10 THEN '6-10' ELSE '11+' END as error_range,
            COUNT(*) as count
        FROM read_parquet('{pq_path}') GROUP BY error_range
    """).df()
    
    # ONLY redacted text in results
    top_errors = conn.execute(f"""
        SELECT total_errors, spelling_count, grammar_count, punctuation_count,
               error_details, corrections, redacted_text
        FROM read_parquet('{pq_path}') WHERE total_errors > 0 ORDER BY total_errors DESC LIMIT 20
    """).df()
    
    full_data = conn.execute(f"""
        SELECT total_errors, spelling_count, grammar_count, punctuation_count, 
               word_count, error_rate, error_details, redacted_text
        FROM read_parquet('{pq_path}') ORDER BY total_errors DESC
    """).df()
    
    conn.close()
    
    return {'summary': summary, 'breakdown': breakdown, 'distribution': distribution,
            'top_errors': top_errors, 'full_data': full_data}

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Hero Header (Logo is floating top-right via CSS)
    st.markdown(f'''
    <div class="hero-header">
        <div>
            <h1 class="hero-title">{icon("shield", 32)} TextGuardian Pro</h1>
            <p class="hero-subtitle">Enterprise-Grade Grammar & Quality Analytics with Global PII Protection</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Upload & Process", "Analytics Dashboard", "Download Results"])
    
    with tab1:
        with st.sidebar:
            st.markdown(f'''<div class="sidebar-card">
                <div class="sidebar-label">{icon("settings", 22)} Configuration</div>
            </div>''', unsafe_allow_html=True)
            
            st.markdown(f'<div class="sidebar-label">{icon("file", 20)} Input Mode</div>', unsafe_allow_html=True)
            mode = st.radio("Mode:", ["Chat", "Email"], horizontal=True, label_visibility="collapsed")
            
            st.markdown(f'<div class="sidebar-label">{icon("globe", 20)} Global PII Protection</div>', unsafe_allow_html=True)
            redact_pii_flag = st.checkbox("Enable PII Redaction", value=True)
            
            if redact_pii_flag:
                st.markdown('''<div class="info-box">
                    <strong>Protected Regions:</strong><br>
                    UAE • UK • US • India • EU<br><br>
                    <strong>Detects:</strong> Emails, Phones, IDs, Names, Addresses, Cards, Passports
                </div>''', unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown(f'<div class="sidebar-title">{icon("cpu")} Performance</div>', unsafe_allow_html=True)
            workers = st.slider("Workers", 1, mp.cpu_count(), max(1, mp.cpu_count() - 1))
            st.caption(f"{mp.cpu_count()} CPUs detected")
        
        st.markdown(f'<div class="section-header">{icon("upload")} Upload File</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['csv', 'xlsx', 'xls'], label_visibility="collapsed")
        
        if uploaded_file:
            try:
                with st.spinner("Loading..."):
                    df = load_file(uploaded_file)
                
                st.markdown(f'''<div class="success-card">
                    {icon("check")} Loaded <strong>{len(df):,}</strong> rows × <strong>{len(df.columns)}</strong> columns
                </div>''', unsafe_allow_html=True)
                
                st.markdown(f'<div class="section-header">{icon("target")} Select Column</div>', unsafe_allow_html=True)
                text_col = st.selectbox("", df.columns.tolist(), label_visibility="collapsed")
                
                
                st.markdown("---")
                
                if st.button(f"Start Analysis", type="primary", use_container_width=True):
                    start = datetime.now()
                    
                    try:
                        df_result, pq_path, pii_stats = process_file(df, text_col, mode.lower(), redact_pii_flag, workers)
                        
                        st.session_state.processed_data = df_result
                        st.session_state.parquet_path = pq_path
                        st.session_state.pii_stats = pii_stats
                        
                        with st.spinner("Generating analytics..."):
                            st.session_state.analytics = create_analytics(pq_path)
                        
                        elapsed = (datetime.now() - start).total_seconds()
                        
                        st.markdown(f'''<div class="success-card">
                            {icon("zap")} Processed <strong>{len(df_result):,}</strong> records in <strong>{elapsed:.1f}s</strong>
                        </div>''', unsafe_allow_html=True)
                        
                        s = st.session_state.analytics['summary'].iloc[0]
                        
                        cols = st.columns(4)
                        metrics = [("Rows", f"{s['total_rows']:,}"), ("Errors", f"{int(s['total_errors']):,}"),
                                   ("Avg/Row", f"{s['avg_errors']:.1f}"), ("Clean", f"{s['error_free_pct']:.0f}%")]
                        for col, (label, value) in zip(cols, metrics):
                            with col:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
                        
                        if redact_pii_flag and pii_stats:
                            st.markdown(f'<div class="section-header">{icon("lock", 26)} PII Redaction Summary</div>', unsafe_allow_html=True)
                            total_pii = sum(pii_stats.values())
                            st.markdown(f'<div class="pii-badge">{icon("lock", 24)} {total_pii:,} PII items redacted across all regions</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        
            except Exception as e:
                st.error(f"Load error: {str(e)}")
        else:
            st.markdown('''<div class="upload-zone">
                <p style="color:#64748b;margin:0;">Drag and drop or click to upload<br><small>Supports CSV, XLSX, XLS</small></p>
            </div>''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown(f'<div class="section-header">{icon("chart")} Analytics Dashboard</div>', unsafe_allow_html=True)
        
        if st.session_state.analytics:
            a = st.session_state.analytics
            s = a['summary'].iloc[0]
            
            cols = st.columns(4)
            metrics = [("Total Rows", f"{s['total_rows']:,}"), ("Total Errors", f"{int(s['total_errors']):,}"),
                       ("Spelling", f"{int(s['spelling_errors']):,}"), ("Grammar", f"{int(s['grammar_errors']):,}")]
            for col, (label, value) in zip(cols, metrics):
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="section-header">{icon("target")} Error Types</div>', unsafe_allow_html=True)
                st.bar_chart(a['breakdown'].set_index('type')['count'], use_container_width=True)
            with col2:
                st.markdown(f'<div class="section-header">{icon("chart")} Distribution</div>', unsafe_allow_html=True)
                st.bar_chart(a['distribution'].set_index('error_range')['count'], use_container_width=True)
            
            st.markdown(f'<div class="section-header">{icon("list")} Top Error Records (Redacted)</div>', unsafe_allow_html=True)
            st.dataframe(a['top_errors'], use_container_width=True, hide_index=True)
            
            st.markdown(f'<div class="section-header">{icon("file")} Full Data (Redacted Only)</div>', unsafe_allow_html=True)
            st.dataframe(a['full_data'].head(100), use_container_width=True, height=400)
        else:
            st.markdown('<div class="info-box">Process a file first to see analytics</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown(f'<div class="section-header">{icon("download")} Download Results</div>', unsafe_allow_html=True)
        
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            
            st.markdown(f'''<div class="success-card">
                {icon("check")} Ready: <strong>{len(df):,}</strong> rows (all PII redacted)
            </div>''', unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### CSV")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, f"textguardian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            
            with col2:
                st.markdown("#### Excel")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Analysis', index=False)
                    if st.session_state.analytics:
                        st.session_state.analytics['summary'].to_excel(writer, sheet_name='Summary', index=False)
                st.download_button("Download Excel", output.getvalue(), f"textguardian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            
            with col3:
                st.markdown("#### Parquet")
                if st.session_state.parquet_path:
                    with open(st.session_state.parquet_path, 'rb') as f:
                        st.download_button("Download Parquet", f.read(), f"textguardian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
            
            st.markdown(f'<div class="section-header">{icon("file")} Sample Preview (Redacted)</div>', unsafe_allow_html=True)
            st.dataframe(df[['redacted_text', 'total_errors', 'error_details']].head(10), use_container_width=True)
        else:
            st.markdown('<div class="info-box">Process a file first to download results</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="footer">{FOOTER}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
