"""
Styles for the Streamlit UI
"""

NOTION_STYLE = """
<style>
    /* Global Notion-like styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main container styles */
    .main {
        font-family: 'Inter', sans-serif;
        color: #000000; /* Default text color black */
        line-height: 1.5;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 24px;
    }
    
    /* Custom scrollbar for Notion feel */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(55, 53, 47, 0.2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(55, 53, 47, 0.4);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #000000; /* Black header text */
        margin-bottom: 1rem;
        line-height: 1.3;
    }
    
    h1 {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-top: 1rem;
    }
    
    h2 {
        font-size: 1.8rem;
        letter-spacing: -0.02em;
        border-bottom: 1px solid rgba(55, 53, 47, 0.09);
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        font-size: 1.4rem;
        letter-spacing: -0.01em;
    }
    
    p {
        color: #000000; /* Black paragraph text */
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        background: #2F3437;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        transition: all 0.15s ease;
        box-shadow: rgba(15, 15, 15, 0.05) 0px 1px 2px;
    }
    
    .stButton button:hover {
        background: #454B4E;
        transform: translateY(-1px);
        box-shadow: rgba(15, 15, 15, 0.1) 0px 2px 4px;
    }
    
    .stButton button:active {
        transform: translateY(0px);
    }
    
    .stButton button[kind="secondary"] {
        background: #F7F8F9;
        color: #2F3437;
        border: 1px solid rgba(47, 52, 55, 0.1);
    }
    
    .stButton button[kind="secondary"]:hover {
        background: #EBECED;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #37352F;
        background: #F7F8F9;
        border-radius: 4px;
        padding: 12px 16px;
        transition: background 0.15s;
    }
    
    .streamlit-expanderHeader:hover {
        background: #EBECED;
    }
    
    details {
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid rgba(47, 52, 55, 0.1);
        margin-bottom: 16px;
        box-shadow: rgba(15, 15, 15, 0.03) 0px 2px 3px;
    }
    
    details > div {
        padding: 18px;
        background: white;
    }
    
    /* Cards */
    .notion-card {
        background: white;
        border: 1px solid rgba(55, 53, 47, 0.1);
        border-radius: 4px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: rgba(15, 15, 15, 0.03) 0px 2px 8px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .notion-card:hover {
        box-shadow: rgba(15, 15, 15, 0.08) 0px 4px 12px;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #2F3437;
        height: 6px;
        border-radius: 3px;
    }
    
    .stProgress {
        margin: 10px 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: #F7F8F9;
        padding: 4px;
        border-radius: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        background-color: transparent;
        border: none;
        color: #37352F;
        font-weight: 500;
        transition: all 0.15s;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(47, 52, 55, 0.05);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        color: #2F3437;
        box-shadow: rgba(15, 15, 15, 0.05) 0px 1px 2px;
        border-radius: 4px;
    }
    
    /* Code blocks */
    .highlight, code {
        font-family: 'SF Mono', SFMono-Regular, ui-monospace, Menlo, monospace;
        font-size: 0.9rem;
    }
    
    .highlight {
        background: #F7F8F9;
        border-radius: 4px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: rgba(15, 15, 15, 0.03) 0px 1px 3px inset;
        border: 1px solid rgba(47, 52, 55, 0.1);
    }
    
    /* File management section */
    .file-management {
        background: white;
        border: 1px solid rgba(55, 53, 47, 0.1);
        border-radius: 4px;
        padding: 24px;
        margin: 24px 0;
        box-shadow: rgba(15, 15, 15, 0.03) 0px 2px 8px;
    }
    
    .download-button {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 12px 0;
        font-weight: 500;
    }
    
    /* Info, warning, error messages */
    .element-container div[data-testid="stAlert"] {
        padding: 16px;
        border-radius: 4px;
        margin: 16px 0;
        border: 1px solid;
    }
    
    /* Tables */
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 16px 0;
    }
    
    table th {
        background: #F7F8F9;
        padding: 12px;
        text-align: left;
        font-weight: 500;
        color: #37352F;
        border-bottom: 1px solid rgba(55, 53, 47, 0.1);
    }
    
    table td {
        padding: 12px;
        border-bottom: 1px solid rgba(55, 53, 47, 0.05);
    }
    
    table tr:last-child td {
        border-bottom: none;
    }
    
    /* Dividers */
    hr {
        height: 1px;
        background-color: rgba(55, 53, 47, 0.09);
        border: none;
        margin: 32px 0;
    }
    
    /* Main content area */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
