"""
============================================================
CONFIGURATION FILE - Edit this to customize your analysis
============================================================
"""

# ============================================================
# TARGET WEBSITE
# ============================================================
TARGET_WEBSITE = "www.example.com"

# ============================================================
# WEBSITE CONTENT
# Paste your website content here, or describe the business
# ============================================================
WEBSITE_CONTENT = """
[PASTE YOUR WEBSITE CONTENT OR BUSINESS DESCRIPTION HERE]

Example format:
- Company Name: 
- Industry: (E-commerce / EdTech / SaaS / Healthcare / etc.)
- Current Products/Services:
- Target Audience:
- Pricing Model:
- Key Features:
- Technology Used:
- Competitors:
"""

# ============================================================
# ANALYSIS QUERY
# What do you want the agents to focus on?
# ============================================================
USER_QUERY = """
Analyze the website and propose:
1. Gaps in current offerings
2. AI-driven features to improve user experience
3. Technical roadmap for implementation
4. Monetization opportunities
"""

# ============================================================
# LLM SETTINGS
# ============================================================
GROQ_MODEL = "llama-3.3-70b-versatile"  # Groq model to use

# ============================================================
# OUTPUT SETTINGS
# ============================================================
OUTPUT_DIR = "./output"

# ============================================================
# WEB SCRAPING SETTINGS (for main.py - optional)
# ============================================================
PAGES_TO_SCRAPE = ["/", "/about", "/pricing", "/features"]
SELENIUM_WAIT_TIME = 10
HEADLESS_MODE = True

# ============================================================
# VECTOR DATABASE SETTINGS (for main.py - optional)
# ============================================================
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "website_content"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
