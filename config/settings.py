"""
Configuration settings for Multi-Agent Research System
"""

# ============================================================
# TARGET WEBSITE - Change this to analyze different sites
# ============================================================
TARGET_URL = "https://www.example.com"
PAGES_TO_SCRAPE = [
    "/",           # Homepage
    "/about",      # About page
    "/pricing",    # Pricing page
    "/features",   # Features page
]

# Selenium configuration
SELENIUM_WAIT_TIME = 10  # seconds to wait for page load
HEADLESS_MODE = True     # Run browser in headless mode

# ============================================================
# VECTOR DATABASE (ChromaDB)
# ============================================================
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "website_content"

# Text chunking configuration
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # overlap between chunks

# ============================================================
# LLM CONFIGURATION (Groq)
# ============================================================
GROQ_MODEL = "llama-3.3-70b-versatile"

# ============================================================
# OUTPUT
# ============================================================
OUTPUT_DIR = "./output"
OUTPUT_FILENAME = "Business_Proposal.md"
