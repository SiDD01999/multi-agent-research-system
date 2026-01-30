"""
Web scraper for NihongoNext website using Selenium
Handles React SPA content extraction
"""

import time
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

import sys
sys.path.append('..')
from config.settings import (
    TARGET_URL, 
    PAGES_TO_SCRAPE, 
    SELENIUM_WAIT_TIME,
    HEADLESS_MODE
)


class NihongoNextScraper:
    """
    Selenium-based web scraper for NihongoNext React SPA
    """
    
    def __init__(self):
        self.driver = None
        self.base_url = TARGET_URL
        
    def _init_driver(self):
        """Initialize Chrome WebDriver with appropriate options"""
        chrome_options = Options()
        
        if HEADLESS_MODE:
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    def _close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            
    def _wait_for_react_render(self):
        """Wait for React content to render"""
        time.sleep(3)  # Initial wait for React hydration
        
        try:
            # Wait for the root div to have content
            WebDriverWait(self.driver, SELENIUM_WAIT_TIME).until(
                lambda d: len(d.find_element(By.ID, "root").text) > 100
            )
        except Exception:
            # If waiting fails, continue anyway
            pass
            
    def _extract_page_content(self) -> Dict[str, str]:
        """Extract structured content from current page"""
        self._wait_for_react_render()
        
        # Get page source after JavaScript execution
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
            
        # Extract different content types
        content = {
            "url": self.driver.current_url,
            "title": self.driver.title,
            "headings": [],
            "paragraphs": [],
            "lists": [],
            "links": [],
            "full_text": ""
        }
        
        # Extract headings
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = heading.get_text(strip=True)
            if text:
                content["headings"].append(text)
                
        # Extract paragraphs
        for para in soup.find_all('p'):
            text = para.get_text(strip=True)
            if text and len(text) > 20:  # Filter out short snippets
                content["paragraphs"].append(text)
                
        # Extract list items
        for li in soup.find_all('li'):
            text = li.get_text(strip=True)
            if text:
                content["lists"].append(text)
                
        # Extract navigation links
        for link in soup.find_all('a', href=True):
            text = link.get_text(strip=True)
            href = link['href']
            if text and not href.startswith('#'):
                content["links"].append({"text": text, "href": href})
                
        # Get full visible text
        body = soup.find('body')
        if body:
            content["full_text"] = body.get_text(separator=' ', strip=True)
            # Clean up multiple spaces
            content["full_text"] = ' '.join(content["full_text"].split())
            
        return content
    
    def scrape_page(self, path: str) -> Dict[str, str]:
        """Scrape a single page"""
        url = f"{self.base_url}{path}"
        print(f"Scraping: {url}")
        
        try:
            self.driver.get(url)
            return self._extract_page_content()
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {"url": url, "error": str(e), "full_text": ""}
    
    def scrape_all_pages(self) -> List[Dict[str, str]]:
        """Scrape all configured pages"""
        self._init_driver()
        results = []
        
        try:
            for path in PAGES_TO_SCRAPE:
                content = self.scrape_page(path)
                results.append(content)
                time.sleep(1)  # Be respectful to the server
        finally:
            self._close_driver()
            
        return results
    
    def get_combined_text(self) -> str:
        """Get all scraped content as a single text document"""
        all_content = self.scrape_all_pages()
        
        combined_parts = []
        for page in all_content:
            if page.get("full_text"):
                section = f"\n\n=== PAGE: {page.get('url', 'Unknown')} ===\n"
                section += f"Title: {page.get('title', 'N/A')}\n\n"
                
                if page.get("headings"):
                    section += "Headings:\n" + "\n".join(f"- {h}" for h in page["headings"]) + "\n\n"
                    
                section += f"Content:\n{page['full_text']}"
                combined_parts.append(section)
                
        return "\n".join(combined_parts)


def main():
    """Test the scraper"""
    scraper = NihongoNextScraper()
    content = scraper.get_combined_text()
    print(f"Scraped {len(content)} characters of content")
    print("\n--- First 2000 characters ---\n")
    print(content[:2000])
    

if __name__ == "__main__":
    main()
