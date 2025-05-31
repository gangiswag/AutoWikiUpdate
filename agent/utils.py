import difflib
import json
import re
import html2text
from bs4 import BeautifulSoup
import os
import io
import aiohttp
import pymupdf as fitz
import pymupdf4llm
from htmldate import find_date
import html2text



def convert_html_to_text(html):
        soup = BeautifulSoup(html, 'html.parser')
        converter = html2text.HTML2Text()
        converter.ignore_links = True   
        converter.ignore_images = True
        converter.body_width = 0  # Don't wrap text
        converter.protect_links = True  # Don't convert links to references

        # Find the main content area - you might need to adjust this selector based on the site structure
        # For example, looking for the main content area:
        main_content = soup.find('main') or soup.find('div', class_='content') or soup.find('div', id='content')

        if main_content:
            # Convert just the main content
            markdown_content = converter.handle(str(main_content))
        else:
            # If can't find specific content area, convert the whole body or a subset
            articles = soup.find_all('article')
            if articles:
                markdown_content = ""
                for article in articles:
                    markdown_content += converter.handle(str(article)) + "\n---\n\n"
            else:
                # Fall back to converting body content
                body = soup.find('body')
                markdown_content = converter.handle(str(body))

        return markdown_content

async def scrape_url(url):
    BRIGHTDATA_API_KEY = os.environ.get("BRIGHTDATA_API_KEY", None)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.brightdata.com/request", 
            headers={"Authorization": f"Bearer {BRIGHTDATA_API_KEY}"}, 
            json={"url": url, "format": "raw", "zone": "web_unlocker1"}
        ) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                pdf_content = await response.read()
                with io.BytesIO(pdf_content) as pdf_file:
                    with fitz.open(stream=pdf_file, filetype="pdf") as doc:
                        content = pymupdf4llm.to_markdown(doc, write_images=False)
                
                article_date = "N/A"            
            else:
                html = await response.text()
                content = convert_html_to_text(html)
                try:
                    publish_date = find_date(url, html)
                    article_date = publish_date.strftime("%b %d, %Y")
                except Exception as e:
                    article_date = "N/A"

    return content, article_date

def get_text_diff(a: str, b: str):
    # Split sentences using multiple delimiters
    sentence_endings = re.compile(r'(?<=[.!?…])\s+')
    a_sentences = sentence_endings.split(a)
    b_sentences = sentence_endings.split(b)
    
    # Find sentences that are different
    a_unique = [s for s in a_sentences if s not in b_sentences]
    b_unique = [s for s in b_sentences if s not in a_sentences]
    
    a_diff = " ".join(a_unique)
    b_diff = " ".join(b_unique)
    
    # Generate a human-readable explanation
    explanation = []
    if True:
        diff = list(difflib.ndiff(a.split(" "), b.split(" ")))
        changes = []
        for d in diff:
            if d.startswith("- "):
                changes.append(f"Removed: '{d[2:]}'")
            elif d.startswith("+ "):
                changes.append(f"Inserted: '{d[2:]}'")
        if changes:
            explanation.append("; ".join(changes))
    
    textualized_difference = " | ".join(explanation) if explanation else "No significant differences."
    
    return a_diff, b_diff, textualized_difference


def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data