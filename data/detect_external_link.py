import requests
import re
import os
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pdb, traceback, sys
from htmldate import find_date
import re
import signal
from utils import *
import wikitextparser as wtp


def timeout_handler(signum, frame):
    raise TimeoutError


def get_page_content_at_revision(page_title, revision_id):
    """
    Fetches the entire Wikipedia page content at a specific revision ID.
    
    Args:
        page_title (str): The Wikipedia page title.
        revision_id (int or str): The specific Wikipedia revision ID.

    Returns:
        str: The full content of the page at the specified revision.
    """
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": page_title,
        "rvprop": "content",
        "rvstartid": revision_id,
    }

    response = requests.get(base_url, params=params).json()
    pages = response.get("query", {}).get("pages", {})

    for _, page_data in pages.items():
        return page_data.get("revisions", [{}])[0].get("*", "")

    return ""

def extract_external_links(content):
    """
    Extracts all external (non-Wikipedia) links from Wikipedia page content.

    Args:
        content (str): The Wikipedia page content.

    Returns:
        set: A set of external links (URLs).
    """
    # Regex to match external links (excluding Wikilinks)
    # external_link_pattern = r'https?://[^\s<>\[\]]+'
    # external_link_pattern = r'https?://[^\s<>\[\]{}"\'(),]+[a-zA-Z0-9]'
    # all_links = set(re.findall(external_link_pattern, content))

    # Filter out Wikipedia internal links
    # none_wiki_links = { link.rstrip('}"),') for link in all_links if "wikipedia.org" not in link}
    wl1 = wtp.parse(content).external_links
    # wl2 = wtp.parse(content).external_links
    none_wiki_links = set([i.url.rstrip(r'\/}"),') for i in wl1])

    return none_wiki_links

def get_new_urls(old_text,new_text):
    old_links = extract_external_links(old_text)
    new_links = extract_external_links(new_text)
    added_links = new_links - old_links
    
    link_and_date = []
    for link in list(added_links):
        date = extract(link)
        link_and_date.append([link,date])
    
    return link_and_date
    
def get_new_urls_without_date(old_text,new_text):
    old_links = extract_external_links(old_text)
    new_links = extract_external_links(new_text)
    added_links = new_links - old_links
    
    return added_links
    
    
def detect_new_external_links(page_title, old_revision_id, new_revision_id):
    """
    Compares two revisions of a Wikipedia page and detects newly added external links.

    Args:
        page_title (str): The Wikipedia page title.
        old_revision_id (int): The previous revision ID.
        new_revision_id (int): The newer revision ID.

    Returns:
        list: A list of newly added external URLs.
    """
    # Get content for both revisions
    old_content = get_page_content_at_revision(page_title, old_revision_id)
    new_content = get_page_content_at_revision(page_title, new_revision_id)

    # Extract external links
    old_links = extract_external_links(old_content)
    new_links = extract_external_links(new_content)

    # Identify new external links
    added_links = new_links - old_links

    return list(added_links)
