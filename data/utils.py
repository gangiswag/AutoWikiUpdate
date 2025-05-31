import json
from htmldate import find_date
import re
import os
import requests

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException



def extract_old(url):
    # Regular expression to extract the archive date
    patterns = [
        r'/web/(\d{4})(\d{2})(\d{2})',  # web archive date pattern
        r'/(\d{4})/(\d{2})/(\d{2})/'    # URL date pattern
    ]

    date = None
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            year, month, day = match.groups()
            # convert to datetime object
            date = f"{year}-{month}-{day}"
            break
    return date

def extract(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful

        # Use htmldate to find the publication date
        publication_date = find_date(url, response.text)

    except:
        print(f"Error fetching URL")
        return None
    
    return publication_date
    

# helper functions
def save_json(data, path = 'test.json'):
    # ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    

    
    
