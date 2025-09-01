import time
import pandas as pd
from typing import List
from tqdm import tqdm
import pdb, traceback, sys, os
from datetime import datetime
import argparse
import requests
from utils import extract, save_json, read_json
from detect_new_sentences import finegrained_diff, finegrained_diff_by_paragraph
from detect_external_link import get_new_urls, get_new_urls_without_date
from preprocess_revision import process_revisions_and_save_initial_data


URL = "https://en.wikipedia.org/w/api.php"

def query_wikipedia(title: List[str], month, year, num_revisions = 100):
    revisions_main = []
    count = 0
    # timestamp = '2024-02-20T21:25:13Z'
    timestamp = None
    print("Fetching data for entity: ", title)
    while True:
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "titles": title,
            "rvprop": "ids|timestamp|content|user|size|tags|comment", 
            "rvslots": "main", 
            "rvlimit": num_revisions,
            # "rvstart": "2024-5-01T19:20:39Z"
            # "explaintext":True
        }
        if timestamp:
            params["rvstart"] = timestamp
        else: 
            params["rvstart"]: f"2025-01-01T19:20:39Z"
        
        # avoid connection Error
        response = requests.get(URL, params=params)
        
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        wiki_data = {}


        for page_id, page in pages.items():
            if page_id not in wiki_data:
                wiki_data[page_id] = {}
            revisions = page.get("revisions", [])
        count += len(revisions)
        for revision in revisions:
            timestamp = revision['timestamp']
            m, y = int(timestamp.split('-')[1]), int(timestamp.split('-')[0])
            if (month == m and year == y) or (year> y): # add year<y becuase some entities might not be edited for a long time. 
                break   
            revisions_main.append(revision)
        
        if count>0:
            if (month == m and year == y) or (year > y):
                break
        else:
            break
    print(f"Total revisions fetched for {title}: {len(revisions_main)}")
    return revisions_main, wiki_data, page_id



def post_process(output):
    data = {}
    # revision_data = train_data[entity_id]
    for revision in output:
        time_of_edit = revision['time_of_edit']
        data[time_of_edit] = {
            'revision_id': revision['revision_id'],
            'parent_revision_id': revision['parent_revision_id'],
            'hierarchy': revision['hierarchy'],
            'section_information': revision['section_information'],
        }
    return data




def fetch_wikipedia(entities, start, end, month, year, dir_path, only_save_edits = False):
    max_retries = 5
    delay = 1

    entitiy_names = list(entities.values())
    name2id = {v: k for k, v in entities.items()}
    entities_list = entitiy_names[start:end]
    
    entity_pageid_list = {}
    
    for entity in tqdm(entities_list, total= len(entities_list)):
        
        one_entity_diff = []

        # get revision
        for attempt in range(max_retries):
            try:
                revision_main, wiki, page_ids = query_wikipedia(entity, month, year, num_revisions=100)
                entity_revisions = process_revisions_and_save_initial_data(revision_main, entity)
                wiki[page_ids] = entity_revisions
                break
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))  
                else:
                    raise
        
        # Extract dataset from the web
        id2text = {}
        id2html = {}
        id2sections = {}
        pageids = list(wiki.keys())[0]
        wikidata = wiki[pageids]
        
        entity_pageid_list[pageids] = entity
        
        for revids, line in wikidata.items():
            id2text[revids] = line['content']
            id2html[revids] = line['raw_html']
            id2sections[revids] = line['sections_information']
        edits = []
        for revids, line in wikidata.items():
            if revids in wikidata.keys() and line['parentid'] in wikidata.keys():
                edits.append((revids, line['parentid']))
            else:
                if line['parentid'] not in wikidata.keys():
                    print(f"Revision {revids} has no parent revision")
                if revids not in wikidata.keys():
                    print(f"Revision {line['parentid']} has no child revision")
             
        
        # Compute the diffs and save the diff data
        for edit in tqdm(edits, total=len(edits), desc="Computing diffs"):
            edit_data_by_section = {}
            edit_data_by_section['revision_id'] = edit[0]
            edit_data_by_section['parent_revision_id'] = edit[1]
            edit_data_by_section['time_of_edit'] = wiki[pageids][edit[0]]['time_of_edit']
            edit_data_by_section['hierarchy'] = wiki[pageids][edit[0]]['hierarchy']
            edit_data_by_section['tags'] = wiki[pageids][edit[0]]['tags']
            edit_data_by_section['user'] = wiki[pageids][edit[0]]['user']
            edit_data_by_section['comment'] = wiki[pageids][edit[0]]['comment']
            edit_data_by_section['size'] = wiki[pageids][edit[0]]['size']
            if edit[1] in id2text: # the last edit pair
                html1 = id2html[edit[1]]
                html2 = id2html[edit[0]]
                 
                # Hierarchy match
                if wiki[pageids][edit[0]]['hierarchy'] == wiki[pageids][edit[1]]['hierarchy']:
                    html1 = id2html[edit[1]]
                    html2 = id2html[edit[0]]
                    # Get the new urls to justify whether there is a new url, if not, skip.
                    new_urls = get_new_urls_without_date(html1, html2)                                    
                    diffs = []
                    paragraph_diffs = []
                    
                    # If there is new urls, we can calculate the diff section by section                
                    
                    # New urls are detected, we can assume new edit is added.
                    if len(new_urls) > 0:
                        new_urls = []
                        
                        wiki[pageids][edit[0]]['insert_new_section'] = False
                        old_sections = id2sections[edit[1]]
                        new_sections = id2sections[edit[0]]
   
                        for section_id in old_sections: 
                            # decide wether this section contain new url. 
                            old_html = old_sections[section_id]['raw_content']
                            new_html = new_sections[section_id]['raw_content']
                            
                            new_url = get_new_urls(old_html, new_html)
                            
                            old_text = old_sections[section_id]['content']
                            new_text = new_sections[section_id]['content']
                            
                            diff = finegrained_diff(old_text, new_text)
                            paragraph_diff = finegrained_diff_by_paragraph(old_text, new_text)
                            
                            
                            diffs.extend(diff)
                            paragraph_diffs.extend(paragraph_diff)
                            
                            wiki[pageids][edit[0]]['sections_information'][section_id]['old_content'] = old_text
                            wiki[pageids][edit[0]]['sections_information'][section_id]['sentence_edit'] = diff
                            wiki[pageids][edit[0]]['sections_information'][section_id]['paragraph_edit'] = paragraph_diff
                            wiki[pageids][edit[0]]['sections_information'][section_id]['new_urls'] = new_url
                            
                            new_urls.extend(new_url)
                            
                        wiki[pageids][edit[0]]['sentence_edit'] = diffs
                        wiki[pageids][edit[0]]['paragraph_edit'] = paragraph_diffs
                        wiki[pageids][edit[0]]['new_urls'] = new_urls
                        wiki[pageids][edit[0]]['insert_new_section'] = True
                        
                
                    # No new urls, we can assume no new edit is added.
                    else: 
                        wiki[pageids][edit[0]]['insert_new_section'] = False
                        wiki[pageids][edit[0]]['sentence_edit'] = diffs
                        wiki[pageids][edit[0]]['paragraph_edit'] = paragraph_diffs
                        wiki[pageids][edit[0]]['new_urls'] = []
                    
                    edit_data_by_section['section_information'] = wiki[pageids][edit[0]]['sections_information']
                    edit_data_by_section['insert_new_section'] = False
                    edit_data_by_section['new_urls'] = new_urls
                    edit_data_by_section['sentence_edit'] = diffs
                    edit_data_by_section['paragraph_edit'] = paragraph_diffs
                    
                
                # Hierarchy not match. Calculate Diff by The whole page. 
                else: 
                    # if wiki[pageids][edit[0]]['hierarchy'] == wiki[pageids][edit[1]]['hierarchy']:
                    wiki[pageids][edit[0]]['insert_new_section'] = True 
                    
                    text1 = id2text[edit[1]] # parent 
                    text2 = id2text[edit[0]] # edited
                    html1 = id2html[edit[1]]
                    html2 = id2html[edit[0]]
                    
                    new_urls = get_new_urls_without_date(html1, html2)
                        # new_urls = get_new_urls_without_date(html1, html2)
                    new_urls = get_new_urls(html1, html2)
                    
                    wiki[pageids][edit[0]]['new_urls'] = new_urls
                    
                    diff = finegrained_diff(text1, text2)
                    paragraph_diff = finegrained_diff_by_paragraph(text1, text2)
                    
                    wiki[pageids][edit[0]]['sentence_edit'] = diff
                    wiki[pageids][edit[0]]['paragraph_edit'] = paragraph_diff
                    
                    wiki[pageids][edit[0]]['new_section'] = []
                    
                    edit_data_by_section['insert_new_section'] = True
                    edit_data_by_section['section_information'] = wiki[pageids][edit[0]]['sections_information']
                    edit_data_by_section['new_urls'] = new_urls
                    edit_data_by_section['sentence_edit'] = diff
                    edit_data_by_section['paragraph_edit'] = paragraph_diff

            else:  # Edit[1] not in the revisions. 
                wiki[pageids][edit[0]]['sentence_edit'] = []
                wiki[pageids][edit[0]]['paragraph_edit'] = []
                wiki[pageids][edit[0]]['new_urls'] = []
                # wiki[pageids][edit[0]]['new_section'] = []
                
                edit_data_by_section['section_information'] = wiki[pageids][edit[0]]['sections_information']
                edit_data_by_section['new_urls'] = []
                edit_data_by_section['insert_new_section'] = False
                edit_data_by_section['sentence_edit'] = []
                edit_data_by_section['paragraph_edit'] = []
                print("Operation timed out")
            
            has_external_url = False
            if len(wiki[pageids][edit[0]]['new_urls'])>0:
                has_external_url = True
            
            has_text_edit = False
            if len(wiki[pageids][edit[0]]['sentence_edit'])>0:
                for one_diff in wiki[pageids][edit[0]]['sentence_edit']:
                    if one_diff[1]=='insert':
                        has_text_edit = True
                        break
            

            #justify the new/old flag
            new_flag = 'Old'
            if has_external_url and has_text_edit:
                time_of_edit = wiki[pageids][edit[0]]['time_of_edit'].replace('T', ' ').replace('Z', '')
                time_of_edit = pd.to_datetime(time_of_edit)
                
                time_of_urls  = [extract(url[0]) for url in wiki[pageids][edit[0]]['new_urls']]
                time_of_urls = [time_of_edit if time is None else time for time in time_of_urls]
                try: 
                    time_delta = [(time_of_edit - pd.to_datetime(time)).days for time in time_of_urls]
                    new_flag = 'New' if any([time < 60 for time in time_delta]) else 'Old'
                except: 
                    new_flag = 'New'
                
            wiki[pageids][edit[0]]['new_flag'] = new_flag
            
            edit_data_by_section['new_flag'] = new_flag
            
            if only_save_edits:
                if has_external_url or has_text_edit:
                    one_entity_diff.append(edit_data_by_section)
            else:
                one_entity_diff.append(edit_data_by_section)

        
        for revision in one_entity_diff:
            for section_id in revision['section_information']:
                revision['section_information'][section_id].pop('raw_content')

    
        path = os.path.join(dir_path, f'{name2id[entity]}.json')
        # post process the revisions and save the initial data
        processed_output = post_process(one_entity_diff)
        save_json(processed_output, path = path)



if __name__ == "__main__":
    
    # Configure logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity_list', type=str,default='entities.json',help='The entity list to fetch data')
    parser.add_argument('--dir_path', type=str,default='data',help='Directory to save the data')
    parser.add_argument('--month',type=int,default=2,help='Month to fetch data')
    parser.add_argument('--year',type=int,default=2024,help='Year to fetch data')
    parser.add_argument('--start',type=int,default=0,help='Start index of entities')
    parser.add_argument('--end',type=int,default=50,help='End index of entities')
    parser.add_argument('--only_save_edits',action='store_true',help='Only save the edits')
    args = parser.parse_args()

    # Read entities from the file
    entities = read_json(args.entity_list)
    
    try:
        # use_five_entities = True
        date = time.strftime("%Y%m%d")        
        fetch_wikipedia(
            entities = entities,
            start = args.start,
            end = args.end,
            month = args.month, 
            year = args.year,
            dir_path = args.dir_path,
            only_save_edits=args.only_save_edits, 
            )
        
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)