from tqdm import tqdm
import re
import wikitextparser as wtp



def parse_wikipedia_sections(content):
    setions_list = {}
    section_pattern = re.compile(r'(?m)^(={2,6})\s*(.*?)\s*\1$')
    # Split content by section headers
    matches = list(section_pattern.finditer(content))
    if len(matches) != 0:
        sections = []
        for i,match in enumerate(matches):
            level = len(match.group(1))
            section_title = match.group(2)

            section_dict = {}
            section_dict['title'] = section_title
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_dict['raw_content'] = content[start:end].strip()
            # section_dict['content'] = mwparserfromhell.parse(section_dict['raw_content']).strip_code()
            
            c = wtp.parse(section_dict['raw_content'])
            
            section_dict['content'] = c.plain_text()
            section_dict['sentence_edit'] = []
            section_dict['paragraph_edit'] = []
            section_dict['new_urls'] = []
            
            
            section_dict['has_table'] = [t.data() for t in c.tables]
            
            
            hierarchy_information = {
                'index': str(i+1),
                'line': section_title,
                'level': level-2,
            }
            section_dict['hierachy_information'] = hierarchy_information
            
            # section_dict['level'] = level
            
            sections.append(hierarchy_information)
            setions_list[str(i+1)] = section_dict
        
        intro_end = matches[0].start()
        introduction = content[:intro_end].strip()
        
        setions_list['0'] = { 
            "title":"Introduction",
            "content":wtp.parse(introduction).plain_text(),
            "raw_content":introduction,
            "sentence_edit": [],
            "paragraph_edit": [],
            "new_urls": [],
            "hierachy_information": {
                'index': '0',
                'line': 'Introduction',
                'level': 0}
            }
        # assert len(setions_list) == len(hierarchy) +1
        return setions_list, sections

def track_section_level(sections):
    level_stack = []  
    hierarchy = []
    # hierarchy.append({"index": "0", "name": "(Top)", "level": 0})
    
    for sec in sections:
        section_index = sec["index"]
        section_name = sec["line"]
        section_level = int(sec["level"]) 

        while level_stack and level_stack[-1][1] >= section_level:
            level_stack.pop() 

        if level_stack:
            full_path = " > ".join([s[0] for s in level_stack] + [section_name])
        else:
            full_path = section_name  

        hierarchy.append((section_index, full_path))

        level_stack.append((section_name, section_level))  
        
    # print(hierarchy)
    return hierarchy


# parse the data and initialize the dataset

def process_revisions_and_save_initial_data(revisions_main, title):
    temperary_file_path = f"temp_revisions.json"
    entity_revisions = {}
    for revision in tqdm(revisions_main[:], total=len(revisions_main), desc=f"Processing revisions for entity: {title}"):
        # from IPython import embed; embed(); exit()
        # try: 
            # get the  of current revision
        content = revision.get("slots", {}).get("main", {}).get("*", "")
        # if content == '':
        #     content = entity_revisions[revid]['raw_html']
        
        # plain_text = mwparserfromhell.parse(content).strip_code()
        plain_text = wtp.parse(content).plain_text()
        revid = revision['revid']
        try:
            sections_information,sections = parse_wikipedia_sections(content)
        except: 
            print(f"Error in parsing section information for revision: {revid}. Skipping.........")
            continue
        hierarchy = track_section_level(sections)
        
        if 'comment' in revision:
            comment = revision['comment']
        else:
            comment = ''
            
        if 'user' in revision:
            user = revision['user']
        else:
            user = ''
        
        if 'tags' in revision:
            tags = revision['tags']
        else:
            tags = []
        
        entity_revisions[revid] = {
            'parentid': revision['parentid'],
            'time_of_edit': revision['timestamp'],
            'comment': comment,
            'user': user,
            'size': revision['size'],
            'tags': tags,
            'content': plain_text,
            'raw_html': content,
            
            'hierarchy': hierarchy,
            'sections_information': sections_information,
            'sentence_edit': [],
            'paragraph_edit': [],
                
            'new_urls': [],
            'insert_new_section': False,
            'new_flag': None
        }

            # print(f"Revision: {revid} has {len(sections_information)} sections.")
        '''
        except: 
            print(f"Error in processing revision: {revid}. Skipping.........")
            continue
        '''
    # save_json(entity_revisions, path = temperary_file_path)
    return entity_revisions 
