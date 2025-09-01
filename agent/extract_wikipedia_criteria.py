from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import json
import datetime
import os
import argparse
import tiktoken
from utils import read_json

CRITERIA_PROMPT = """
You are an assistant tasked with proposing criteria for adding or updating information on the Wikipedia page of {page_title}. You are collaborating with a knowledge update assistant that will scan news articles for recent developments related to {page_title}. This knowledge update assistant will rely on your criteria to determine which updates are both relevant and sufficiently noteworthy to warrant inclusion on the Wikipedia page of {page_title}.

Your mission is to examine the current version of the Wikipedia page and produce a set of criteria indicating what kind of information could be relevant for each corresponding section, subsection and subsubsection. 

Below is the outline of the page, with sections at L1, corresponding subsections at L2 and so on:

### INDEX:
{wiki_sections}

Here is the full Wikipedia page content:

### CONTENT:
{wiki_content}

### INSTRUCTIONS:
Now, please provide your criteria for adding or including updates. Output a dedicated set of criteria for each section, subsection and sub-subsection. If a section only contains subsections and has no standalone content, do not provide criteria for the main section; instead, supply criteria for its subsections directly. Whenever possible, we need to ensure that an update is assigned to only one relevant section or subsection-therefore, avoid repeating or duplicating the same criteria in multiple places. You may also skip sections or subsections that contain minimal information (e.g., a few sentences or bullet points), as well as sections covering references, footnotes, citations, further reading, external links, or other miscellaneous content.

Your response must be a dictionary that Python's json.loads can parse. Use the format below:

{{
    "criteria":
        [
            {{
                "name": "Use <section> if criteria apply to a section, <section>_<subsection> if they apply to a subsection, or <section>_<subsection>_<subsubsection> if they apply at a sub-subsection level. Ensure that all spaces are replaced with underscores. Copy the section, subsection, or sub-subsection title exactly as it appears in the Content.",
                "thoughts": "Explanation for why these criteria are proposed",
                "criteria": [
                    "Provide concise criteria here, with no more than four criteria. Make sure the criteria are generic and extensively cover the section/subsection/sub-subsection."
                ]
            }},
            {{
                ...additional entries for other sections, subsections, sub-subsections...
            }}
        ]
}}
"""

def get_outline(outline, indent, hierarchy):
    for item in hierarchy:
        # item format: [id, "Section or Parent > Child > ... > Title"]
        full_title = item[1]
        # Level is determined by the number of '>' characters plus one
        level = full_title.count('>') + 1
        # Extract the actual title (the text after the final '>')
        title = full_title.split('>')[-1].strip()
        # Compute indentation: level-1 tabs for nested sections
        current_indent = indent * (level - 1)
        outline += current_indent + "L" + str(level) + ": " + title + "\n"
    return outline

def get_section_content(section_content, sections, hierarchy):
    # sections = {1: {"title": "Section Title", "content": "Section Content"}, 2: {...}, ...}
    key_map = {'0': 'Summary'}
    for line in hierarchy:
        prefix = line[1].replace(" > ", "_").replace(" ", "_")
        key_map[line[0]] = prefix
    for idx, section_text in sections.items():
        section_prefix = key_map[idx]
        if section_text['content']:
            section_content[section_prefix] = section_text['content'].strip()

    return section_content


def combine_sections(page):
    full_text = ""
    idx = 0
    while str(idx) in page:
        section_content = page[str(idx)]
        full_text += section_content['title'] +  "\n" + section_content['content'].strip() + "\n\n"
        idx += 1
    return full_text


def get_wiki_criteria(page_title, page_data):

    full_text = "Summary\n" + combine_sections(page_data['section_information'])
    outline = get_outline("L1: Summary\n", "\t", page_data['hierarchy']) 
    section_content = get_section_content({}, page_data['section_information'], page_data['hierarchy'])
    inp_prompt = CRITERIA_PROMPT.format(page_title=page_title, 
                                        wiki_sections=outline, 
                                        wiki_content=full_text)

    llm = ChatOpenAI(model=args.model, max_tokens=32000, temperature=0)

    messages = [{"role": "user", "content": inp_prompt}]
    
    structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)
    with get_openai_callback() as cb:
        response = structured_llm.invoke(messages)

    criteria_cost = float(cb.total_cost)
    # can save to track cost of runs

    if response["parsing_error"]:
        print("Failed to parse content")
        # abort the process
        output = {
            "criteria": [],
            "section_content": section_content,
            "page_name": page_title,
            "error": response["parsing_error"]
        }
    else:
        output = response["parsed"]


    output["section_content"] = section_content
    output["page_name"] = page_title
    for item in output["criteria"]:
        if item["name"] not in output["section_content"]:
            # if not present find the closest section and replace the name
            closest_section = min(output["section_content"].keys(), key=lambda x: len(set(x.split("_")) & set(item["name"].split("_"))))
            item["name"] = closest_section
            print("Section name '", item["name"], "' is not present, replacing it the closest matched section - '", closest_section, "'")
    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Model to use for navigation",
        default="gpt-4o",
        type=str
    )
    parser.add_argument(
        "--page_id",
        help="ID of the Wiki Page",
        required=False,
        default="307",
        type=str
    )
    parser.add_argument(
        "--entity_file",
        help="json file containing entity ID to name mapping",
        required=True,
        type=str
    )
    parser.add_argument(
        "--data_dir",
        help="Directory containing the wikipedia data",
        required=True,
        type=str
    )

    parser.add_argument(
        "--out_dir",
        help="Output directory to store criteria and page content",
        required=True,
        default="output",
        type=str
    )
    parser.add_argument(
        "--time_delta",
        help="Time delta for querying",
        required=True,
        default=14,
        type=int
    )
    parser.add_argument(
        "--time_delta_file",
        help="Time delta file to store the time delta between runs",
        required=False,
        default="time_delta_between_runs.json",
        type=str
    )
    args = parser.parse_args()

    entity_lookup = read_json(args.entity_file)
    page = read_json(f'{args.data_dir}/{args.page_id}.json')

    # query biweekly
    dates = list(page.keys())
    dates.sort()
    # get the starting date
    date = datetime.datetime.strptime(dates[0], "%Y-%m-%dT%H:%M:%SZ")
    end_date = datetime.datetime.strptime(dates[-1], "%Y-%m-%dT%H:%M:%SZ")
    time_delta_between_runs = {}
    previous_run_date = None
    while date <= end_date:
        # Filter out dates that are in the past relative to the current date
        future_dates = [d for d in dates if datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ") >= date]
        if not future_dates:
            break

        # Find the closest available date key from future_dates
        closest_date = min(
            future_dates,
            key=lambda d: datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ") - date
        )
        current_run_date = datetime.datetime.strptime(closest_date, "%Y-%m-%dT%H:%M:%SZ")
        if previous_run_date is not None:
            delta_days = (current_run_date - previous_run_date).days
            time_delta_between_runs[previous_run_date.strftime("%Y-%m-%dT%H:%M:%SZ")] = delta_days # time[T] = time between run T and T+1
        previous_run_date = current_run_date

        criteria = get_wiki_criteria(entity_lookup[args.page_id], page[closest_date])
        # Ensure the output file path is correctly created (open file for writing)
        output_path = os.path.join(args.out_dir, f"{args.page_id}_{closest_date}.json")

        with open(output_path, 'w') as outfile:
            json.dump(criteria, outfile, indent=4)
        print("Completed extraction for date: ", closest_date)

        # Increment the date by the time delta (ensures moving forward)
        date = datetime.datetime.strptime(closest_date, "%Y-%m-%dT%H:%M:%SZ") + datetime.timedelta(days=args.time_delta)
        
    # Add default time delta for the last run
    if previous_run_date is not None:
        time_delta_between_runs[previous_run_date.strftime("%Y-%m-%dT%H:%M:%SZ")] = args.time_delta
    # Save the time delta between runs to a file
    with open(args.time_delta_file, 'w') as f:
        json.dump(time_delta_between_runs, f, indent=4)