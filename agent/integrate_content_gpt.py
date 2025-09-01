from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import json
import os
import argparse
from datetime import datetime
from vllm import LLM, SamplingParams
import copy
from utils import get_text_diff, read_json

EDIT_PROMPT = """You are a helpful assistant to integrate a piece of news information into a Wikipedia article. You should read the original paragraph, find where and how to insert the news information, and return to me a new paragraph with the news information integrated. You should do the following when integrating the news information:
1. Only integrate objective news information instead of subjective opinions and commentaries.
2. Make less change as possible to the original paragraph.
3. Make sure the new paragraph is coherent and grammatically correct.

**Original Paragraph**
{original}

**News Information**
{news}

**Updated Paragraph**
"""


SYSTEM_PROMPT = """You are a helpful assistant to integrate a piece of news information into a Wikipedia article. You should read the original paragraph, find where and how to insert the news information, and return to me a new paragraph with the news information integrated. You should do the following when integrating the news information:
1. Only integrate objective news information instead of subjective opinions and commentaries.
2. Make less change as possible to the original paragraph.
3. Make sure the new paragraph is coherent and grammatically correct."""

USER_PROMPT = """**Original Paragraph**
<original>

**News Information**
<news>

**Updated Paragraph**
"""


class EditorLLM:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=int(os.getenv("WORLD_SIZE", 1)),
            gpu_memory_utilization=0.80,
            max_model_len=5096,
        )
        self.sampling_params = SamplingParams(
            max_tokens=5096, 
            temperature=0.001,
        )
        
    def query_editor_model(self, news_information, original_content):
        # use vllm model to generate response        
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT.replace("<original>", original_content).replace("<news>", news_information)
        input_messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        result = self.model.chat(input_messages, sampling_params=self.sampling_params)
        response = result[0].outputs[0].text.strip()
        return response

def edit_wiki_sections(args, edit_data):
    edits = edit_data['aggregated_output']
    sorted_edits = sorted(edits, key=lambda x: datetime.strptime(x['date'], '%b %d, %Y'))
    edited_page_data = copy.deepcopy(edit_data)
    edited_page_data['agent_edits'] = {}
    if args.use_trained_model:
        editorLLM = EditorLLM(model_path=args.trained_model_path)
    for idx, edit in enumerate(sorted_edits):
        section_name = edit['section']
        news = edit['text']
        if section_name not in edit_data['meta_data']['section_content']:
            print("Not Present: ", section_name)
            continue
        section_conent = edit_data['meta_data']['section_content'][section_name]
        if idx == 0:
            # original content should be grouped with section name
            orginal_section_content = section_conent
        
        if args.use_trained_model:
            # Use the user trained model
            output = editorLLM.query_editor_model(news, section_conent)
            criteria_cost = 0 # since we are using the user trained model
        else:
            inp_prompt = EDIT_PROMPT.format(original=section_conent, news=news)
            llm = ChatOpenAI(model=args.model, max_tokens=16000, temperature=0)
            messages = [{"role": "user", "content": inp_prompt}]
            with get_openai_callback() as cb:
                response = llm.invoke(messages)
            output = response.content
            criteria_cost = float(cb.total_cost)

        original_s, updated_s, explanation = get_text_diff(section_conent, output)
        # update the section content with the new content

        edit_data['meta_data']['section_content'][section_name] = output

        if section_name not in edited_page_data['agent_edits']:
            edited_page_data['agent_edits'][section_name] = []

        edited_page_data['agent_edits'][section_name].append(
                {"original": original_s, 
                 "updated": updated_s, 
                 "explanation": explanation, 
                 'original_section': orginal_section_content,
                 "edited_output": output},
            )
        
    return edited_page_data



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Model to use for navigation",
        default="gpt-4o",
        type=str
    )
    parser.add_argument(
        "--trained_model_path",
        help="Path to the trained model",
        default="gangiswag/llama3.1-8b-lora-wiki1000-0311-integrated",
        type=str
    )
    parser.add_argument(
        "--news_data_file",
        help="File containing the news data",
        required=True,
        type=str
    )
    parser.add_argument(
        "--out_dir",
        help="Output directory to store the edited page content",
        required=False,
        default="output",
        type=str
    )
    parser.add_argument(
        "--use_trained_model",
        help="Uses the editor model trained on the user data",
        action="store_true"
    )
    args = parser.parse_args()

    # Read the news data
    try:
        edit_data = read_json(args.news_data_file)
    except FileNotFoundError:
        print(f"Error: The file {args.news_data_file} does not exist.")
        exit(1)
    # Edit the Wikipedia data with the news data
    edited_data = edit_wiki_sections(args, edit_data)
    # Write the edited data to a file
    file_name = os.path.join(args.out_dir, "edited_" + os.path.basename(args.news_data_file).split('updates_')[1])
    print("Writing to file: ", file_name)
    with open(file_name, 'w') as f:
        json.dump(edited_data, f, indent=4)
