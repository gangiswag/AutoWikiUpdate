import json
import re
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from autogpt import AutoGPT
from langchain_openai import ChatOpenAI
from langchain_community.docstore import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain_experimental.autonomous_agents.autogpt.output_parser import preprocess_json_input
from newspaper import Article 
import faiss
from typing import List, Dict
import tiktoken
import argparse
from copy import deepcopy
import time
from datetime import datetime, timedelta
import os
from utils import scrape_url

class BFSAgent:
    def __init__(self, navigator_model="gpt-4o-mini", 
                 aggregator_model="gpt-4o-mini", 
                 extractor_model="gpt-4o-mini", 
                 num_iterations=10,
                 start_time=None, 
                 time_delta=7,
                 use_web=False,
                 no_criteria=False):

        self.NAVIGATOR_MODEL=navigator_model
        self.AGGREGATOR_MODEL=aggregator_model
        self.EXTRACTOR_MODEL=extractor_model
        self.use_web = use_web
        if use_web:
            self.search_tool = GoogleSerperAPIWrapper()
        else:
            self.search_tool = GoogleSerperAPIWrapper(type="news")

        self.no_criteria = no_criteria

        self.enc = tiktoken.get_encoding("cl100k_base")
        

        self.aggregator_messages = list()

        self.num_iterations = num_iterations

        self.aggregated_data = list()
        self.counter = 1
        self.aggregator_cost = 0.0
        self.extractor_cost = 0.0
        self.extractor_time = 0.0
        self.aggregator_time = 0.0
        self.search_time = 0.0
        self.parse_time = 0.0
        # MM-DD-YYYY
        self.start_date = datetime.strptime(start_time, "%m-%d-%Y")
        self.end_date = self.start_date + timedelta(days=time_delta)

        self.current_query = None
        self.last_search_thought = "Not Available"
        self.last_selection_thought = "Not Available"
        self.previous_iterations = list()
        self.trace_log = list()

        self.urls_visited = set() # Set to keep track of visited URLs
        self.article_cache = {}  # New cache for scraped articles
        self.init_nav_agent()

    def setup(self, user_task, wiki):
        self.user_task = user_task
        self.wiki = wiki

    def init_nav_agent(self):

        embeddings_model = OpenAIEmbeddings() 
        tools = [
                    Tool(
                        name = "search",
                        func=self.search,
                        description="Useful for when you need to gather information from the web. You should ask targeted questions. You need not include the date in the search query, the tool automatically narrows results to the dates under consideration."
                    ),
                    Tool(
                        name = "extract",
                        func=self.extract,
                        description="Useful to extract relevant information from provided URL and get feedback from Aggregator Module on how to proceed next"
                    )
                ]
        
        # Initialize the vectorstore as empty
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

        agent_llm = ChatOpenAI(temperature=0.7, model_name=self.NAVIGATOR_MODEL)

        self.agent = AutoGPT.from_llm_and_tools(
                    ai_name="BFSAgent",
                    ai_role="Assistant",
                    tools=tools,
                    llm=agent_llm,
                    memory=vectorstore.as_retriever(),
                    max_iterations=3*self.num_iterations
                )
        # Set verbose to be true
        self.agent.chain.verbose = False

    def format_section_criteria(self):
        formatted_criteria = ""
        for item in self.wiki["criteria"]:
            if item["name"] in self.wiki["section_content"]:
                formatted_criteria += "Section: " + item["name"]+"\n"
                formatted_criteria += "Criteria: "
                for criteria in item["criteria"]:
                    formatted_criteria += "- " + criteria + "\n"
                formatted_criteria += "\n\n"
        return formatted_criteria.strip()
    
    def format_section_no_criteria(self):
        formatted_criteria = ""
        for item in self.wiki["criteria"]:
            if item["name"] in self.wiki["section_content"]:
                formatted_criteria += "Section: " + item["name"]+"\n"
                formatted_criteria += "\n\n"
        return formatted_criteria.strip()


    def extract_info_prompt(self, task, data, article_data):
        extract_start = time.time()
        encoded = self.enc.encode(data)
        # Need to truncate input to OpenAI call as GPT 3.5 turbo doesn't accept more than 16k tokens for input + output
        if len(encoded) > 100000:
            data = self.enc.decode(encoded[:100000])

        prompt = """WIKIPEDIA PAGE NAME: {query}
        
        NEWS ARTICLE DATE: {article_date}\n\n
        NEWS ARTICLE CONTENT: {data}\n\n\n
        The above news article was chosen with the following motivation by the web navigator when searching the web: {search_motivation}
        
        You must follow the navigator's current motivation to identify the relevant updates from the provided content. You must return the relevant information in the form of a summary of what the update is for the wikipedia page, and it should NOT be longer than 6 sentences.\n  

        Your goal is to be an EXTRACTOR that identifies any relevant updates from the news article content, that are relevant for the Wikipedia page. You are provided below the corresponding Wikipedia page's individual sections. ONLY identify relevant updates that are WORTHY of being added to or used for updating any one of the provided sections in the wikipedia page. The corresponding criteria for updating content within each of the sections is also provided for more context. The updates should only correspond to anything that happended between {start_date} and {end_date} since this is the duration for which you are trying to update the Wikipedia page. You MUST pick only one section of the wikipedia page that the content would be most relevant for.

        SECTIONS AND CRITERIA:
        {section_criteria} 
        
        You MUST NOT draw any inferences on your own based on the news article content or directly try to answer the navigator's motivation. You can choose to merge or combine these updates as needed. 
        
        You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads.
        Response Format: 
        {{ 
            "thoughts": "Concise justification for why the update was deemed relevant and why it is worthy of being added to the corresponding Wikipedia section.",
            "section": "The wikipedia section for which the updates in the news article are most relevant to. You MUST pick only one section.",
            "update": "Summary of relevant updates needed to be made to the section of the wikipedia page",
            
        }}"""

        no_criteria_prompt = """WIKIPEDIA PAGE NAME: {query}
        
        NEWS ARTICLE DATE: {article_date}\n\n
        NEWS ARTICLE CONTENT: {data}\n\n\n
        The above news article was chosen with the following motivation by the web navigator when searching the web: {search_motivation}
        
        You must follow the navigator's current motivation to identify the relevant updates from the provided content. You must return the relevant information in the form of a summary of what the update is for the wikipedia page, and it should NOT be longer than 6 sentences.\n  

        Your goal is to be an EXTRACTOR that identifies any relevant updates from the news article content, that are relevant for the Wikipedia page. You are provided below the corresponding Wikipedia page's individual sections. ONLY identify relevant updates that are WORTHY of being added to or used for updating any one of the provided sections in the wikipedia page. The updates should only correspond to anything that happended between {start_date} and {end_date} since this is the duration for which you are trying to update the Wikipedia page. You MUST pick only one section of the wikipedia page that the content would be most relevant for.

        SECTIONS:
        {section_criteria} 
        
        You MUST NOT draw any inferences on your own based on the news article content or directly try to answer the navigator's motivation. You can choose to merge or combine these updates as needed. 
        
        You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads.
        Response Format: 
        {{ 
            "thoughts": "Concise justification for why the update was deemed relevant and why it is worthy of being added to the corresponding Wikipedia section.",
            "section": "The wikipedia section for which the updates in the news article are most relevant to. You MUST pick only one section.",
            "update": "Summary of relevant updates needed to be made to the section of the wikipedia page",
            
        }}"""


        if self.no_criteria:
            inp_prompt = no_criteria_prompt.format(article_date=article_data["date"], query=task, data=data, search_motivation=self.last_search_thought, start_date=self.start_date.strftime("%b %d, %Y"), end_date=self.end_date.strftime("%b %d, %Y"), section_criteria=self.format_section_no_criteria())
        else:
            inp_prompt = prompt.format(article_date=article_data["date"], query=task, data=data, search_motivation=self.last_search_thought, start_date=self.start_date.strftime("%b %d, %Y"), end_date=self.end_date.strftime("%b %d, %Y"), section_criteria=self.format_section_criteria())
        messages = [
                        {"role": "system", "content": """You are an assistant helping to aggregate relevant news updates about a Wikipedia page under consideration. You are working with a web nagivator that iteratively searches for updates about the Wikipedia page. Your goal is identify and extract any relevant updates from the news article content that the web navigator has provided to you in the current iteration. ONLY incorporate information that is WORTHY of being mentioned in a section of the Wikipedia page. DO NOT RESPOND with your own knowledge, only respond BASED on information provided in the text."""},
                        {"role": "user", "content": inp_prompt}
                    ]
        llm = ChatOpenAI(model=self.EXTRACTOR_MODEL, max_tokens=2000, temperature=0)
        structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)
        with get_openai_callback() as cb:
            response = structured_llm.invoke(messages)  
        self.extractor_cost += float(cb.total_cost)
        self.extractor_time += time.time() - extract_start

        if not response["parsing_error"]:
            data = response["parsed"]
            self.trace_log.append({
                    "extractor_thoughts": data["thoughts"],
                    "time_step": self.counter
                })
            return data, False
        else:
            return {}, True

            
    def article_scrape(self, url):
        if url in self.article_cache:
            return self.article_cache[url]
        
        if self.use_web:
            content, article_date = scrape_url(url)            
            article_data = {
                "title": url,
                "url": url,
                "date": article_date
            }
            print(article_data)

        else:        
            cc_article = Article(url)
            cc_article.download()
            cc_article.parse()
            content = cc_article.text
            article_data = {
                "title": cc_article.title,
                "url": url,
                "date": cc_article.publish_date.strftime("%b %d, %Y")
            }

        self.article_cache[url] = (content, article_data)
        return content, article_data

    # async def async_article_scrape(self, url):
    #     loop = asyncio.get_running_loop()
    #     return await loop.run_in_executor(None, self.article_scrape, url)


    def search(self, query:str) -> List[Dict]:
        assistant_reply = self.agent.chat_history_memory.messages[-1].content
        try:
            parsed = json.loads(assistant_reply, strict=False)
            self.last_search_thought = parsed["thoughts"]["text"] + " " + parsed["thoughts"]["reasoning"]
            self.trace_log.append({
                "navigator_thoughts": parsed["thoughts"]["text"],
                "navigator_reasoning": parsed["thoughts"]["reasoning"],
                "navigator_action": "Search: " + query,
                "time_step": self.counter
            })
        except json.JSONDecodeError:
            preprocessed_text = preprocess_json_input(assistant_reply)
            try:
                parsed = json.loads(preprocessed_text, strict=False)   
                self.last_search_thought = parsed["thoughts"]["text"] + " " + parsed["thoughts"]["reasoning"]     
                self.trace_log.append({
                    "navigator_thoughts": parsed["thoughts"]["text"],
                    "navigator_reasoning": parsed["thoughts"]["reasoning"],
                    "navigator_action": "Search: " + query,
                    "time_step": self.counter
                })        
            except Exception:
                print("Error parsing search thought")

        search_start = time.time()
        """Useful for when you need to gather information from the web. You should ask targeted questions"""
        
        results = self.search_tool.results(query + " after:" + self.start_date.strftime('%Y-%m-%d') + " before:" + self.end_date.strftime('%Y-%m-%d'))
        snippets = list()

        if self.use_web:
            for r in results["organic"][:10]:
                link = r["link"]
                if link in self.urls_visited:
                    continue

                # skip any links from wikipedia domain
                if "wikipedia.org" in link:
                    continue

                snippets.append({
                    "title": r["title"],
                    "url": r["link"],
                    "snippet": r["snippet"]
                })
        else:
            for r in results["news"][:10]:
                link = r["link"]
                if link in self.urls_visited:
                    continue            
                # check if the publish date is in the range of start and end date
                try:
                    _ , article_data = self.article_scrape(link)
                except Exception as e:
                    print("Error: ", str(e))
                    continue
                article_date = datetime.strptime(article_data["date"], "%b %d, %Y")
                if self.start_date < article_date and article_date < self.end_date:
                    snippets.append({
                    "title": r["title"],
                    "url": r["link"],
                    "snippet": r["snippet"],
                    "date": r["date"],
                    "source": r["source"]
                })
        self.search_time += time.time() - search_start
        self.current_query = query
        return snippets

    def aggregator(self, data, article):
        aggregator_start = time.time()
        if len(self.aggregated_data) > 0:
            aggregated_list = ""
            for pid, agg_item in enumerate(self.aggregated_data):
                aggregated_list += "ID: E" + str(pid+1) + " Text: " + agg_item["text"] + "\n\n"
        else:
            aggregated_list = "None"

        previous_iterations = ""
        for c, iter in enumerate(self.previous_iterations):
            previous_iterations += """Iteration {i}: Thoughts - {t}\nFeedback - {f}\n\n""".format(i=str(c+1), t=iter["thoughts"], f=iter["feedback"])

        system_prompt = """You are an information aggregation assistant designed to aggregate relevant updates about a Wikipedia page under consideration. The downstream task you are helping on is to automatically identify the necessary updates that need to be made for a given Wikipedia page, based on recent news articles. Your goal at every iteration is to identify whether the relevant updates identified in the news are WORTHY of being added to or used to update the given section content of the Wikipedia page. Make sure to not gather information that is duplicate, i.e. do not add redudant information into what you have already aggregated. You should stop aggregating once you believe the necessary updates have been gathered. REMEMBER that your goal is ONLY to aggregate relevant information."""
        # system_input = system_prompt.format(num_to_aggregate=str(self.num_to_aggregate))
        system_input = system_prompt

        prompt = """You will be provided with updates collected from a news article by a navigator assistant. You need to decide whether the provided updates should be added into the aggregated information list. Note that all the news updates collected in the aggregated list will later be used to update the Wikipedia page. Hence, you should ONLY choose to incorporate the updates into the aggregated list if the update is WORTHY of being added to or revising the relevant section content of the Wikipedia page.
        
        Further, you should provide feedback to the navigator assistant on what specific information to look for next. If the information needs to be gathered in multiple steps, you can break it down to multiple steps and sequentially instruct the navigator. The navigator assistant cannot see the information aggregated so far, so you make sure the feedback is clear and does not have any ambiguity. Make sure to refer to any entities by their full names when providing the feedback so that the navigator knows what you are referring to. You should instruct the navigator to terminate if you believe the necessary updates have been gathered.

        You have a maximum of {num_iterations} iterations overall, after which the information aggregated will be automatically returned. You SHOULD instruct the navigator to TERMINATE after all iterations have been completed. You are also provided with your thoughts and navigator feedback you provided in the previous iterations, so do not repeat yourself or provide the same feedback again. In addition, the navigator's thoughts are also shown to you to give additional context for why the provided information was extracted.
        
        Current Iteration Counter: {counter}

        Wikipedia Page Name: {user_task}

        Previous Iterations: {previous_iterations}

        Navigator Thoughts: {navigator_motivation}

        Information Aggregated so far: 
        {aggregated_list}

        Provided Updates: 
        {provided_list}  

        Below is the Wikipedia section which has been deemed most relevant to potentially revise based on the provided updates. Review the section content carefully to understand what kind of information is typically present in Wikipedia, and decide whether the newly provided updates are WORTHY of adding to or revising the section.

        Section Name: {section_name}
        Section Content: {section_content}

        You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads
        Response Format: 
        {{
            "thoughts": "Your step-by-step reasoning for what action to perform based on the provided information",
            "action": "The action to perform. Allowed actions are: IGNORE() if the updates are not relevant or worthy of being incorporated to the wikipedia section, REPLACE(existing_id) if updates are worthy of being incorporated to the wikipedia section and existing_id in aggregated information should be replaced instead by the provided information, and ADD() if updates are worthy of being incorporated to the wikipedia section and provided information should be added as another item in the aggregated information list",
            "feedback": "Feedback to return to the navigator assistant on what specific information to look for next. The navigator assistant does not have access to the information aggregated, so be clear in your feedback. Also let the navigator assistant know how many more iterations are left. Also, you SHOULD instruct the navigator to TERMINATE incase iterations have been completed or no more updates need to be gathered."
        }}"""
        inp_prompt = prompt.format(num_iterations=str(self.num_iterations), counter=str(self.counter), user_task=self.user_task, previous_iterations=previous_iterations, navigator_motivation=self.last_search_thought, aggregated_list=aggregated_list, provided_list=data["update"], section_name=data["section"], section_content=self.wiki["section_content"][data["section"]])

        messages = [
                        {"role": "system", "content": system_input},
                        {"role": "user", "content": inp_prompt}
                    ]

        llm = ChatOpenAI(model=self.AGGREGATOR_MODEL, max_tokens=1000, temperature=0.7)
        structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)
        
        with get_openai_callback() as cb:
            response = structured_llm.invoke(messages)
        self.aggregator_cost += float(cb.total_cost)
        self.aggregator_time += time.time() - aggregator_start

        if response["parsing_error"]:
            return "Aggregator failed to parse the website content. Continue with a different website.", response["raw"].content, True
        else:
            output = response["parsed"]
            self.trace_log.append({
                        "aggregator_thoughts": output["thoughts"],
                        "aggregator_feedback": output["feedback"],
                        "time_step": self.counter
                    })
            self.previous_iterations.append({
                "thoughts": output["thoughts"],
                "feedback": output["feedback"]
            })
            # print(output)
            try:
                if "REPLACE" in output["action"]:
                    pattern = r'^REPLACE\(E(\d+)\)$'
                    match = re.match(pattern, output["action"])
                    if match:
                        integer1 = match.groups()
                        add_item = article
                        add_item["time_step"] = self.counter 
                        add_item["text"] = data["update"]
                        add_item["section"] = data["section"]
                        add_item["thoughts"] = output["thoughts"]
                        
                        self.aggregated_data[int(integer1)-1] = deepcopy(add_item)
                    else:
                        print("Invalid Action: ", output["action"])

                if "ADD" in output["action"]:
                    add_item = article
                    add_item["time_step"] = self.counter 
                    add_item["text"] = data["update"]
                    add_item["section"] = data["section"]
                    add_item["thoughts"] = output["thoughts"]
                    
                    self.aggregated_data.append(deepcopy(add_item))
            except Exception as e:
                print(e)
            
            return output["feedback"] + " Passages aggregated so far: " + str(len(self.aggregated_data)), response["parsed"], False
        
    def extract(self, url: str) -> List[str]:
        # TODO: keep track of visited URLS here
        self.urls_visited.add(url)
        assistant_reply = self.agent.chat_history_memory.messages[-1].content
        try:
            parsed = json.loads(assistant_reply, strict=False)
            self.last_selection_thought = parsed["thoughts"]["text"] + " " + parsed["thoughts"]["reasoning"]
            self.trace_log.append({
                "navigator_thoughts": parsed["thoughts"]["text"],
                "navigator_reasoning": parsed["thoughts"]["reasoning"],
                "navigator_action": "Extract: " + url,
                "time_step": self.counter
            })
        except json.JSONDecodeError:
            preprocessed_text = preprocess_json_input(assistant_reply)
            try:
                parsed = json.loads(preprocessed_text, strict=False)
                self.last_selection_thought = parsed["thoughts"]["text"] + " " + parsed["thoughts"]["reasoning"]
                self.trace_log.append({
                    "navigator_thoughts": parsed["thoughts"]["text"],
                    "navigator_reasoning": parsed["thoughts"]["reasoning"],
                    "navigator_action": "Extract: " + url,
                    "time_step": self.counter
                })
            except Exception:
                print("Error parsing selection thought")

        parse_start = time.time()

        content, article_data = self.article_scrape(url)

        try:
            content, article_data = self.article_scrape(url)
            # TODO: reject the article if the published date doesn't fall in the range of start and end
        except Exception as e:
            print(str(e))
            content = ""

        print("URL parsing done")
        self.parse_time += time.time() - parse_start

        if content.strip():
            extracted_data, extractor_parse_error = self.extract_info_prompt(self.user_task, content, article_data)
            # print("Extracted Data: ", extracted_data)
            if not extractor_parse_error and extracted_data["update"].strip():
                aggregator_feedback, aggregator_log, aggregator_parse_error = self.aggregator(extracted_data, article_data)
                self.counter += 1
            else:
                aggregator_feedback = "Did not find any relevant information on the website. Continue collecting information." + " Passages aggregated so far: " + str(len(self.aggregated_data)) 
                aggregator_log = None
                aggregator_parse_error = False
        else:
            aggregator_feedback =  "Did not find any relevant information on the website. Continue collecting information." + " Passages aggregated so far: " + str(len(self.aggregated_data)) 
            aggregator_log = None
            aggregator_parse_error = False
            extractor_parse_error = False
            extracted_data = {}
        
        self.aggregator_messages.append({
            "url": url,
            "extractor_log": str(extracted_data),
            "extractor_parse_error": extractor_parse_error,
            "aggregator_log": aggregator_log,
            "aggregator_parse_error": aggregator_parse_error
        })

        return aggregator_feedback

    def run(self):
        output = dict()

        prompt = """You are an assistant aiding an information aggregation process designed to aggregate relevant updates about a Wikipedia page under consideration. The downstream task you are helping on is to automatically identify the necessary updates that need to be made for a given Wikipedia page, based on recent news articles. You are looking for updates between the dates {start_date} and {end_date}.
        
        You are provided access to the web (using the "search" command) which returns news articles relevant to your search query. Based on the provided websites, you should then choose to vist (using the "extract" command) a news article that is most relevant. Along with the news article URL, you are also provided with a short snippet from the article that can help to decide whether the article is relevant. 

        You should only consider news articles that you think will contain relevant updates that are worthy of incorporating into the Wikipedia page. If the news articles returned do not contain any relevant updates, you can choose to perform a different search. DO NOT visit a news article that you have already have visited before. Note that information cannot be directly aggregated based on the search command. You MUST ALSO visit the news article using the extract command in order to be able to aggregate the relevant updates from it. 
        
        You will work in conjunction with an aggregator assistant (which runs as part of the "extract" command) that keeps track of relevant updates aggregated so far and will give feedback to you on what to look for next. You can decide to stop if aggregator assitant tells you so or if you keep running into a loop. You can simply terminate at the end with a message saying aggregation is done.

        Note that you can only search for one piece of information at a time. If the extract command suggests you to search for multiple pieces of information, you should search for each piece sequentially over different iterations.

        Below is the user query. 
        Query: {task}"""

        self.agent.run([prompt.format(task=self.user_task, start_date=self.start_date.strftime("%b %d, %Y"), end_date=self.end_date.strftime("%b %d, %Y"))])
        output["aggregated_output"] = self.aggregated_data
        navigator_cost = float(self.agent.chain_cost)
        output["cost"] = {
            "total_cost": navigator_cost + self.aggregator_cost + self.extractor_cost,
            "navigator_cost": navigator_cost,
            "aggregator_cost": self.aggregator_cost,
            "extractor_cost": self.extractor_cost
        }
        output["meta_data"] = {
            "start_date": self.start_date.strftime("%b %d, %Y"),
            "end_date": self.end_date.strftime("%b %d, %Y"),
            "page_name": self.user_task,
            'criteria': self.wiki["criteria"],
            "section_content": self.wiki["section_content"],
        }

        output["aggregator_messages"] = self.aggregator_messages
        output["trace_log"] = self.trace_log

        return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--navigator_model",
        help="Model to use for navigation",
        default="gpt-4o-mini",
        type=str
    )
    parser.add_argument(
        "--aggregator_model",
        help="Model to use for aggregation",
        default="gpt-4o-mini",
        type=str
    )
    parser.add_argument(
        "--extractor_model",
        help="Model to use for extraction",
        default="gpt-4o-mini",
        type=str
    )
    parser.add_argument(
        "--num_iterations",
        help="Number of iterations to run the aggregator for",
        default=10,
        type=int
    )
    parser.add_argument(
        "--wiki_file",
        help="Wikipedia criteria and section content file",
        required=True,
        type=str
    )
    parser.add_argument(
        "--out_path",
        help="Output file path",
        required=True,
        type=str
    )
    parser.add_argument(
        "--log_path",
        help="Log file path",
        required=True,
        type=str
    )
    parser.add_argument(
        "--use_web",
        help="Whether to use the entire web or just the news articles",
        action="store_true"
    )
    parser.add_argument(
        "--start_time",
        help="End date for the news articles",
        required=True,
        type=str
    )
    parser.add_argument(
        "--time_delta",
        help="Number of days to look back for news articles",
        default=7,
        type=int
    )
    parser.add_argument(
        "--no_criteria",
        help="Whether to use the criteria or not",
        action="store_true"
    )

    args = parser.parse_args()

    BRIGHTDATA_API_KEY = os.environ.get("BRIGHTDATA_API_KEY", None)
    if not BRIGHTDATA_API_KEY:
        raise ValueError("BRIGHTDATA_API_KEY is not set")

    output = list()
    log = list()    
    wiki_content = json.load(open(args.wiki_file))
    entity = wiki_content["page_name"]
    bfsagent = BFSAgent(
        navigator_model=args.navigator_model, 
        extractor_model=args.extractor_model, 
        aggregator_model=args.aggregator_model, 
        num_iterations=args.num_iterations,
        start_time=args.start_time,
        time_delta=args.time_delta,
        use_web=args.use_web,
        no_criteria=args.no_criteria
    )
    bfsagent.setup(user_task=entity, wiki=wiki_content)
    try:
        time_start = time.time()
        output_item = bfsagent.run()
        total_time = time.time() - time_start
        output_item["time"] = {
            "total_time": total_time,
            "navigator_time": total_time - (bfsagent.search_time + bfsagent.parse_time + bfsagent.extractor_time + bfsagent.aggregator_time),
            "extractor_time": bfsagent.extractor_time,
            "aggregator_time": bfsagent.aggregator_time,
            "search_time": bfsagent.search_time,
            "parse_time": bfsagent.parse_time
        }
        output.append(deepcopy(output_item))
    except Exception as e:
        log = {
            "error": str(e)
        } 
    
    json.dump(output_item, open(args.out_path, "w"), indent=4)
    # make sure the directory exists
    if not os.path.exists(os.path.dirname(args.log_path)):
        os.makedirs(os.path.dirname(args.log_path))
    json.dump(log, open(args.log_path, "w"), indent=4)