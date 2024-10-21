import os
import requests
from bs4 import BeautifulSoup
import wikipedia
from serpapi import GoogleSearch  
from newsapi import NewsApiClient
from datetime import datetime
import logging
import re
import time
import subprocess
import random
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForCausalLM


logging.basicConfig(level=logging.DEBUG)  
logger = logging.getLogger("Info_Retriever")


OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')  


if NEWSAPI_KEY:
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    logger.info("NewsAPI client initialized.")
else:
    logger.error("NewsAPI key not found. Please set the NEWSAPI_KEY environment variable.")

class InfoRetriever:
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
        self.tokenizer = tokenizer
        self.model = model
        self.summary_cache = {}
        logger.debug("InfoRetriever initialized with tokenizer and model.")

    def search_wikipedia(self, query: str) -> str:
        try:
            logger.info(f"Searching Wikipedia for: {query}")
            summary = wikipedia.summary(query, sentences=3)
            logger.debug(f"Wikipedia summary: {summary}")
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"Disambiguation error for query '{query}': {e}")
            return f"Your query '{query}' resulted in multiple topics. Please be more specific."
        except wikipedia.exceptions.PageError:
            logger.warning(f"No Wikipedia page found for '{query}'.")
            return f"No Wikipedia page found for '{query}'."
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}", exc_info=True)
            return "An error occurred while searching Wikipedia."

    def get_weather(self, city: str) -> str:
        try:
            logger.info(f"Fetching weather for: {city}")
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
            response = requests.get(url)
            data = response.json()
            logger.debug(f"Weather API response: {data}")
            if data.get('cod') != 200:
                logger.warning(f"Weather data not found for '{city}': {data.get('message')}")
                return f"Weather data not found for '{city}'."
            weather = data['weather'][0]['description']
            temp = data['main']['temp']
            logger.info(f"Weather in {city}: {weather}, {temp}°C")
            return f"The current weather in {city} is {weather} with a temperature of {temp}°C."
        except Exception as e:
            logger.error(f"Weather API error: {e}", exc_info=True)
            return "An error occurred while fetching the weather information."

    def get_news(self, topic: str) -> str:
        try:
            logger.info(f"Fetching news for: {topic}")
            all_articles = newsapi.get_everything(q=topic,
                                                  language='en',
                                                  sort_by='publishedAt',
                                                  page_size=3)
            articles = all_articles.get('articles')
            if not articles:
                logger.warning(f"No news articles found for '{topic}'.")
                return f"No news articles found for '{topic}'."
            news_summary = f"Here are the latest news articles about {topic}:\n"
            for article in articles:
                news_summary += f"- {article['title']} ({article['source']['name']})\n"
            logger.debug(f"News summary: {news_summary}")
            return news_summary
        except Exception as e:
            logger.error(f"NewsAPI error: {e}", exc_info=True)
            return "An error occurred while fetching news information."

    def perform_serpapi_search(self, query: str, max_results: int = 5) -> str:
        try:
            logger.info(f"Performing SerpAPI search for: {query}")
            params = {
                "engine": "google",
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "num": max_results
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            if "error" in results:
                logger.error(f"SerpAPI error: {results['error']}")
                return "An error occurred while performing the search."
            organic_results = results.get("organic_results", [])
            if not organic_results:
                logger.warning(f"No search results found for '{query}'.")
                return f"No search results found for '{query}'."
            search_summary = f"Here are the top search results for '{query}':\n"
            for result in organic_results:
                title = result.get('title')
                link = result.get('link')
                snippet = result.get('snippet')
                search_summary += f"- {title}: {snippet} ({link})\n"
            logger.debug(f"SerpAPI search summary: {search_summary}")

            summarized_info = self.summarize_text_local(search_summary)
            return summarized_info
        except Exception as e:
            logger.error(f"SerpAPI search error: {e}", exc_info=True)
            return "An error occurred while performing the search."


    def scrape_wikipedia_page(self, query: str) -> str:
        try:
            logger.info(f"Scraping Wikipedia page for: {query}")
            page = wikipedia.page(query)
            content = page.content
            logger.debug(f"Scraped Wikipedia content: {content[:500]}...")  
            return content
        except Exception as e:
            logger.error(f"Error scraping Wikipedia page: {e}", exc_info=True)
            return "An error occurred while scraping the Wikipedia page."

    def summarize_text_local(self, text: str, max_length: int = 150, timeout: int = 30) -> str:
        """
        Summarize text using the local Vicuna model with a timeout.
        """
        def generate_summary():
            logger.debug("Starting summary generation.")
            prompt = f"Summarize the following information professionally:\n\n{text}"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            logger.debug("Tokenized prompt successfully.")

            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id
            )
            logger.debug("Generated summary IDs successfully.")

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            logger.debug(f"Local summarization result: {summary}")

            self.summary_cache[text] = summary  
            logger.debug("Summary cached successfully.")
            return summary

        try:
            logger.info("Summarizing text using the local Vicuna model.")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(generate_summary)
                summary = future.result(timeout=timeout)
                logger.debug(f"Local summarization result: {summary}")
                logger.info("Summarization completed successfully.")
                return summary
        except concurrent.futures.TimeoutError:
            logger.error("Summarization timed out.")
            return "An error occurred while summarizing the information due to a timeout."
        except Exception as e:
            logger.error(f"Local summarization error: {e}", exc_info=True)
            return "An error occurred while summarizing the information locally."

    def list_files(self, directory: str) -> str:
        try:
            logger.info(f"Listing files in directory: {directory}")
            files = os.listdir(directory)
            file_list = "\n".join(files)
            logger.debug(f"Files in '{directory}':\n{file_list}")
            return f"Files in '{directory}':\n{file_list}"
        except Exception as e:
            logger.error(f"Error listing files in directory '{directory}': {e}", exc_info=True)
            return f"An error occurred while listing files in '{directory}'."

    def open_application(self, app_path: str) -> str:
        try:
            logger.info(f"Opening application: {app_path}")
            subprocess.Popen([app_path])
            return f"Opened application at {app_path}."
        except Exception as e:
            logger.error(f"Error opening application '{app_path}': {e}", exc_info=True)
            return f"An error occurred while opening '{app_path}'."

    def retrieve_information(self, query: str) -> str:
        """
        Determine the type of query and fetch information from appropriate sources.
        Prioritize sources based on query intent.
        """
        logger.info(f"Retrieving information for query: {query}")

        if 'capital of' in query.lower():
            try:

                match = re.search(r'capital of\s+([a-zA-Z\s]+)', query.lower())
                if match:
                    country = match.group(1).strip().title()
                    logger.debug(f"Extracted country for capital retrieval: {country}")

                    summary = self.search_wikipedia(f"Capital of {country}")
                    if "may refer to" in summary:

                        summary = self.perform_serpapi_search(query) 
                    return summary
                else:
                    logger.warning("Failed to extract country from 'capital of' query.")
                    return "Please specify the country for which you want to know the capital."
            except Exception as e:
                logger.error(f"Error retrieving capital: {e}", exc_info=True)
                return "An error occurred while retrieving the capital information."
        elif 'weather' in query.lower():

            try:

                match = re.search(r'weather in\s+([a-zA-Z\s]+)', query.lower())
                if match:
                    city = match.group(1).strip().title()
                else:
                    city = 'New York'  
                return self.get_weather(city)
            except Exception as e:
                logger.error(f"Error extracting city from weather query: {e}", exc_info=True)
                return "Please specify the city for which you want the weather information."
        elif 'news' in query.lower():

            try:
                match = re.search(r'news about\s+([a-zA-Z\s]+)', query.lower())
                if match:
                    topic = match.group(1).strip().title()
                else:
                    topic = 'technology'
                return self.get_news(topic)
            except Exception as e:
                logger.error(f"Error extracting news topic from query: {e}", exc_info=True)
                return "Please specify the topic for which you want the latest news."
        elif any(keyword in query.lower() for keyword in ['who is', 'what is', 'tell me about']):
            try:
                if 'who is' in query.lower():
                    parts = query.lower().split('who is')
                    subject = parts[-1].strip().capitalize()
                elif 'what is' in query.lower():
                    parts = query.lower().split('what is')
                    subject = parts[-1].strip().capitalize()
                else:
                    parts = query.lower().split('tell me about')
                    subject = parts[-1].strip().capitalize()
                logger.debug(f"Extracted subject for information retrieval: {subject}")
                summary = self.search_wikipedia(subject)
                if "may refer to" in summary:

                    summary = self.perform_serpapi_search(query)  
                return summary
            except Exception as e:
                logger.error(f"Error extracting subject from query: {e}", exc_info=True)
                return "Please specify the subject you want information about."
        elif any(keyword in query.lower() for keyword in ['if i am', 'when was i born', 'how old am i']):

            try:

                match = re.search(r'if i am\s+(\d+)', query.lower())
                if match:
                    age = int(match.group(1))
                    current_year = datetime.now().year
                    birth_year = current_year - age
                    logger.info(f"Calculated birth year: {birth_year} for age: {age}")
                    return f"If you are {age} years old, you were born in {birth_year}."
                else:
                    logger.warning("Failed to extract age from query.")
                    return "Please specify your age to calculate your birth year."
            except Exception as e:
                logger.error(f"Error processing age-related query: {e}", exc_info=True)
                return "An error occurred while processing your query."
        elif any(keyword in query.lower() for keyword in ['list files in', 'open application']):

            try:
                if 'list files in' in query.lower():

                    match = re.search(r'list files in\s+([a-zA-Z0-9_\\/:\.\s]+)', query.lower())
                    if match:
                        directory = match.group(1).strip()
                        return self.list_files(directory)
                    else:
                        logger.warning("Failed to extract directory from 'list files in' query.")
                        return "Please specify the directory you want to list files from."
                elif 'open application' in query.lower():

                    match = re.search(r'open application\s+([a-zA-Z0-9_]+)', query.lower())
                    if match:
                        app_name = match.group(1).strip().lower()

                        app_paths = {
                            "notepad": "C:\\Windows\\System32\\notepad.exe",
                            "calculator": "C:\\Windows\\System32\\calc.exe",

                        }
                        app_path = app_paths.get(app_name)
                        if app_path:
                            return self.open_application(app_path)
                        else:
                            logger.warning(f"Application '{app_name}' not recognized.")
                            return f"Application '{app_name}' not recognized."
                    else:
                        logger.warning("Failed to extract application name from 'open application' query.")
                        return "Please specify the application you want to open."
                else:
                    logger.warning(f"Unhandled system operation in query: {query}")
                    return "I'm sorry, I can't perform that operation."
            except Exception as e:
                logger.error(f"Error handling system operation: {e}", exc_info=True)
                return "An error occurred while performing the requested operation."
        elif 'read screen' in query.lower():

            logger.warning("The 'read screen' feature is currently disabled.")
            return "I'm sorry, the 'read screen' feature is currently unavailable."
        else:

            if not SERPAPI_API_KEY:
                logger.error("SerpAPI API key not set.")
                return "Search functionality is currently unavailable. Please try again later."

            logger.debug("Performing general SerpAPI search.")
            search_result = self.perform_serpapi_search(query)
            if any(error_phrase in search_result.lower() for error_phrase in ["an error occurred", "please specify", "no search results found"]):

                logger.info("Falling back to Wikipedia due to search error.")
                wiki_summary = self.search_wikipedia(query)
                if "may refer to" in wiki_summary:

                    wiki_summary = self.scrape_wikipedia_page(query)

                summarized_info = self.summarize_text_local(wiki_summary)
                if "An error occurred while summarizing" in summarized_info:

                    return wiki_summary
                else:

                    return summarized_info
            else:
                summarized_info = self.summarize_text_local(search_result)
                return summarized_info