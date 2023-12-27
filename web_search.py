import logging
import re
from typing import List, Optional

# Importing necessary modules and classes from various libraries.
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.llms import LlamaCpp
from langchain.llms.base import BaseLLM
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.retrievers import WebResearchRetriever
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger(__name__)

# Setting up a logger for logging information and debugging.

class SearchQueries(BaseModel):
    """Search queries to research for the user's goal."""
    queries: List[str] = Field(
        ..., description="List of search queries to look up on Google"
    )
    # A Pydantic model to represent search queries.

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question", "specialty"],
    template="""<<SYS>> \n You are an assistant tasked with improving Google search \
results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries for {specialty} that \
are similar to this question. The output should be a numbered list of questions \
and each should have a question mark at the end: \n\n {question} [/INST]""",
)
# A prompt template for LlamaCpp model, instructing it to generate Google search queries.

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question", "specialty"],
    template="""You are an assistant tasked with improving Google search \
results. Generate THREE Google search queries for {specialty} that are similar to \
this question. The output should be a numbered list of questions and each \
should have a question mark at the end: {question}""",
)
# A default prompt template for generating Google search queries.

def process_personalization_in_prompt(prompt, query):
    text = prompt.template
    parts = query.split('$$$')
    if len(parts) > 1:
        pers_str = parts[1]
        text = text.replace('{specialty}', pers_str)
        prompt.template = text
    return prompt
# A function to process personalization in the prompt. It replaces the '{specialty}' placeholder in the prompt template with the user's input.

class LineList(BaseModel):
    """List of questions."""
    lines: List[str] = Field(description="Questions")
    # A Pydantic model to represent a list of questions.

class QuestionListOutputParser(PydanticOutputParser):
    """Output parser for a list of numbered questions."""
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = re.findall(r"\d+\..*?(?:\n|$)", text)
        return LineList(lines=lines)
# A custom output parser class that extends PydanticOutputParser. It parses the output text to extract a list of numbered questions.

class PersonalizedWebResearch(WebResearchRetriever):
    """`Google Search API` retriever."""
    # This class extends WebResearchRetriever to perform personalized web research using Google Search API.

    # Class attributes
    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )
    # VectorStore instance for storing and retrieving web page content.

    llm_chain: LLMChain
    # LLMChain instance for processing queries using a language model.

    search: GoogleSearchAPIWrapper = Field(..., description="Google Search API Wrapper")
    # GoogleSearchAPIWrapper instance for performing Google searches.

    num_search_results: int = Field(1, description="Number of pages per Google search")
    # Specifies the number of search results to retrieve per Google search.

    text_splitter: RecursiveCharacterTextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )
    # Text splitter for dividing web pages into manageable chunks.

    url_database: List[str] = Field(
        default_factory=list, description="List of processed URLs"
    )
    # A list to keep track of URLs that have already been processed.

    personalization_profile = {}
    # A dictionary to store personalization profiles.

    @classmethod
    def set_personalization_profile(cls, profile):
        cls.personalization_profile = profile
    # Class method to set the personalization profile.

    @classmethod
    def from_llm(
        cls,
        vectorstore: VectorStore,
        llm: BaseLLM,
        search: GoogleSearchAPIWrapper,
        prompt: Optional[BasePromptTemplate] = None,
        num_search_results: int = 1,
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        ),
    ) -> "WebResearchRetriever":
        """
        Initialize from llm using default template.

        Args:
            vectorstore: Vector store for storing web pages
            llm: llm for search question generation
            search: GoogleSearchAPIWrapper
            prompt: prompt to generating search questions
            num_search_results: Number of pages per Google search
            text_splitter: Text splitter for splitting web pages into chunks

        Returns:
            WebResearchRetriever
        """
        # Factory method to create an instance of the retriever from a language model.

        if not prompt:
            # If no prompt is provided, use the default search prompt.
            prompt = DEFAULT_SEARCH_PROMPT

        # Use chat model prompt
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=QuestionListOutputParser(),
        )
        # Setting up the LLMChain with the provided language model, prompt, and output parser.

        return cls(
            vectorstore=vectorstore,
            llm_chain=llm_chain,
            search=search,
            num_search_results=num_search_results,
            text_splitter=text_splitter,
        )
        # Returning an instance of the class with the specified configuration.

    def clean_search_query(self, query: str) -> str:
        """
        Clean the search query to ensure compatibility with search engines.

        Args:
            query: The search query to be cleaned.

        Returns:
            The cleaned search query.
        """
        # Method to clean and format the search query.

        if query[0].isdigit():
            # If the query starts with a digit, find the first quote and extract the query part after it.
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                query = query[first_quote_pos + 1 :]
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()
        # Return the cleaned query.

    def search_tool(self, query: str, num_search_results: int = 1) -> List[dict]:
        """
        Perform a Google search and return the results.

        Args:
            query: The search query.
            num_search_results: The number of search results to return.

        Returns:
            A list of search results.
        """
        # Method to perform a Google search using the provided query.

        query_clean = self.clean_search_query(query)
        result = self.search.results(query_clean, num_search_results)
        return result
        # Return the search results.

      
