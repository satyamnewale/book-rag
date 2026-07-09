from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

model = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.5,
    max_tokens = 150
)

class Queries(BaseModel):
    query: List[str] = Field(description="5 different diverse retrieval queries.")

structured_model = model.with_structured_output(Queries)

prompt = ChatPromptTemplate.from_template(
    """You are helping RAG system improve document retrieval.

    Given a user's question, generate 5 different diverse retrieval queries that maximize recall.

    Rules:

    - Each query should explore a different aspect of the original question rather than being a simple paraphrase.
    - Focus on retrieving complementary evidence from different parts of the document.
    - Consider different retrieval perspectives such as:
        + definitions or identification
        + chronology or sequence of events
        + causes or effects
        + actions or behaviors
        + relationships or dependencies
        + locations or contexts
        + descriptions or attributes
        + evidence or supporting details
        + summaries of relevant information

    - Use only perspectives that naturally fit user's question.
    - Use terminology likely to appear in document.
    - Do not force categories that are not applicable.
    - If the question asks for a complete explanation, ensure some queries target partial pieces that together reconstruct the full answer.
    - Keep each query concise and self-contained.
    - Avoid producing near-duplicate paraphrases.
    - questions length should be less than 30 tokens.

    Return only the queries as a numbered list.

    Question:
    {question}"""
)


chain = prompt | structured_model 
