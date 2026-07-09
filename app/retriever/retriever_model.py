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

class multiQueries(BaseModel):
    query: List[str] = Field(description="5 different diverse retrieval queries.")

structured_model_multiQuery = model.with_structured_output(multiQueries)

prompt_multi_query = ChatPromptTemplate.from_template(
    """You generate retrieval queries for a RAG system.

    Given a question, generate 5 diverse search queries that maximize recall.

    Rules:
    - Each query must retrieve different evidence, not be a paraphrase.
    - Cover different relevant aspects when applicable (definition, events, causes, effects, actions, relationships, locations, attributes, evidence, summary).
    - Use terms likely to appear in the document.
    - the queries should be relevant to main query but not too similar.
    - Keep each query under 30 tokens.

    Question:
    {question}
    """
)


multi_query_chain = prompt_multi_query | structured_model_multiQuery

# -----------------------
# decomposition
# -----------------------

class decomposeQuery(BaseModel):
    query: List[str] = Field(description="5 different diverse retrieval queries.")

structured_model_decompose = model.with_structured_output(decomposeQuery)

prompt_decompose = ChatPromptTemplate.from_template(
    """
    You are an expert at decomposing complex questions.
    Break the user's question into the minimum number of independent subquestions needed to answer it completely. 
    try to break it in minimum 3 and maximum 5 subquestions.

    Rules:
    - Each subquestion should retrieve different evidence. 
    - Preserve the original meaning.  
    - Avoid redundant subquestions.
    - Keep each sub-query under 30 tokens.

    Question: {question}
    """
)

decompose_chain = prompt_decompose | structured_model_decompose