from pydantic import BaseModel, Field
from typing import Literal

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class RetrievalStrategy(BaseModel):
    strategy: Literal[
        "hybrid",
        "multi_query",
        "decomposition"
    ]

    reason : str = Field(description="The reason of strategy in 10-20 words")

model = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.5,
    max_tokens = 150
)

structured_model = model.with_structured_output(RetrievalStrategy)

prompt = ChatPromptTemplate.from_template(
    """
    You are an expert retrieval strategy selector.

    Choose the single best retrieval strategy for answering the user's question.

    Available strategies:

    1. hybrid
    - Use for simple factual questions.
    - Use when one retrieval is likely sufficient.
    - Use for definitions, people, places, events, and direct facts.

    2. multi_query
    - Use when the same concept can be expressed in many ways.
    - Use when synonyms or alternative wording may retrieve additional evidence.
    - Use for descriptive or explanatory questions.

    3. decomposition
    - Use when the question requires multiple independent pieces of evidence.
    - Use when comparison, reasoning, chronology, causality, or multi-hop reasoning is required.

    Question:
    {question}
    """
)

strategy_chain = prompt | structured_model