import os
import json
import mlflow
import pandas as pd
from typing import List, TypedDict

from langgraph.graph import StateGraph
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# =========================================================
# MLflow Configuration
# =========================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Course_Recommendation_Engine")

# =========================================================
# Gemini Configuration
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# =========================================================
# Dataset Loading
# =========================================================

DATASET_PATH = "assignment2dataset.csv"

courses_df = pd.read_csv(DATASET_PATH)

# =========================================================
# ChromaDB Vector Store Setup
# =========================================================

persist_directory = "chroma_db"


def build_vector_store():
    documents = []

    for _, row in courses_df.iterrows():
        doc = Document(
            page_content=row["description"],
            metadata={
                "course_id": row["course_id"],
                "title": row["title"],
            },
        )
        documents.append(doc)

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return vector_db


vector_db = build_vector_store()

# =========================================================
# Graph State
# =========================================================

class GraphState(TypedDict, total=False):
    user_query: str
    completed_courses: List[str]
    retrieved_courses: list
    final_output: dict

# =========================================================
# Output Schema
# =========================================================

class Recommendation(BaseModel):
    rank: int
    course_id: str
    title: str
    similarity_score: float
    rationale: str


class RecommendationOutput(BaseModel):
    user_query: str
    completed_courses: List[str]
    recommendations: List[Recommendation]
    total_recommendations: int
    model_used: str
    embedding_model: str


parser = PydanticOutputParser(pydantic_object=RecommendationOutput)

# =========================================================
# Retrieval Step
# =========================================================


def retrieve_courses(state: GraphState) -> GraphState:
    with mlflow.start_span("retrieve_courses"):

        docs = vector_db.similarity_search_with_score(
            state["user_query"],
            k=10,
        )

        filtered = []

        for doc, score in docs:
            course_id = doc.metadata["course_id"]

            if course_id not in state["completed_courses"]:
                filtered.append(
                    {
                        "course_id": course_id,
                        "title": doc.metadata["title"],
                        "description": doc.page_content,
                        "similarity_score": round(1 - score, 2),
                    }
                )

        state["retrieved_courses"] = filtered[:5]

        return state

# =========================================================
# Rationale Generation
# =========================================================


def generate_rationale(state: GraphState) -> GraphState:
    with mlflow.start_span("generate_rationale"):

        prompt = PromptTemplate(
            template="""
            You are an AI Learning Advisor.

            User Query:
            {query}

            Completed Courses:
            {completed}

            Recommended Courses:
            {courses}

            Generate recommendation rationale for each course.

            {format_instructions}
            """,
            input_variables=["query", "completed", "courses"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )

        chain = prompt | llm | parser

        result = chain.invoke(
            {
                "query": state["user_query"],
                "completed": ", ".join(state["completed_courses"]),
                "courses": json.dumps(state["retrieved_courses"], indent=2),
            }
        )

        state["final_output"] = result.dict()

        return state

# =========================================================
# LangGraph Workflow
# =========================================================

workflow = StateGraph(GraphState)

workflow.add_node("retrieve_courses", retrieve_courses)
workflow.add_node("generate_rationale", generate_rationale)

workflow.set_entry_point("retrieve_courses")

workflow.add_edge("retrieve_courses", "generate_rationale")

pipeline = workflow.compile()

# =========================================================
# Sample Test Queries
# =========================================================

TEST_QUERIES = [
    {
        "query": "I've completed Python Programming for Data Science and enjoy data visualization.",
        "completed": ["C001"],
    },
    {
        "query": "I know Azure basics and want to manage containers and build CI/CD pipelines.",
        "completed": ["C010"],
    },
]

# =========================================================
# Main Runner
# =========================================================

if __name__ == "__main__":

    with mlflow.start_run(run_name="Course_Recommendation_Run"):

        for item in TEST_QUERIES:

            result = pipeline.invoke(
                {
                    "user_query": item["query"],
                    "completed_courses": item["completed"],
                }
            )

            print("\n" + "=" * 80)
            print("FINAL OUTPUT")
            print("=" * 80)
            print(json.dumps(result["final_output"], indent=2))
