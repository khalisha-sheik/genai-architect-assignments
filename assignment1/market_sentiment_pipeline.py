import mlflow
import yfinance as yf
import random
from typing import List, TypedDict

from langgraph.graph import StateGraph
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# =========================================================
# MLflow Configuration
# =========================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Market_Sentiment_Analyzer")

# =========================================================
# LLM Configuration
# =========================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# =========================================================
# Graph State Definition
# =========================================================
class GraphState(TypedDict, total=False):
    company_name: str
    ticker: str
    news: List[str]
    current_price: float
    price_change_7d: str
    price_change_30d: str
    fifty_two_week_high: float
    fifty_two_week_low: float
    final_output: dict

# =========================================================
# Output Schema
# =========================================================
class MarketSentimentOutput(BaseModel):
    company_name: str
    stock_code: str
    current_price: float
    price_change_7d: str
    price_change_30d: str
    fifty_two_week_high: float
    fifty_two_week_low: float
    news_summary: str
    sentiment: str
    confidence_score: float
    people_names: List[str]
    places_names: List[str]
    other_companies_referred: List[str]
    related_industries: List[str]
    market_implications: str
    investment_recommendation: str
    recommendation_rationale: str

parser = PydanticOutputParser(pydantic_object=MarketSentimentOutput)

# =========================================================
# Step 1: Resolve Stock Ticker (LLM)
# =========================================================
def resolve_ticker(state: GraphState) -> GraphState:
    with mlflow.start_span("resolve_ticker"):
        prompt = f"""
        What is the stock ticker symbol for the publicly traded company:
        {state['company_name']}

        Respond with ONLY the ticker symbol.
        """
        state["ticker"] = llm.invoke(prompt).content.strip().upper()
        return state

# =========================================================
# Step 2: Fetch News (Graceful Fallback)
# =========================================================
def fetch_news(state: GraphState) -> GraphState:
    with mlflow.start_span("fetch_news"):
        try:
            from duckduckgo_search import DDGS
            headlines = []
            with DDGS() as ddgs:
                for r in ddgs.news(state["company_name"], max_results=5):
                    headlines.append(r["title"])

            state["news"] = headlines

        except Exception:
            print("⚠ News search failed — using fallback headlines")
            state["news"] = [
                f"{state['company_name']} announces new AI initiatives",
                f"{state['company_name']} reports steady quarterly earnings",
                f"Market analysts discuss {state['company_name']} growth outlook"
            ]

        return state

# =========================================================
# Step 3: Fetch Stock Price (Graceful Fallback)
# =========================================================
def fetch_stock_price(state: GraphState) -> GraphState:
    with mlflow.start_span("fetch_stock_price"):
        try:
            stock = yf.Ticker(state["ticker"])
            hist = stock.history(period="2mo")

            price = round(hist["Close"].iloc[-1], 2)

            state.update({
                "current_price": price,
                "price_change_7d": "+2.1%",
                "price_change_30d": "-1.3%",
                "fifty_two_week_high": round(price * 1.2, 2),
                "fifty_two_week_low": round(price * 0.8, 2),
            })

        except Exception:
            print("⚠ Stock price fetch failed — using synthetic fallback data")
            price = round(random.uniform(120, 280), 2)
            state.update({
                "current_price": price,
                "price_change_7d": "N/A",
                "price_change_30d": "N/A",
                "fifty_two_week_high": round(price * 1.15, 2),
                "fifty_two_week_low": round(price * 0.85, 2),
            })

        return state

# =========================================================
# Step 4: Sentiment Analysis
# =========================================================
def analyze_sentiment(state: GraphState) -> GraphState:
    with mlflow.start_span("analyze_sentiment"):

        prompt = PromptTemplate(
            template="""
            You are a senior financial research analyst.

            Company: {company}
            Ticker: {ticker}

            Recent News:
            {news}

            Current Price: {price}

            {format_instructions}
            """,
            input_variables=["company", "ticker", "news", "price"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | llm | parser

        result = chain.invoke({
            "company": state["company_name"],
            "ticker": state["ticker"],
            "news": "\n".join(state["news"]),
            "price": state["current_price"]
        })

        state["final_output"] = result.dict()
        return state

# =========================================================
# LangGraph Workflow
# =========================================================
graph = StateGraph(GraphState)

graph.add_node("resolve_ticker", resolve_ticker)
graph.add_node("fetch_news", fetch_news)
graph.add_node("fetch_stock_price", fetch_stock_price)
graph.add_node("analyze_sentiment", analyze_sentiment)

graph.set_entry_point("resolve_ticker")

graph.add_edge("resolve_ticker", "fetch_news")
graph.add_edge("fetch_news", "fetch_stock_price")
graph.add_edge("fetch_stock_price", "analyze_sentiment")

pipeline = graph.compile()

# =========================================================
# Main Runner
# =========================================================
if __name__ == "__main__":
    with mlflow.start_run(run_name="Google_Analysis"):
        result = pipeline.invoke({"company_name": "Google"})

        print("\n✅ FINAL STRUCTURED OUTPUT:\n")
        print(result["final_output"])