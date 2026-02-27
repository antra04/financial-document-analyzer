import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent
from langchain_groq import ChatGroq
from tools import read_data_tool, analyze_investment_tool, create_risk_assessment_tool

llm = ChatGroq(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

financial_analyst = Agent(
    role="Senior Financial Analyst",
    goal="Read the financial document and answer the user's query: {query}. Extract key metrics like revenue, net income, EPS, margins, and cash flow.",
    verbose=True,
    memory=False,
    backstory="You have 15 years of experience analyzing corporate financial reports. You only work with facts from the document.",
    tools=[read_data_tool],
    llm=llm,
    max_iter=3,
    max_rpm=1,
    allow_delegation=False
)

verifier = Agent(
    role="Financial Document Verifier",
    goal="Check whether the uploaded document is a legitimate financial report. Look for balance sheets, income statements, or earnings data.",
    verbose=True,
    memory=False,
    backstory="You spent 10 years in financial compliance reviewing thousands of documents.",
    tools=[read_data_tool],
    llm=llm,
    max_iter=2,
    max_rpm=1,
    allow_delegation=False
)

investment_advisor = Agent(
    role="Investment Advisor",
    goal="Based on the financial analysis, provide clear investment insights for the query: {query}. Highlight strengths, weaknesses, and trends.",
    verbose=True,
    memory=False,
    backstory="You have worked as a buy-side analyst for over a decade. You translate financial data into plain language.",
    tools=[analyze_investment_tool],
    llm=llm,
    max_iter=3,
    max_rpm=1,
    allow_delegation=False
)

risk_assessor = Agent(
    role="Financial Risk Analyst",
    goal="Identify key risk factors in the financial document. Look at debt levels, cash burn, revenue concentration, and risk disclosures.",
    verbose=True,
    memory=False,
    backstory="You come from a risk management background at a commercial bank. You are systematic and thorough.",
    tools=[create_risk_assessment_tool],
    llm=llm,
    max_iter=3,
    max_rpm=1,
    allow_delegation=False
)
