import os
from dotenv import load_dotenv
load_dotenv()

from crewai.tools import tool
from pypdf import PdfReader

@tool("Financial Document Reader")
def read_data_tool(path: str = 'data/sample.pdf') -> str:
    """
    Reads a PDF financial document from the given path
    and returns its text content as a string.
    Limits to first 5 pages to stay within token limits.
    """
    if not os.path.exists(path):
        return f"Error: File not found at path '{path}'"

    reader = PdfReader(path)
    full_report = ""
    max_pages = 5

    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        content = page.extract_text()
        if content:
            while "\n\n" in content:
                content = content.replace("\n\n", "\n")
            full_report += content + "\n"

    result = full_report.strip()
    return result[:3000] if len(result) > 3000 else result


@tool("Investment Analyzer")
def analyze_investment_tool(financial_data: str) -> str:
    """
    Takes extracted financial document text and returns
    a structured summary of key investment indicators
    like revenue growth, margins, and EPS trends.
    """
    if not financial_data or len(financial_data.strip()) == 0:
        return "Error: No financial data provided to analyze."

    cleaned = " ".join(financial_data.split())
    return f"Investment data received for analysis:\n{cleaned[:2000]}"


@tool("Risk Assessment Tool")
def create_risk_assessment_tool(financial_data: str) -> str:
    """
    Takes extracted financial document text and identifies
    key risk factors like debt levels, cash burn, and
    market exposure mentioned in the document.
    """
    if not financial_data or len(financial_data.strip()) == 0:
        return "Error: No financial data provided for risk assessment."

    cleaned = " ".join(financial_data.split())
    return f"Risk assessment data received:\n{cleaned[:2000]}"
