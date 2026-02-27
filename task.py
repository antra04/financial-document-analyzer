from crewai import Task

def create_tasks(query: str, file_path: str, agents: dict):

    verification = Task(
        description=f"""Read the PDF financial document located at this exact path: {file_path}
        Use the Financial Document Reader tool with path='{file_path}'.
        Verify it is a legitimate financial report by looking for income statements,
        balance sheets, cash flow data, or earnings figures.""",
        expected_output="A clear verdict: either 'This is a valid financial document' with a brief summary of what financial data it contains, or 'This is NOT a financial document' with an explanation.",
        agent=agents["verifier"],
    )

    analyze_financial_document = Task(
        description=f"""Read the PDF financial document at this exact path: {file_path}
        Use the Financial Document Reader tool with path='{file_path}'.
        Answer the user's query: {query}
        Extract and explain key financial metrics — revenue, net income, EPS, margins, cash flow.
        Be specific and cite figures from the document.""",
        expected_output="A structured financial analysis with key metrics clearly listed, directly addressing the user's query. All figures must come from the document.",
        agent=agents["financial_analyst"],
    )

    investment_analysis = Task(
        description=f"""Based on the financial analysis from the previous task, provide investment insights.
        The document is at: {file_path}
        User query: {query}
        Highlight financial strengths, weaknesses, and trends visible in the data.
        Do not make specific buy/sell recommendations.""",
        expected_output="A clear investment insight report with 3-5 key observations grounded in the document's financial data.",
        agent=agents["investment_advisor"],
    )

    risk_assessment = Task(
        description=f"""Based on the financial analysis from the previous tasks, identify key financial risks.
        The document is at: {file_path}
        Check debt-to-equity ratio, cash burn rate, revenue concentration, and any stated risk factors.
        Reference actual numbers from the document where possible.""",
        expected_output="A structured risk assessment with 3-5 specific risks, each backed by figures or disclosures from the document.",
        agent=agents["risk_assessor"],
    )

    return [verification, analyze_financial_document, investment_analysis, risk_assessment]
