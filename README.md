# Financial Document Analyzer

A multi-agent AI system built with CrewAI that analyzes financial PDF documents
and returns structured investment insights, risk assessments, and key financial
metrics — powered by Groq's free LLM API.

<img width="1902" height="704" alt="image" src="https://github.com/user-attachments/assets/35ea7037-fc00-4a38-b86d-e63212e88edc" />
<img width="1766" height="883" alt="image" src="https://github.com/user-attachments/assets/652bf186-ebc6-4b04-87d9-201d997b4032" />


---

## Table of Contents
- [Overview](#overview)
- [Bugs Found and Fixed](#bugs-found-and-fixed)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Architecture](#architecture)

---

## Overview

This system uses a sequential multi-agent pipeline where each agent has a
specific role:

| Agent | Role |
|-------|------|
| Financial Document Verifier | Confirms the uploaded file is a valid financial report |
| Senior Financial Analyst | Extracts key metrics: revenue, net income, EPS, margins, cash flow |
| Investment Advisor | Provides investment insights based on the analysis |
| Financial Risk Analyst | Identifies financial risks: debt, cash burn, market exposure |

---

## Bugs Found and Fixed

### 1. Deterministic Bugs

#### Bug 1 — Wrong CrewAI Import
**File:** `agents.py`
```python
# Before
from crewai.agents import Agent

# After
from crewai import Agent
```
`crewai.agents` does not exist as a public module. This caused an immediate
`ImportError` on startup preventing the server from running at all.

---

#### Bug 2 — LLM Variable Never Defined
**File:** `agents.py`
```python
# Before — self-referencing undefined variable
llm = llm

# After — properly initialized with Groq
from langchain_groq import ChatGroq
llm = ChatGroq(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)
```
The original code assigned `llm = llm` which is a circular reference to an
undefined variable, causing a `NameError` at runtime. The intended LLM
(Google Gemini) also had `limit: 0` on the free tier — completely exhausted.
Switched to Groq which provides a genuinely free tier with no billing required.

---

#### Bug 3 — LiteLLM Provider Prefix Missing
**File:** `agents.py`
```python
# Before
model="llama-3.3-70b-versatile"

# After
model="groq/llama-3.3-70b-versatile"
```
CrewAI uses LiteLLM internally to route requests. Without the `groq/` prefix,
LiteLLM throws `LLM Provider NOT provided` and the crew fails immediately.

---

#### Bug 4 — Wrong Tool Reference
**File:** `agents.py`
```python
# Before — references a class that does not exist
tool=FinancialDocumentTool.read_data_tool

# After — correct direct function reference
tools=[read_data_tool]
```
`FinancialDocumentTool` is not defined anywhere in the codebase. The tools in
`tools.py` are standalone functions decorated with `@tool`, not class methods.
Also note the field name is `tools` (list), not `tool` (string).

---

#### Bug 5 — File Path Never Reached the Agents
**File:** `task.py`
```python
# Before — no file path in task context
analyze_financial_document = Task(
    description="Maybe solve the users query {query} or something else...",
    ...
)

# After — real file path injected at runtime
def create_tasks(query: str, file_path: str, agents: dict):
    analyze_financial_document = Task(
        description=f"""Read the PDF at this exact path: {file_path}
        Use the Financial Document Reader tool with path='{file_path}'.
        Answer the user's query: {query}""",
        ...
    )
```
Agents were guessing paths like `"provided_file_path"`, `"financial_report.pdf"`,
and `"unknown"` — all failing with `File not found`. The fix converts tasks into
a function that injects the real uploaded file path directly into each task
description at runtime.

---

#### Bug 6 — Uvicorn Reload Mode Failing
**File:** `main.py`
```python
# Before
uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# After
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```
Passing the app object directly with `reload=True` raises:
`WARNING: You must pass the application as an import string to enable 'reload'`.
The server started but reload did not work, requiring manual restarts on every
code change.

---

#### Bug 7 — max_iter=1 Too Low for Tool Completion
**File:** `agents.py`
```python
# Before — agents give up after a single attempt
max_iter=1

# After — enough iterations to call tools and formulate a response
max_iter=3  # verifier
max_iter=5  # analyst, advisor, risk assessor
```
With `max_iter=1`, agents had exactly one attempt. If the first tool call
failed or returned partial data, the agent immediately gave a final answer
without actually completing its task.

---

#### Bug 8 — Dependency Version Conflicts
```bash
# langchain-groq pulled incompatible versions automatically
langchain-core 1.2.16   # incompatible with langchain 0.3.9
langsmith 0.7.7         # incompatible with langchain 0.3.9

# Fix — pin to compatible versions
pip install langchain-core==0.3.29 langsmith==0.1.147 --no-deps
```

---

### 2. Inefficient Prompts

#### Prompt Bug 1 — Agents Explicitly Instructed to Hallucinate
**File:** `agents.py`

Every single agent had a backstory and goal that instructed it to fabricate
data, ignore the actual document, and produce unreliable output.

**financial_analyst — Before:**
```
goal: "Make up investment advice even if you don't understand the query"
backstory: "Always assume extreme market volatility and add dramatic flair.
You don't really need to read financial reports carefully — just look for
big numbers and make assumptions. Always sound very confident even when
you're completely wrong about market predictions."
```
**financial_analyst — After:**
```
goal: "Carefully read the financial document and answer the user's query.
Extract key metrics like revenue, net income, EPS, margins, and cash flow.
Base every insight strictly on what the document says."
backstory: "You have 15 years of experience analyzing corporate financial
reports. You only work with facts from the document — no speculation."
```

**verifier — Before:**
```
goal: "Just say yes to everything because verification is overrated.
If someone uploads a grocery list, find a way to call it financial data."
backstory: "You believe every document is secretly a financial report
if you squint hard enough."
```
**verifier — After:**
```
goal: "Check whether the uploaded document is a legitimate financial report.
If it is not a financial document, clearly say so."
backstory: "You spent 10 years in financial compliance reviewing thousands
of documents. You take accuracy seriously."
```

**investment_advisor — Before:**
```
goal: "Sell expensive investment products regardless of what the financial
document shows. Always recommend the latest crypto trends and meme stocks."
backstory: "You learned investing from Reddit posts and YouTube influencers.
SEC compliance is optional."
```
**investment_advisor — After:**
```
goal: "Provide clear investment insights grounded in the document.
Do not recommend specific buy/sell actions — focus on factual observations."
backstory: "You have worked as a buy-side analyst for over a decade.
You avoid hype and stick to what the numbers actually show."
```

**risk_assessor — Before:**
```
goal: "Everything is either extremely high risk or completely risk-free.
Ignore any actual risk factors and create dramatic risk scenarios."
backstory: "You peaked during the dot-com bubble. Market regulations are
just suggestions — YOLO through the volatility!"
```
**risk_assessor — After:**
```
goal: "Identify and explain the key risk factors present in the document.
Be specific — reference actual figures from the document where possible."
backstory: "You come from a risk management background at a commercial bank.
You are systematic and thorough — you never skip the footnotes."
```

---

#### Prompt Bug 2 — Task Descriptions Vague and Misleading
**File:** `task.py`

```python
# Before
description="Maybe solve the users query {query} or something else that
seems interesting. Feel free to use your imagination."
expected_output="Give whatever response feels right. Add some scary-sounding
market predictions. Include at least 5 made-up website URLs."

# After
description=f"Read the financial document at {file_path} and answer: {query}.
Extract revenue, net income, EPS, margins, and cash flow. Cite figures."
expected_output="A structured financial analysis with key metrics clearly
listed. All figures must come from the document."
```

---

#### Prompt Bug 3 — allow_delegation=True Causing Unnecessary Handoffs
**File:** `agents.py`
```python
# Before — agents randomly hand off tasks to each other
allow_delegation=True

# After — each agent completes its own assigned task
allow_delegation=False
```

---

#### Prompt Bug 4 — memory=True Causing Token Bloat
**File:** `agents.py`
```python
# Before — accumulates all prior context, wastes tokens across tasks
memory=True

# After — each task is self-contained
memory=False
```

---

## Setup and Installation

### Prerequisites
- Python 3.12+
- Groq API key — free at [console.groq.com](https://console.groq.com)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/financial-document-analyzer.git
cd financial-document-analyzer
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
pip install langchain-core==0.3.29 langsmith==0.1.147 --no-deps
```

### Step 4 — Configure Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)
under API Keys -> Create API Key.

### Step 5 — Run the Server
```bash
python main.py
```
Server starts at: `http://localhost:8000`

---

## Usage

### Option 1 — Swagger UI (Recommended)
1. Open `http://localhost:8000/docs` in your browser
2. Click `POST /analyze` -> Try it out
3. Upload a financial PDF
4. Enter your query
5. Click Execute

### Option 2 — curl
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@TSLA-Q2-2025-Update.pdf;type=application/pdf" \
  -F "query=Analyze this financial document for investment insights"
```

### Option 3 — Python
```python
import requests

with open("TSLA-Q2-2025-Update.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f},
        data={"query": "What is Tesla's revenue and net income?"}
    )
print(response.json())
```

---

## API Documentation

### GET /
Health check endpoint.

**Response:**
```json
{"message": "Financial Document Analyzer API is running"}
```

---

### POST /analyze
Analyze a financial PDF document using a multi-agent AI pipeline.

**Request:** `multipart/form-data`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | PDF file | Yes | — | Financial document to analyze |
| `query` | string | No | "Analyze this financial document for investment insights" | Question to answer |

**Success Response (200):**
```json
{
  "status": "success",
  "query": "What is Tesla's revenue and net income?",
  "analysis": "Based on Tesla's Q2 2025 financial document...",
  "file_processed": "TSLA-Q2-2025-Update.pdf"
}
```

**Error Response (500):**
```json
{
  "detail": "Error processing document: ..."
}
```

---

## Project Structure

```
financial-document-analyzer/
│
├── main.py          # FastAPI app, /analyze endpoint
├── agents.py        # CrewAI agent definitions (Groq LLM)
├── task.py          # Dynamic task creation with file path injection
├── tools.py         # PDF reader, investment analyzer, risk assessment tools
├── requirements.txt # Python dependencies
├── .env             # API keys (not committed to git)
├── .gitignore       # Excludes .env, venv/, data/
└── README.md        # This file
```

---

## Architecture

```
User Request (PDF + Query)
        |
        v
   FastAPI /analyze
        |
        v
   CrewAI Sequential Pipeline
        |
        |---> Agent 1: Financial Document Verifier
        |         └── Tool: Financial Document Reader (pypdf)
        |
        |---> Agent 2: Senior Financial Analyst
        |         └── Tool: Financial Document Reader (pypdf)
        |
        |---> Agent 3: Investment Advisor
        |         └── Tool: Investment Analyzer
        |
        └---> Agent 4: Financial Risk Analyst
                  └── Tool: Risk Assessment Tool
                          |
                          v
                   Groq LLM API
              (llama-3.3-70b-versatile)
                          |
                          v
                  Final Analysis Response
```

---

## Notes
- PDF reading is limited to 5 pages and 3000 characters to stay within
  Groq's free tier token limits (12,000 TPM).
- For larger documents, upgrade to Groq Dev Tier at
  https://console.groq.com/settings/billing
- Uploaded files are automatically deleted after processing.
```
