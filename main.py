import os
import uuid
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from crewai import Crew, Process
from agents import financial_analyst, verifier, investment_advisor, risk_assessor
from task import create_tasks

app = FastAPI(title="Financial Document Analyzer")

def run_crew(query: str, filepath: str):
    agents = {
        "verifier": verifier,
        "financial_analyst": financial_analyst,
        "investment_advisor": investment_advisor,
        "risk_assessor": risk_assessor,
    }

    tasks = create_tasks(query=query, file_path=filepath, agents=agents)

    financial_crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    time.sleep(10)
    result = financial_crew.kickoff()
    return result

@app.get("/")
async def root():
    return {"message": "Financial Document Analyzer API is running"}

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    query: str = Form(default="Analyze this financial document for investment insights")
):
    file_id = str(uuid.uuid4())
    filepath = f"data/financial_document_{file_id}.pdf"

    try:
        os.makedirs("data", exist_ok=True)

        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)

        if not query:
            query = "Analyze this financial document for investment insights"

        response = run_crew(query=query.strip(), filepath=filepath)
        return {
            "status": "success",
            "query": query,
            "analysis": str(response),
            "file_processed": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
