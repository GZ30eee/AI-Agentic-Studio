import os
import hashlib
import time
import json
import logging
from datetime import datetime
from typing import TypedDict, Type

import streamlit as st
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv

# Core Frameworks
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

# PDF & Email
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

load_dotenv()

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- TOOL (Safe wrapper) ---
class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")

class DuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Search internet using DuckDuckGo"
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str) -> str:
        try:
            search = DuckDuckGoSearchRun()
            return search.run(query)
        except Exception as e:
            return f"Search error for '{query}'. Using internal knowledge base."

search_tool = DuckDuckGoTool()

# --- STATE ---
class AgentState(TypedDict):
    topic: str
    email: str
    plan: str
    raw_research: str
    final_report: str
    quality_score: float
    status: str

# --- LLM CONFIG (High Token Limit for Long Content) ---
extra_params = {"max_rpm": 50}
local_llm = LLM(
    model="ollama/llama3.1", 
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    config={"num_predict": 4096, "temperature": 0.7} # Increased output capacity
)

# --- NODES ---
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def planning_node(state: AgentState):
    logger.info(f"Planning: {state['topic']}")
    director = Agent(
        role='Research Director', 
        goal=f'Strategic plan for {state["topic"]}',
        backstory="Senior Mastek strategist specialized in deep-dive whitepapers.", 
        llm=local_llm, **extra_params
    )
    task = Task(description=f"Plan an extensive 5-section research structure for {state['topic']}", agent=director, expected_output="A detailed 5-point roadmap.")
    result = Crew(agents=[director], tasks=[task], timeout=120).kickoff()
    return {"plan": str(result), "status": "Structure Planned"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def research_node(state: AgentState):
    researcher = Agent(
        role='Lead Researcher', 
        goal=f'Find comprehensive data on {state["topic"]}',
        backstory="Expert data analyst who gathers technical details, statistics, and trends.", 
        tools=[search_tool], llm=local_llm, **extra_params
    )
    task = Task(
        description=f"Gather detailed technical trends and market data for: {state['topic']}. Provide enough detail for a 1000-word report.", 
        agent=researcher, 
        expected_output="Detailed research notes with at least 10 key findings."
    )
    result = Crew(agents=[researcher], tasks=[task], timeout=180).kickoff()
    return {"raw_research": str(result), "status": "Deep Research Complete"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def writing_node(state: AgentState):
    writer = Agent(
        role='Senior Technical Consultant', 
        goal='Create a high-impact, long-form executive whitepaper.',
        backstory="Lead consultant at Mastek. You are known for transforming brief notes into expansive, 1000-word professional documents.", 
        llm=local_llm, **extra_params
    )
    # Instruction for more content
    task = Task(
        description=f"""Based on this research: {state.get('raw_research', '')}, 
        write a comprehensive technical whitepaper. 
        YOU MUST INCLUDE:
        1. EXECUTIVE SUMMARY (Detailed)
        2. MARKET DYNAMICS & RECENT TRENDS (Deep dive into each research point)
        3. STRATEGIC RECOMMENDATIONS FOR ENTERPRISES
        4. TECHNICAL CHALLENGES & SOLUTIONS
        5. FUTURE OUTLOOK & CONCLUSION
        Be verbose, professional, and thorough. Aim for significant depth in every section.""", 
        agent=writer, 
        expected_output="A multi-page, detailed formal report."
    )
    result = Crew(agents=[writer], tasks=[task], timeout=300).kickoff() # High timeout for long text
    return {"final_report": str(result), "status": "Full-Length Report Generated"}

def quality_node(state: AgentState):
    # Higher threshold for length to ensure "more content"
    report = state.get("final_report", "")
    score = max(0, len(report) / 800) 
    return {"quality_score": score, "status": f"Quality/Length Score: {score:.1f}"}

def route_quality(state: AgentState):
    # If report is too short, send it back for expansion
    if state.get("quality_score", 0) < 1.8:
        return "writer"
    return END

# --- GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planning_node)
workflow.add_node("researcher", research_node)
workflow.add_node("writer", writing_node)
workflow.add_node("quality_gate", quality_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "quality_gate")
workflow.add_conditional_edges("quality_gate", route_quality, {"writer": "writer", END: END})
app_graph = workflow.compile()

# --- UTILITIES ---
def generate_pdf(content: str, topic: str):
    report_id = hashlib.md5(topic.lower().encode()).hexdigest()[:8]
    filename = f"Mastek_DeepDive_{report_id}.pdf"
    try:
        doc = SimpleDocTemplate(filename, pagesize=LETTER)
        styles = getSampleStyleSheet()
        
        # Professional styling
        style_body = styles['BodyText']
        style_body.leading = 14
        
        story = [
            Paragraph(f"MASTEK ENTERPRISE RESEARCH: {topic.upper()}", styles['Title']),
            Spacer(1, 24)
        ]
        
        # Split content into paragraphs for ReportLab
        paragraphs = str(content).split('\n')
        for p in paragraphs:
            if p.strip():
                # Detect and format headers
                if p.strip().startswith('#') or len(p) < 60:
                    story.append(Paragraph(p.replace('#', ''), styles['Heading2']))
                else:
                    story.append(Paragraph(p, style_body))
                story.append(Spacer(1, 10))
        
        doc.build(story)
        return filename
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return None

def send_email(email: str, pdf_path: str):
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    if not all([sender, password, pdf_path]):
        return False
    
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = email
    msg['Subject'] = f"Deep-Dive Research Report: {topic}"
    msg.attach(MIMEText("Please find your comprehensive research report attached.", 'plain'))
    
    try:
        with open(pdf_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(pdf_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(pdf_path)}"'
            msg.attach(part)
        
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        return True
    except Exception as e:
        logger.error(f"Email error: {e}")
        return False

# --- STREAMLIT ---
st.set_page_config(page_title="Mastek Deep-Dive Studio", layout="wide")
st.title("🤖 Mastek Agentic Studio - Tier 3")

with st.sidebar:
    st.header("⚙️ Config")
    topic = st.text_input("Topic", "Impact of GenAI on Software Development Lifecycle 2026")
    email = st.text_input("Email")
    run_btn = st.button("🚀 Launch Deep Research", use_container_width=True, type="primary")

if run_btn:
    start_time = time.time()
    final_state = {}
    
    with st.status("🤖 Deep Research Agents at work (This may take 5-8 mins)...", expanded=True) as status:
        inputs = {"topic": topic, "email": email}
        
        try:
            for output in app_graph.stream(inputs):
                for node, data in output.items():
                    st.write(f"✅ **{node}**: {data.get('status', 'Done')}")
                    final_state.update(data)
            
            duration = round(time.time() - start_time, 2)
            status.update(label=f"✅ Deep Dive Complete in {duration}s!", state="complete")
            
        except Exception as e:
            st.error(f"Workflow error: {e}")
            status.update(label="❌ Failed", state="error")
    
    # Results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Full Executive Whitepaper")
        st.markdown(final_state.get("final_report", "No report."))
    
    with col2:
        st.subheader("📦 Professional Deliverables")
        pdf_path = generate_pdf(final_state.get("final_report", ""), topic)
        
        if pdf_path:
            with open(pdf_path, "rb") as f:
                st.download_button("💾 Download Whitepaper (PDF)", f, file_name=pdf_path)
            
            if email and st.button("📧 Send to Email"):
                with st.spinner("Dispatching..."):
                    if send_email(email, pdf_path):
                        st.success("✅ Dispatched via SMTP!")
                    else:
                        st.warning("Check .env credentials")
        
        st.subheader("📊 Performance Metrics")
        st.json({
            "execution_time": f"{duration}s",
            "quality_depth_score": round(final_state.get("quality_score", 0), 2),
            "engine": "LangGraph + CrewAI + Llama 3.1",
            "mode": "Long-form Drafting"
        })

st.info("🎯 **Enterprise Ready** - This model utilizes recursive quality checks to ensure content depth.")