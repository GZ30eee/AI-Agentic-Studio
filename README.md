# AI Agentic Studio

AI Agentic Studio is a comprehensive multi-agent research and reporting framework. It leverages advanced language models and agentic orchestration to perform deep-dive research, verify facts, and generate professional whitepapers in various formats.

## Core Features
- Multi-Agent Orchestration: Strategic coordination between planning, research, and writing agents using CrewAI and LangGraph.
- Retrieval-Augmented Generation (RAG): Bring your own data by uploading PDF, TXT, or CSV files to enhance research accuracy.
- Multi-Model Support: Native integration with OpenAI, Anthropic (Claude), Google (Gemini), and local execution via Ollama.
- High-Impact Reporting: Automatically generate professional whitepapers, strategic recommendations, and executive summaries.
- Export Formats: Support for PDF, DOCX, PPTX, and Markdown exports.
- Secure Reporting: Built-in email functionality for delivering generated whitepapers directly to stakeholders.
- Fact-Checking and Quality Control: Automated review cycles to ensure research depth and factual consistency.

## Tech Stack
- Frontend: Streamlit
- Agent Framework: CrewAI, LangGraph
- Vector Database: ChromaDB
- LLMs: OpenAI, Anthropic, Google Gemini, Ollama
- Documentation: ReportLab, Python-Docx, Python-Pptx, Markdown

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GZ30eee/AI-Agentic-Studio.git
   cd AI-Agentic-Studio
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Copy `.env.example` to `.env` and fill in your API keys.

## Running the Application
To start the studio, run:
```bash
streamlit run app.py
```

## Usage
- Enter a research topic and the number of researchers you wish to deploy.
- (Optional) Upload custom documents to the sidebar to ground the research in your own data.
- Monitor the agentic workflow as it progresses through planning, research, and writing.
- Review and export the final whitepaper directly from the interface or receive it via email.

## License
MIT License
