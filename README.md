# 🚀 AI Executive Whitepaper Deep-Dive 

An advanced multi-agent orchestration system that transforms a single topic into a professional executive whitepaper. Built with **LangGraph** for workflow control, **CrewAI** for role-based agent collaboration, and **Streamlit** for a seamless user experience.

---

## 🌟 Key Features
- **Intelligent Orchestration**: Uses LangGraph to manage complex states and ensure workflow reliability.
- **Autonomous Agent Teams**: Powered by CrewAI, featuring specialized agents for web research and technical writing.
- **Live Web Research**: Real-time information gathering using the DuckDuckGo Search tool.
- **Professional Deliverables**: 
  - 📄 Full Markdown whitepapers generated in-app.
  - 💾 Exportable PDF versions using ReportLab.
  - 📧 Direct SMTP email dispatch for instant sharing.
- **Real-time Performance Metrics**: Track execution time and quality depth scores for every deep-dive.

---

## 🛠️ Tech Stack
- **Frameworks**: [LangGraph](https://github.com/langchain-ai/langgraph), [CrewAI](https://github.com/crewAIInc/crewAI)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Search**: DuckDuckGo Search API
- **PDF Generation**: ReportLab
- **Environment**: Python 3.10+

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed. You will also need API keys for your chosen LLM (e.g., OpenAI, Anthropic, or Groq) if not running locally.

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/yourusername/ai-whitepaper-deepdive.git](https://github.com/yourusername/ai-whitepaper-deepdive.git)
cd ai-whitepaper-deepdive

# Install dependencies
pip install -r requirements.txt

```

### 3. Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_key_here
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

```

### 4. Running the App

```bash
streamlit run app.py

```

---

## 📊 Workflow Architecture

The system follows a stateful graph architecture:

1. **Input**: User provides a topic and email.
2. **Research Node**: CrewAI Researcher agent performs deep searches.
3. **Writing Node**: Writer agent synthesizes data into a whitepaper.
4. **Export**: The system triggers PDF generation and email dispatch.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📄 License

This project is licensed under the MIT License.
