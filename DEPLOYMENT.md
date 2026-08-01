# 🚀 Deployment Guide

## How to Run the App Locally & Host on Streamlit Cloud

## 📦 Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/GZ30eee/AI-Agentic-Studio.git
cd AI-Agentic-Studio
```

### 2. Create & Activate Virtual Environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy the example .env file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use any text editor
```

**Minimum required for different providers:**

| Provider | Required Keys |
|----------|---------------|
| **OpenAI** | `OPENAI_API_KEY=sk-...` |
| **Anthropic** | `ANTHROPIC_API_KEY=sk-ant-...` |
| **Gemini** | `GOOGLE_API_KEY=AIzaSy...` |
| **Ollama** | `OLLAMA_URL=http://localhost:11434` (default) |

**Optional for email:**
```
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_password
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
```

**Optional for observability (LangSmith):**
```
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-studio
```

### 5. (Optional) Install Ollama for Local LLMs

If you want to use Ollama (free, local):

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull phi3:mini
ollama pull llama3.1

# Start Ollama server (usually auto-starts)
ollama serve
```

### 6. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🧪 Quick Test

Test that everything works:

```bash
# Test Ollama (if using)
curl http://localhost:11434/api/tags

# Test API keys
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))"
```

---

## ☁️ Hosting on Streamlit Cloud

### Prerequisites

1. GitHub repository with your code
2. Streamlit Cloud account (free with GitHub)

### Step 1: Push Code to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. **Go to** [share.streamlit.io](https://share.streamlit.io)
2. **Sign in** with your GitHub account
3. Click **"New app"**
4. **Fill in the deployment form:**
   - **Repository:** `GZ30eee/AI-Agentic-Studio`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy"**

### Step 3: Configure Secrets (Environment Variables)

**IMPORTANT:** You cannot use a `.env` file on Streamlit Cloud. You must use **Secrets**.

1. In your app dashboard, go to **Settings** → **Secrets**
2. Add the following as TOML:

```toml
# .streamlit/secrets.toml

OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
GOOGLE_API_KEY = "AIzaSy..."
NEWS_API_KEY = "..."
OLLAMA_URL = "http://localhost:11434"  # Not needed in cloud unless you have a hosted Ollama

EMAIL_USER = "your_email@gmail.com"
EMAIL_PASS = "your_app_password"
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = "587"

LANGCHAIN_API_KEY = "ls__..."
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "agentic-studio"
```

3. Click **"Save"**

### Step 4: Modify app.py to Read Secrets

Add this at the top of `app.py` after importing streamlit:

```python
# Load secrets for Streamlit Cloud
if "secrets" in dir(st):
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
```

**OR**, update `load_dotenv()` to check for secrets first:

```python
# At the top of app.py, replace load_dotenv() with:
if "secrets" in dir(st):
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
else:
    load_dotenv()
```

### Step 5: Requirements for Cloud

Make sure your `requirements.txt` includes all dependencies. Streamlit Cloud automatically installs them.

**Important for ChromaDB:** Add this to `requirements.txt` if you encounter issues:

```
pysqlite3-binary ; sys_platform == 'linux'
```

### Step 6: Re-deploy

Click **"Reboot"** or push a new commit to trigger a rebuild.

---

## 🎯 Troubleshooting Common Issues

### Issue 1: Missing API Keys

**Error:** `OpenAI API key is missing`

**Fix:** Add the key to `.env` (local) or `st.secrets` (cloud)

### Issue 2: Ollama Not Reachable (Cloud)

**Problem:** `OLLAMA_URL=http://localhost:11434` won't work on Streamlit Cloud

**Solutions:**
- Use OpenAI/Anthropic/Gemini instead
- Host Ollama on a public server (not recommended for production)
- Use Hugging Face models via `llama-cpp-python` (requires code change)

### Issue 3: ChromaDB Persistence

**Problem:** ChromaDB tries to write to filesystem, which is limited on Streamlit Cloud

**Solution:** The code already falls back to `EphemeralClient()` if persistent fails:

```python
try:
    client = chromadb.PersistentClient(path="./chroma_db")
except:
    client = chromadb.EphemeralClient()  # Falls back automatically
```

### Issue 4: Large Dependencies

**Problem:** Deployment fails because of timeout during installation

**Solution:**
- Use `streamlit` minimal image (it's lightweight)
- Split dependencies into `requirements.txt` and `packages.txt` if needed

### Issue 5: Port Already in Use (Local)

**Error:** `Address already in use`

**Fix:**
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8501   # Windows, then kill PID
```

---

## 📊 Cost Optimization Tips

### For Local Development
- Use **Ollama** (free, local) for most testing
- Switch to OpenAI only for final reports

### For Production/Cloud
- Use **OpenAI gpt-4o-mini** (cheap: ~$0.15 per 1M tokens)
- Cache common responses
- Set `max_rpm=50` to avoid throttling

---

## 🚀 Quick Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] `requirements.txt` up to date
- [ ] No `.env` file in repo (use `.env.example` only)
- [ ] All API keys added to `st.secrets`
- [ ] `app.py` reads from `st.secrets` first
- [ ] Re-deployed after changes

---

## 🔗 Useful Links

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [Ollama Installation](https://ollama.ai/download)
- [LangSmith Setup](https://docs.smith.langchain.com/)
- [News API](https://newsapi.org/register)

---

## ✨ Success! You should now see:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

The app will be accessible at your Streamlit Cloud URL: `https://ai-agentic-studio-username.streamlit.app`
