services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OLLAMA_URL=http://ollama:11434
      - EMAIL_USER=${EMAIL_USER}
      - EMAIL_PASS=${EMAIL_PASS}
    volumes:
      - ./chroma_db:/app/chroma_db
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ./ollama_data:/root/.ollama