# KB RAG service

RAG service over a 50-passage technical knowledge base (distributed systems, ML, databases, networking, security).

```
POST /query
{"question": "What port does PostgreSQL listen on?", "top_k": 3}
-> {"answer": "...port 5432.", "sources": [20, 21, 30], "scores": [0.83, 0.50, 0.48], "confidence": 0.83}
```

- Embeddings: fastembed `BAAI/bge-small-en-v1.5` (ONNX), hybrid with IDF keyword scoring
- Answers are extracted KB sentences (no LLM); abstains with `answer: null` when the KB lacks the fact
- CORS `*`; `GET /health` for health checks

Deploy: Render (Docker, free plan) via `render.yaml`. The container listens on `$PORT` (default 7860).
