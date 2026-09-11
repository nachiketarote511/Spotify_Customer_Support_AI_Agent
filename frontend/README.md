# Frontend — Hiver AI Customer Support Agent

Single-page chatbot UI for the Hiver AI Customer Support Agent.

## Serving

The frontend is served automatically by the backend API at `http://localhost:8000/`.

For standalone development, you can use any static file server:

```bash
cd frontend/public
python -m http.server 5500
```

Then configure the API URL in the HTML file if needed.

## Structure

```
frontend/
└── public/
    └── index.html    # Complete chatbot UI (HTML + CSS + JS)
```
