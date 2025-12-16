# Story Finder (Streamlit-only)

This is a Streamlit app for semantic story search:
- Loads a JSON dataset of stories
- Creates one embedding per story (no chunking)
- Embeds the user query and returns top-k most similar stories
- Everything is in-memory (re-embed on restart)

## Required files
- streamlit_app.py
- requirements.txt
- stories-database.json

## Dataset format
`stories-database.json` must be a JSON array like:

[
  { "id": 1, "fullText": "Story text..." },
  { "id": 2, "fullText": "Another story..." }
]

## Run locally
1) Install deps:
   pip install -r requirements.txt

2) Set your OpenAI key (Streamlit secrets recommended):
   Create a file: .streamlit/secrets.toml
   with:
   OPENAI_API_KEY="your_key_here"

3) Start:
   streamlit run streamlit_app.py

## Deploy on Streamlit Community Cloud (free tier)
1) Push these files to GitHub:
   - streamlit_app.py
   - requirements.txt
   - stories-database.json

2) In Streamlit Cloud -> App -> Settings -> Secrets:
   Add:
   OPENAI_API_KEY="your_key_here"

3) Deploy.

## Notes
- With a few hundred stories, embedding time is usually acceptable.
- If/when you grow beyond that, the next step is persistence (FAISS/Chroma)
  so you don't have to re-embed each restart.
