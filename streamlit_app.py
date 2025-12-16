import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from openai import OpenAI

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Story Finder", page_icon="📖", layout="centered")

DEFAULT_DATASET_PATH = "stories-database.json"
DEFAULT_TOP_K = 5

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# For a first iteration, a simple character cap is a practical safety guard.
# (If you later want token-accurate truncation, we can add tiktoken.)
MAX_EMBED_CHARS = 12000


# -----------------------------
# Data types
# -----------------------------
@dataclass
class Story:
    id: Any
    fullText: str


@dataclass
class SearchResult:
    id: Any
    fullText: str
    similarity: float


# -----------------------------
# Helpers
# -----------------------------
def _truncate_for_embedding(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= MAX_EMBED_CHARS:
        return text
    return text[:MAX_EMBED_CHARS]


def _validate_and_parse_stories(raw: Any) -> List[Story]:
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("Stories JSON must be a non-empty array of objects.")

    stories: List[Story] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Story at index {i} is not an object.")
        if "id" not in item or "fullText" not in item:
            raise ValueError('Each story must have "id" and "fullText" fields.')
        if not isinstance(item["fullText"], str) or not item["fullText"].strip():
            raise ValueError(f'Story id={item.get("id")} has empty "fullText".')
        stories.append(Story(id=item["id"], fullText=item["fullText"].strip()))
    return stories


def load_stories_from_file(path: str) -> List[Story]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _validate_and_parse_stories(raw)


def load_stories_from_upload(uploaded_bytes: bytes) -> List[Story]:
    raw = json.loads(uploaded_bytes.decode("utf-8"))
    return _validate_and_parse_stories(raw)


def get_openai_client() -> OpenAI:
    # Streamlit Community Cloud: put OPENAI_API_KEY in Secrets.
    # If running locally, you can also set it as an environment variable;
    # openai SDK can pick it up, but we prefer explicit secrets here.
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it in Streamlit Secrets:\n"
            'OPENAI_API_KEY="your_key_here"'
        )
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """
    Returns shape: (n_texts, dim) float32 numpy array.
    Uses batching in one API call when possible.
    """
    cleaned = [_truncate_for_embedding(t) for t in texts]
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=cleaned)
    # OpenAI returns embeddings in resp.data in input order
    embs = [d.embedding for d in resp.data]
    return np.array(embs, dtype=np.float32)


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def cosine_top_k(
    query_vec: np.ndarray,
    doc_mat: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    """
    query_vec: (dim,)
    doc_mat: (n_docs, dim) normalized
    Returns list of (index, similarity) sorted desc.
    """
    # Ensure shapes
    q = query_vec.reshape(1, -1).astype(np.float32)
    q = normalize_rows(q)[0]
    sims = doc_mat @ q  # (n_docs,)
    k = max(1, min(k, sims.shape[0]))
    top_idx = np.argpartition(-sims, kth=k - 1)[:k]
    top_sorted = top_idx[np.argsort(-sims[top_idx])]
    return [(int(i), float(sims[i])) for i in top_sorted]


# -----------------------------
# Session state
# -----------------------------
if "stories" not in st.session_state:
    st.session_state.stories = []

if "story_embeddings" not in st.session_state:
    st.session_state.story_embeddings = None

if "embedded_story_count" not in st.session_state:
    st.session_state.embedded_story_count = 0

if "last_dataset_label" not in st.session_state:
    st.session_state.last_dataset_label = ""



# -----------------------------
# UI
# -----------------------------
st.title("📖 Story Finder")
st.caption("Semantic story search (RAG-style retrieval) — Streamlit-only deployment.")

with st.expander("What this app does", expanded=False):
    st.write(
        "- Loads a JSON dataset of stories (each has `id` and `fullText`).\n"
        "- Generates **one embedding per story**.\n"
        "- Embeds your query and returns the most semantically similar stories.\n"
        "- Everything is **in-memory** (you’ll re-embed if the app restarts)."
    )

st.divider()

# -----------------------------
# Dataset loader
# -----------------------------
st.subheader("1) Load stories")

dataset_mode = st.radio(
    "Choose dataset source:",
    options=["Use repo file (stories-database.json)", "Upload JSON"],
    index=0,
    horizontal=True,
)

uploaded = None
dataset_label = ""

try:
    if dataset_mode == "Upload JSON":
        uploaded = st.file_uploader(
            "Upload a JSON file (array of { id, fullText })",
            type=["json"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            dataset_label = f"upload:{uploaded.name}"
    else:
        dataset_label = f"file:{DEFAULT_DATASET_PATH}"

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        if st.button("Load stories", type="primary", use_container_width=True):
            if dataset_mode == "Upload JSON":
                if uploaded is None:
                    st.error("Please upload a JSON file first.")
                else:
                    stories = load_stories_from_upload(uploaded.getvalue())
                    st.session_state.stories = stories
            else:
                stories = load_stories_from_file(DEFAULT_DATASET_PATH)
                st.session_state.stories = stories

            # Reset embeddings whenever dataset changes
            st.session_state.story_embeddings = None
            st.session_state.embedded_story_count = 0
            st.session_state.last_dataset_label = dataset_label

    with col_b:
        if st.button("Clear dataset", use_container_width=True):
            st.session_state.stories = []
            st.session_state.story_embeddings = None
            st.session_state.embedded_story_count = 0
            st.session_state.last_dataset_label = ""
            st.success("Cleared.")

except Exception as e:
    st.error(f"Dataset error: {e}")

if st.session_state.stories:
    st.success(f"Loaded {len(st.session_state.stories)} stories.")
else:
    st.info("No stories loaded yet.")

st.divider()

# -----------------------------
# Embedding / indexing
# -----------------------------
st.subheader("2) Generate embeddings (indexing)")

if not st.session_state.stories:
    st.warning("Load stories first.")
else:
    top_k = st.slider("How many results to return (k)?", 1, 20, DEFAULT_TOP_K)

    embed_col1, embed_col2 = st.columns([1, 1], gap="large")

    with embed_col1:
        if st.button("Generate embeddings", use_container_width=True):
            try:
                client = get_openai_client()
                texts = [s.fullText for s in st.session_state.stories]

                progress = st.progress(0, text="Embedding stories...")
                # Batch in chunks to be safe with a few hundred stories
                batch_size = 64
                all_embs: List[np.ndarray] = []

                n = len(texts)
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    batch = texts[start:end]
                    embs = embed_texts(client, batch)
                    all_embs.append(embs)

                    pct = int((end / n) * 100)
                    progress.progress(pct, text=f"Embedding stories... ({end}/{n})")

                story_mat = np.vstack(all_embs)
                story_mat = normalize_rows(story_mat)

                st.session_state.story_embeddings = story_mat
                st.session_state.embedded_story_count = len(st.session_state.stories)

                progress.progress(100, text="Done.")
                st.success(f"Embedded {len(st.session_state.stories)} stories.")
            except Exception as e:
                st.error(f"Embedding failed: {e}")

    with embed_col2:
        ready = st.session_state.story_embeddings is not None
        st.metric(
            "Index status",
            value="Ready ✅" if ready else "Not ready ❌",
            delta=f"{st.session_state.embedded_story_count} embedded" if ready else "Embed to enable search",
        )

st.divider()

# -----------------------------
# Search
# -----------------------------
st.subheader("3) Search")

query = st.text_input(
    "Describe the kind of story you want:",
    placeholder="e.g., overcoming failure, finding purpose, friendship, loneliness, starting over",
)

search_btn = st.button("Search", disabled=not query.strip(), use_container_width=True)

if search_btn:
    if not st.session_state.stories:
        st.error("No stories loaded.")
    elif st.session_state.story_embeddings is None:
        st.error("Please generate embeddings first.")
    else:
        try:
            client = get_openai_client()
            with st.spinner("Embedding query and searching..."):
                q_vec = embed_texts(client, [query])[0]
                top = cosine_top_k(q_vec, st.session_state.story_embeddings, k=top_k)

            results: List[SearchResult] = []
            for idx, sim in top:
                s = st.session_state.stories[idx]
                results.append(SearchResult(id=s.id, fullText=s.fullText, similarity=sim))

            st.subheader("Results")
            for r in results:
                with st.expander(f"Story {r.id} — similarity {r.similarity:.3f}"):
                    st.write(r.fullText)

        except Exception as e:
            st.error(f"Search failed: {e}")

st.divider()

# -----------------------------
# Quick dataset preview
# -----------------------------
if st.session_state.stories:
    with st.expander("Preview dataset (first 5 stories)", expanded=False):
        for s in st.session_state.stories[:5]:
            preview = s.fullText[:220] + ("..." if len(s.fullText) > 220 else "")
            st.markdown(f"**ID:** {s.id}")
            st.write(preview)
            st.write("---")
