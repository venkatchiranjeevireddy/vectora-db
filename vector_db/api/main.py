import io
import os
import re
import time
from typing import Dict, List, Union
from pathlib import Path

import numpy as np  # type: ignore
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pypdf import PdfReader
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from groq import Groq

from vector_db.hnsw.graph import HNSWGraph
from vector_db.hnsw.insert import insert
from vector_db.hnsw.search import hnsw_search


class InsertRequest(BaseModel):
    id: Union[str, int]
    vector: List[float] = Field(..., description="Vector values to insert.")


class InsertResponse(BaseModel):
    status: str
    id: Union[str, int]


class SearchRequest(BaseModel):
    vector: List[float] = Field(..., description="Query vector values.")
    k: int = Field(..., ge=1, description="Number of nearest neighbors to return.")
    ef_search: int = Field(..., ge=1, description="Search width for HNSW.")


class SearchResult(BaseModel):
    id: Union[str, int]
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class StatsResponse(BaseModel):
    total_vectors: int
    entry_point: Union[str, int, None]
    entry_level: Union[int, None]


class UploadPdfResponse(BaseModel):
    status: str
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(5, ge=1)
    ef_search: int = Field(100, ge=1)


class AskResultItem(BaseModel):
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    llm_answer: str
    results: List[AskResultItem]
    latency_ms: float
    total_chunks_searched: int


class IndexStatsResponse(BaseModel):
    total_chunks: int
    entry_point: Union[str, int, None]
    index_health: str


# Global state requested by user
graph = HNSWGraph(M=32, ef_construction=400)
vectors = {}
chunk_texts: Dict[str, str] = {}
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_counter = 0
groq_client = None


app = FastAPI(title="Vector DB API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def smart_chunk(text: str) -> List[str]:
    """
    Smart resume chunking designed for noisy PDF extraction.

    Steps:
    1) Split by known section headers.
    2) Split each section by newlines and keep meaningful lines.
    3) Group every 3 lines into one focused chunk.
    """

    # Step 1: Split by resume section keywords.
    section_pattern = (
        r"\n(?=Education|Experience|Projects|Skills|Technical Skills|"
        r"Certifications|Summary|Internship|Leadership)"
    )
    sections = re.split(section_pattern, text)

    chunks: List[str] = []

    header_pattern = re.compile(
        r"^(Education|Experience|Projects|Skills|Technical Skills|Certifications|Summary|Internship|Leadership)\b",
        flags=re.IGNORECASE,
    )

    for section in sections:
        if len(section.strip()) < 20:
            continue

        # Step 2: Split each section by newlines, keep sufficiently long lines.
        lines = [l.strip() for l in section.split("\n") if len(l.strip()) > 20]
        if not lines:
            continue

        # Infer current section header from the section's first line.
        first_line = lines[0]
        match = header_pattern.match(first_line)
        current_header = match.group(1) if match else "General"

        # Step 3: Group every 3 lines into one chunk.
        for i in range(0, len(lines), 3):
            group = lines[i : i + 3]
            chunk_text = f"[{current_header}] {' '.join(group)}"
            if len(chunk_text.split()) > 10:
                chunks.append(chunk_text)

    # Deduplicate chunks using first 50 chars as key.
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = chunk[:50].strip()
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    chunks = unique_chunks

    return chunks


def _tokenize_for_overlap(text: str) -> List[str]:
    """
    Lowercase tokenization for lightweight lexical overlap reranking.
    """

    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _intent_boost(question_tokens: set, chunk_tokens: set) -> float:
    """
    Small heuristic boost for intent-aligned chunks and penalty for profile headers.
    """

    # Penalize contact/header-like chunks for content-focused questions.
    contact_terms = {"phone", "email", "linkedin", "github", "kadapa", "andhra", "india"}
    contact_hits = len(chunk_tokens.intersection(contact_terms))
    penalty = 0.08 * contact_hits

    # Intent-specific boosts (simple lexical priors).
    experience_terms = {
        "experience", "intern", "internship", "worked", "work", "role",
        "engineer", "project", "projects", "apis", "backend", "frontend",
        "technical", "skills",
    }
    education_terms = {
        "education", "degree", "btech", "b", "tech", "university",
        "institute", "iiit", "coursework",
    }

    boost = 0.0
    if question_tokens.intersection({"experience", "internship", "work", "worked", "role"}):
        boost += 0.12 * len(chunk_tokens.intersection(experience_terms))
    if question_tokens.intersection({"education", "degree", "study", "studied", "college"}):
        boost += 0.12 * len(chunk_tokens.intersection(education_terms))

    return boost - penalty


def _expand_question(question: str) -> str:
    """
    Expand short queries with resume-oriented context terms.
    """

    expansions = {
        "skills": "technical skills programming languages frameworks libraries tools",
        "experience": "work experience internship job role responsibilities",
        "projects": "projects built developed implemented created",
        "education": "education degree university college coursework",
        "certifications": "certifications courses workshops training",
    }

    expanded_question = question.strip()
    for keyword, expansion in expansions.items():
        if keyword in expanded_question.lower():
            expanded_question = expanded_question + " " + expansion
            break

    # Keep short-query expansion requested earlier for robustness.
    if len(expanded_question.split()) < 4:
        expanded_question = expanded_question + " experience background knowledge"

    return expanded_question


def _chunk_context_text(chunk_id: str) -> str:
    """
    Return merged context text: previous + current + next chunk when available.
    """

    # Only apply neighbor lookup for ids in format "chunk_<int>".
    if not chunk_id.startswith("chunk_"):
        return chunk_texts.get(chunk_id, "")

    try:
        idx = int(chunk_id.split("_", 1)[1])
    except ValueError:
        return chunk_texts.get(chunk_id, "")

    ids = [f"chunk_{idx - 1}", f"chunk_{idx}", f"chunk_{idx + 1}"]
    parts = [chunk_texts[cid] for cid in ids if cid in chunk_texts]
    return " ".join(parts).strip()


def _get_groq_client():
    """
    Lazily create Groq client from current environment.
    """

    global groq_client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fallback: load .env directly if server wasn't started with --env-file.
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None
    if groq_client is None:
        groq_client = Groq(api_key=api_key)
    return groq_client


@app.get("/")
def serve_frontend():
    return FileResponse("vector_db/frontend/index.html")


@app.post("/insert", response_model=InsertResponse)
def insert_vector(request: InsertRequest) -> InsertResponse:
    vector_array = np.asarray(request.vector, dtype=float)
    insert(id=request.id, vector=vector_array, graph=graph, vectors=vectors)
    return InsertResponse(status="inserted", id=request.id)


@app.post("/search", response_model=SearchResponse)
def search_vector(request: SearchRequest) -> SearchResponse:
    query_array = np.asarray(request.vector, dtype=float)
    results = hnsw_search(
        query=query_array,
        k=request.k,
        graph=graph,
        vectors=vectors,
        ef_search=request.ef_search,
    )
    response_results = [SearchResult(id=result_id, score=float(score)) for result_id, score in results]
    return SearchResponse(results=response_results)


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    entry_point = graph.entry_point
    entry_level = graph.levels.get(entry_point) if entry_point is not None else None
    return StatsResponse(total_vectors=len(vectors), entry_point=entry_point, entry_level=entry_level)


@app.post("/upload-pdf", response_model=UploadPdfResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadPdfResponse:
    global chunk_counter

    raw_bytes = await file.read()
    reader = PdfReader(io.BytesIO(raw_bytes))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text() or ""

        # 1) Remove URLs.
        text = re.sub(r"http\S+", "", text)
        # 2) Remove emails.
        text = re.sub(r"\S+@\S+", "", text)
        # 3) Remove special symbols (keep word chars, whitespace, and basic punctuation).
        text = re.sub(r"[^\w\s\.\,\-\(\)\:]", " ", text)
        # 4) Remove multiple spaces.
        text = re.sub(r" +", " ", text)
        # 5) Remove multiple newlines.
        text = re.sub(r"\n+", "\n", text)
        # 6) Strip each line.
        text = "\n".join(line.strip() for line in text.split("\n"))

        text_parts.append(text)

    full_text = "\n".join(text_parts)

    chunks = smart_chunk(full_text)
    if not chunks:
        return UploadPdfResponse(status="indexed", total_chunks=0)

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(float)

    for i, chunk in enumerate(chunks):
        node_id = f"chunk_{chunk_counter + i}"
        vector = embeddings[i]
        insert(id=node_id, vector=vector, graph=graph, vectors=vectors)
        chunk_texts[node_id] = chunk

    chunk_counter += len(chunks)
    return UploadPdfResponse(status="indexed", total_chunks=len(chunks))


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    start = time.perf_counter()

    expanded_question = _expand_question(request.question)
    query_vec = embedder.encode(
        expanded_question,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(float)

    # Always retrieve a wider set internally, then rerank and return top-k.
    candidate_k = max(request.k * 3, request.k)
    ef_effective = max(50, int(request.ef_search))
    ranked = hnsw_search(
        query=query_vec,
        k=candidate_k,
        graph=graph,
        vectors=vectors,
        ef_search=ef_effective,
    )

    question_tokens = set(_tokenize_for_overlap(expanded_question))
    reranked = []

    for chunk_id, sem_score in ranked:
        cid = str(chunk_id)
        base_text = chunk_texts.get(cid, chunk_texts.get(chunk_id, ""))
        merged_text = _chunk_context_text(cid)
        text = merged_text if merged_text else base_text

        # Lexical overlap ratio as a small, query-intent-aware boost.
        chunk_tokens = set(_tokenize_for_overlap(text))
        overlap = (
            len(question_tokens.intersection(chunk_tokens)) / len(question_tokens)
            if question_tokens
            else 0.0
        )

        # Blend semantic similarity with lexical intent signal.
        # Keep semantic dominant; lexical helps reorder tied/nearby chunks.
        intent = _intent_boost(question_tokens, chunk_tokens)
        final_score = 0.78 * float(sem_score) + 0.22 * float(overlap) + intent
        reranked.append((cid, final_score, text))

    reranked.sort(key=lambda x: x[1], reverse=True)
    top = reranked[: request.k]

    results = [
        AskResultItem(chunk_id=cid, score=float(score), text=text)
        for cid, score, text in top
    ]
    # Filter low-score noise.
    results = [r for r in results if r.score > 0.2]

    # Build context from top chunks and call Groq LLM.
    context = "\n\n".join([r.text for r in results[:3]])
    question = request.question
    try:
        client = _get_groq_client()
        if client is None:
            raise RuntimeError("GROQ_API_KEY is not set")
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Answer the question using ONLY the context provided. "
                        "Be concise, clear and accurate. "
                        "If the answer is not in the context, say "
                        "'I could not find this in the document.'"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            max_tokens=300,
            temperature=0.1,
        )
        llm_answer = completion.choices[0].message.content or ""
    except Exception as e:
        llm_answer = f"LLM unavailable: {str(e)}"

    latency_ms = (time.perf_counter() - start) * 1000.0
    return AskResponse(
        question=request.question,
        llm_answer=llm_answer,
        results=results,
        latency_ms=latency_ms,
        total_chunks_searched=len(vectors),
    )


@app.get("/index-stats", response_model=IndexStatsResponse)
def index_stats() -> IndexStatsResponse:
    total_chunks = len(chunk_texts)
    entry_point = graph.entry_point
    if total_chunks == 0:
        health = "empty"
    elif entry_point is None:
        health = "degraded"
    else:
        health = "healthy"
    return IndexStatsResponse(total_chunks=total_chunks, entry_point=entry_point, index_health=health)

