import re
import io
import numpy as np
import pypdf
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────
# STEP 1 - PDF EXTRACTION
# ─────────────────────────────────────────

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text(extraction_mode="layout")
        except Exception:
            text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


# ─────────────────────────────────────────
# STEP 2 - TEXT CLEANING
# ─────────────────────────────────────────

def clean_text(text: str) -> str:
    # normalize bullet points
    text = re.sub(r"[•·▪▸●▶]+\s*", "\n• ", text)
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # remove emails
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    # keep only readable characters
    text = re.sub(r"[^\w\s\.\,\:\;\(\)\[\]\|•\n\%\+\-]", " ", text)
    # fix multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # fix multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [l.strip() for l in text.split("\n")]
    return "\n".join(lines)


# ─────────────────────────────────────────
# STEP 3 - UNIVERSAL SEMANTIC CHUNKING
# Works for any document type:
# research papers, books, reports, resumes
# ─────────────────────────────────────────

def is_section_header(line: str) -> bool:
    """
    Detect likely section headers across resumes/reports/papers/books.
    """
    section_keywords = [
        "education", "experience", "projects",
        "technical skills", "skills", "certifications",
        "summary", "internship", "research", "leadership",
        "coursework", "publications", "achievements",
        "awards", "volunteering", "objective", "profile",
        "about", "introduction", "background", "abstract",
        "conclusion", "methodology", "results", "references",
        "chapter", "section", "appendix", "overview",
        "contact", "personal",
    ]
    s = line.strip().lower().rstrip(":")
    return any(s == k for k in section_keywords)

def semantic_chunk(text: str, max_chunk_words: int = 80, overlap_lines: int = 1) -> list[str]:
    """
    Universal chunking strategy:

    1. Split into paragraphs first
    2. If paragraph is short: group with next
    3. If paragraph is long: split into sentences
    4. Add overlap between chunks for context continuity
    5. Each chunk = self-contained meaningful unit

    Works for ANY document type.
    """

    # split into paragraphs and force hard breaks at detected section headers
    lines = text.split("\n")
    rebuilt_lines = []
    for line in lines:
        if is_section_header(line):
            rebuilt_lines.append("\n")
            rebuilt_lines.append(line.strip())
            rebuilt_lines.append("\n")
        else:
            rebuilt_lines.append(line)
    text_with_sections = "\n".join(rebuilt_lines)
    paragraphs = re.split(r"\n{2,}", text_with_sections)

    # clean paragraphs
    clean_paras = []
    for para in paragraphs:
        # join lines within paragraph
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        if not lines:
            continue
        joined = " ".join(lines)
        if len(joined.split()) < 5:
            continue
        clean_paras.append(joined)

    # group paragraphs into chunks
    chunks = []
    buffer = []
    buffer_words = 0

    for para in clean_paras:
        para_words = len(para.split())

        # if single paragraph is too long: split by sentences
        if para_words > max_chunk_words:
            # flush buffer first
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buffer_words = 0

            # split long paragraph into sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_buffer = []
            sent_words = 0

            for sent in sentences:
                w = len(sent.split())
                if sent_words + w > max_chunk_words and sent_buffer:
                    chunks.append(" ".join(sent_buffer))
                    # overlap: keep last sentence
                    sent_buffer = sent_buffer[-overlap_lines:]
                    sent_words = sum(len(s.split()) for s in sent_buffer)
                sent_buffer.append(sent)
                sent_words += w

            if sent_buffer:
                chunks.append(" ".join(sent_buffer))

        # if adding this para exceeds limit: flush
        elif buffer_words + para_words > max_chunk_words:
            if buffer:
                chunks.append(" ".join(buffer))
                # overlap: keep last paragraph
                buffer = buffer[-overlap_lines:]
                buffer_words = sum(len(p.split()) for p in buffer)
            buffer.append(para)
            buffer_words += para_words

        else:
            buffer.append(para)
            buffer_words += para_words

    # flush remaining
    if buffer:
        chunks.append(" ".join(buffer))

    # deduplicate
    seen = set()
    unique = []
    for chunk in chunks:
        key = chunk[:80].strip().lower()
        if key not in seen and len(chunk.split()) > 5:
            seen.add(key)
            unique.append(chunk.strip())

    return unique


# ─────────────────────────────────────────
# STEP 4 - EMBEDDING
# ─────────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    embedding = embedder.encode(text, convert_to_numpy=True)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def embed_batch(texts: list[str]) -> np.ndarray:
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True,
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


# ─────────────────────────────────────────
# STEP 5 - QUERY EXPANSION
# Universal - works for any document type
# ─────────────────────────────────────────

def expand_query(question: str) -> str:
    """
    Expand query with related terms.
    Improves semantic matching for short queries.
    Only expands if query is short (under 6 words).
    """
    if len(question.split()) > 6:
        return question

    expansions = {
        # resume
        "skills": "technical skills programming languages frameworks tools",
        "experience": "work experience job role internship responsibilities",
        "projects": "projects built developed implemented created",
        "education": "education degree university college coursework",
        "certifications": "certifications courses training completed",
        # research papers
        "abstract": "abstract summary overview introduction",
        "conclusion": "conclusion findings results summary outcome",
        "methodology": "methodology method approach technique procedure",
        "results": "results findings outcomes performance evaluation",
        "dataset": "dataset data corpus benchmark evaluation",
        # general documents
        "summary": "summary overview key points main ideas",
        "definition": "definition meaning explanation concept description",
        "example": "example instance case illustration demonstration",
        "introduction": "introduction background context motivation",
        "author": "author writer name person created by",
        "who": "name person candidate author contact information identity",
        "name": "name person candidate author contact",
        "contact": "name email phone location address contact",
        "what is": "definition explanation description overview",
        "when": "date time year period duration",
        "where": "location place university company city country",
    }

    question_lower = question.lower()
    for keyword, expansion in expansions.items():
        if keyword in question_lower:
            return question + " " + expansion

    return question


# ─────────────────────────────────────────
# STEP 6 - RETRIEVAL
# ─────────────────────────────────────────

def retrieve(
    question: str,
    graph,
    vectors: dict,
    chunk_texts: dict,
    k: int = 5,
    ef_search: int = 100,
) -> list[dict]:
    """
    Full retrieval:
    1. Expand query
    2. Embed
    3. HNSW search
    4. Rank and return results
    """
    from vector_db.hnsw.search import hnsw_search

    if not vectors:
        return []

    # expand query
    expanded = expand_query(question)

    # embed
    query_vec = embed_text(expanded)

    # search more candidates then rerank
    k_internal = min(k * 3, len(vectors))
    ef = max(ef_search, k_internal * 2)

    raw_results = hnsw_search(query=query_vec, k=k_internal, graph=graph, vectors=vectors, ef_search=ef)

    # build results
    results = []
    for chunk_id, score in raw_results:
        text = chunk_texts.get(chunk_id, "")
        if not text:
            continue
        results.append(
            {
                "chunk_id": chunk_id,
                "score": round(float(score), 4),
                "score_pct": f"{round(float(score) * 100, 1)}%",
                "text": text,
            }
        )

    # sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:k]


# ─────────────────────────────────────────
# STEP 7 - BUILD INDEX FROM PDF
# ─────────────────────────────────────────

def build_index_from_pdf(file_bytes: bytes):
    """
    Complete pipeline:
    PDF bytes -> text -> clean -> chunk -> embed -> HNSW index

    Works for ANY PDF document.
    Returns: graph, vectors dict, chunk_texts dict
    """
    from vector_db.hnsw.graph import HNSWGraph
    from vector_db.hnsw.insert import insert

    # extract text
    raw_text = extract_pdf_text(file_bytes)

    # clean text
    clean = clean_text(raw_text)

    # extract first 5 lines as name/contact chunk
    first_lines = [l.strip() for l in clean.split("\n") if len(l.strip()) > 3][:5]
    name_chunk = "[Contact/Name] " + " | ".join(first_lines)

    # get all other chunks
    other_chunks = semantic_chunk(clean)

    # combine: name chunk first
    chunks = [name_chunk] + other_chunks

    # debug output
    print(f"\n{'='*60}")
    print(f"DOCUMENT INDEXED: {len(chunks)} chunks")
    print(f"{'='*60}")
    for i, ch in enumerate(chunks):
        print(f"CHUNK {i:02d} ({len(ch.split())} words): {ch[:100]}...")
    print(f"{'='*60}\n")

    # build HNSW index
    graph = HNSWGraph(M=32, ef_construction=400)
    vectors = {}
    chunk_texts = {}

    # batch embed all chunks
    embeddings = embed_batch(chunks)

    # insert into index
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"chunk_{i}"
        vectors[chunk_id] = emb
        chunk_texts[chunk_id] = chunk
        insert(id=chunk_id, vector=emb, graph=graph, vectors=vectors)

    return graph, vectors, chunk_texts

