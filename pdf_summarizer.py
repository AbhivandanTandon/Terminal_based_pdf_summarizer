# pdf_summarizer_onefile.py
# Single-file, resilient PDF summarizer for online interpreters.
# Features: PDF text extraction with page markers, cleaning, chunking,
# extractive (TF-IDF + TextRank) and optional abstractive (Transformers) map-reduce summarization.

import sys, os, re, math, io, textwrap, tempfile, subprocess
from typing import List, Tuple, Optional

# --------- Helper: safe optional import with best-effort pip ----------
def safe_import(pkg, import_as=None, pip_name=None):
    try:
        module = __import__(pkg) if import_as is None else __import__(import_as)
        return module
    except Exception:
        # In many online interpreters network pip may be blocked; we try once.
        name = pip_name or pkg
        try:
            print(f"[info] Attempting to install '{name}' ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            module = __import__(pkg) if import_as is None else __import__(import_as)
            print(f"[info] Installed '{name}'.")
            return module
        except Exception:
            print(f"[warn] Could not install or import '{name}'. Continuing without it.")
            return None

# Try useful packages
pdfplumber = safe_import("pdfplumber")
PyPDF2 = safe_import("PyPDF2")
sklearn = safe_import("sklearn")
networkx = safe_import("networkx")
numpy = safe_import("numpy")
transformers = safe_import("transformers")
torch = safe_import("torch")

# --------- Minimal utilities ----------
def normalize_text(t: str) -> str:
    t = t.replace("\r", "")
    # dehyphenate line breaks: word-\nword -> wordword
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # fix multiple spaces/newlines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # trim long leading/trailing space per line
    t = "\n".join(line.strip() for line in t.splitlines())
    return t.strip()

def sentence_split(text: str) -> List[str]:
    # simple regex splitter (no external tokenizer)
    sents = re.split(r'(?<=[.!?])\s+', text)
    # keep non-empty, avoid super-short fragments
    return [s.strip() for s in sents if len(s.strip()) > 1]

def word_count(text: str) -> int:
    return len(text.split())

# --------- Input handling ----------
def read_pdf_by_page(path: str) -> List[Tuple[int, str]]:
    """Return list of (page_no, text). Uses pdfplumber if available, else PyPDF2."""
    pages = []
    if pdfplumber is not None:
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    txt = page.extract_text() or ""
                    pages.append((i, txt))
            return pages
        except Exception as e:
            print(f"[warn] pdfplumber failed: {e}. Falling back to PyPDF2.")
    if PyPDF2 is not None:
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages, start=1):
                    try:
                        txt = page.extract_text() or ""
                    except Exception:
                        txt = ""
                    pages.append((i, txt))
            return pages
        except Exception as e:
            print(f"[error] PyPDF2 also failed: {e}")
    raise RuntimeError("No PDF backend available or failed to read PDF.")

# --------- Extractive scoring ----------
def tfidf_textrank_scores(sentences: List[str], doc_text: str) -> List[float]:
    """Return a normalized importance score for each sentence using TF-IDF doc-sim + TextRank sim graph.
       Falls back to frequency scoring if sklearn/networkx/numpy unavailable.
    """
    if not sentences:
        return []
    if sklearn and networkx and numpy:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        import networkx as nx
        try:
            # Build sentence vectors
            vec = TfidfVectorizer()
            X_sents = vec.fit_transform(sentences)
            # Similarity graph for TextRank (cosine via dot product since X is L2-normalized by default)
            sim = (X_sents * X_sents.T).A
            np.fill_diagonal(sim, 0.0)
            G = nx.from_numpy_array(sim)
            pr = nx.pagerank(G)
            tr_scores = np.array([pr.get(i, 0.0) for i in range(len(sentences))])

            # Doc relevance via TF-IDF similarity
            X_all = vec.fit_transform([doc_text] + sentences)
            doc_vec = X_all[0]
            sent_vecs = X_all[1:]
            doc_sim = (sent_vecs @ doc_vec.T).toarray().ravel()

            # Combine and normalize
            scores = tr_scores + doc_sim
            # normalize
            mn, mx = float(scores.min()), float(scores.max())
            norm = (scores - mn) / (mx - mn + 1e-9)
            return norm.tolist()
        except Exception as e:
            print(f"[warn] Advanced extractive scoring failed: {e}. Falling back to simple scoring.")

    # Fallback: simple term frequency score for each sentence
    freq = {}
    tokens = re.findall(r"\w+", doc_text.lower())
    for w in tokens:
        freq[w] = freq.get(w, 0) + 1
    scores = []
    for s in sentences:
        words = re.findall(r"\w+", s.lower())
        sc = sum(freq.get(w, 0) for w in words) / (len(words) + 1e-9)
        scores.append(sc)
    # normalize
    if not scores:
        return [0.0]*len(sentences)
    mn, mx = min(scores), max(scores)
    return [(x - mn) / (mx - mn + 1e-9) for x in scores]

# --------- Abstractive (optional) ----------
class AbstractiveSummarizer:
    def __init__(self):
        self.ok = False
        self.pipe = None
        if transformers is None:
            return
        try:
            # Smaller, faster CNN model. If the env blocks downloads, this will fail and we fall back.
            from transformers import pipeline
            # If GPU available, pipeline will discover it; otherwise CPU.
            self.pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            self.ok = True
            print("[info] Abstractive model ready (distilbart-cnn).")
        except Exception as e:
            print(f"[warn] Transformers available but model not loaded: {e}. Using extractive only.")

    def summarize(self, text: str, max_len=220, min_len=60) -> str:
        if not self.ok:
            return text  # no-op if not available
        try:
            out = self.pipe(text, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)[0]["summary_text"]
            return out
        except Exception as e:
            print(f"[warn] Abstractive inference failed: {e}. Returning input text.")
            return text

# --------- Chunking & map-reduce ----------
def chunk_long_text(text: str, target_words: int = 1200) -> List[str]:
    words = text.split()
    if not words:
        return []
    n = max(1, math.ceil(len(words) / target_words))
    size = math.ceil(len(words) / n)
    return [" ".join(words[i*size:(i+1)*size]) for i in range(n)]

def summarize_chunk_extractively(chunk_text: str, doc_text: str, k_min: int = 3) -> str:
    sents = sentence_split(chunk_text)
    if len(sents) <= k_min:
        return chunk_text
    scores = tfidf_textrank_scores(sents, doc_text)
    # pick top-k (~20% or at least k_min)
    k = max(k_min, max(1, len(sents)//5))
    top_idx = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)[:k]
    top_idx.sort()
    return " ".join(sents[i] for i in top_idx)

def map_reduce_summarize(parts: List[Tuple[str, str]], method: str = "auto", target_words: int = 300) -> str:
    """
    parts: list of (page_marker, text) where page_marker like '[p12]' appended to claims
    method: 'extractive' | 'abstractive' | 'auto'
    """
    abstr = AbstractiveSummarizer() if method in ("auto", "abstractive") else None

    # Map: summarize each chunk; append page markers
    partials = []
    combined_doc = " ".join(p for _, p in parts)
    for marker, text in parts:
        # Extractive shortlist
        shortlist = summarize_chunk_extractively(text, combined_doc)
        if abstr and abstr.ok and method in ("auto", "abstractive"):
            # prepend hint to keep citations
            shortlist_with_hint = f"Keep factual details and preserve references like {marker}.\n\n{shortlist} {marker}"
            mapped = abstr.summarize(shortlist_with_hint, max_len=220, min_len=60)
            # ensure marker present (if model dropped it)
            if marker not in mapped:
                mapped = mapped.rstrip() + f" {marker}"
        else:
            mapped = shortlist.rstrip() + f" {marker}"
        partials.append(mapped)

    # Reduce: join partials and compress once more if abstractive is available
    joined = "\n\n".join(partials)
    target_tokens = max(120, min(600, int(target_words * 1.6)))  # rough cap
    if abstr and abstr.ok and method in ("auto", "abstractive"):
        reducer_prompt = (
            "Create a single, concise, non-redundant summary (~"
            f"{target_words} words). Keep key entities and page markers like [p3].\n\n{joined}"
        )
        reduced = abstr.summarize(reducer_prompt, max_len=target_tokens, min_len=max(60, target_words//2))
        # if all markers dropped, append a minimal citation block
        if not re.search(r"\[p\d+\]", reduced):
            # get a few unique markers from partials
            markers = sorted(set(m for m, _ in parts))[:5]
            reduced = reduced.rstrip() + "\n\nSources: " + " ".join(markers)
        return reduced
    else:
        # Extractive-only reduction: take top sentences again from the concatenation
        reduced = summarize_chunk_extractively(joined, joined, k_min=6)
        return reduced

# --------- End-to-end driver ----------
def summarize_document(
    pdf_path: Optional[str] = None,
    pasted_text: Optional[str] = None,
    method: str = "auto",           # 'auto' | 'extractive' | 'abstractive'
    chunk_words: int = 1200,
    target_words: int = 300
) -> str:
    assert pdf_path or pasted_text, "Provide either pdf_path or pasted_text."
    page_texts: List[Tuple[int, str]] = []

    if pdf_path:
        page_texts = read_pdf_by_page(pdf_path)
    else:
        # Treat the entire pasted text as one “page”
        page_texts = [(1, pasted_text or "")]

    # Clean and prepare parts with page markers
    parts: List[Tuple[str, str]] = []
    for pno, txt in page_texts:
        clean = normalize_text(txt)
        # chunk per page if too long
        subchunks = chunk_long_text(clean, target_words=chunk_words)
        if not subchunks:
            continue
        if len(subchunks) == 1:
            parts.append((f"[p{pno}]", subchunks[0]))
        else:
            for j, sub in enumerate(subchunks, start=1):
                parts.append((f"[p{pno}]", sub))

    if not parts:
        return "No extractable text found."

    summary = map_reduce_summarize(parts, method=method, target_words=target_words)
    # light tidy
    summary = re.sub(r"\n{3,}", "\n\n", summary).strip()
    return summary

# --------- Simple interactive CLI (works in online REPLs) ----------
def _prompt_yesno(q: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    ans = input(f"{q} [{d}]: ").strip().lower()
    if ans == "" and default: return True
    if ans in ("y", "yes"): return True
    return False

def main():
    print("=== PDF Summarizer (Single-File) ===")
    use_pdf = _prompt_yesno("Do you want to summarize a PDF file?", True)
    pasted = None
    path = None

    if use_pdf:
        path = input("Enter PDF path (uploaded or accessible in the environment): ").strip().strip('"').strip("'")
        if not os.path.exists(path):
            print("[error] File not found. You can paste text instead.")
            use_pdf = False

    if not use_pdf:
        print("\nPaste your text below. Finish with an empty line:")
        buf = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "":
                break
            buf.append(line)
        pasted = "\n".join(buf)

    method = input("\nChoose method: 'auto' (try abstractive) | 'extractive' | 'abstractive' [auto]: ").strip().lower() or "auto"
    if method not in ("auto", "extractive", "abstractive"):
        method = "auto"

    try:
        chunk_words = int(input("Chunk size (words per chunk) [1200]: ").strip() or "1200")
    except Exception:
        chunk_words = 1200
    try:
        target_words = int(input("Target summary length (words) [300]: ").strip() or "300")
    except Exception:
        target_words = 300

    print("\n[info] Summarizing...\n")
    try:
        out = summarize_document(
            pdf_path=path if use_pdf else None,
            pasted_text=pasted if not use_pdf else None,
            method=method,
            chunk_words=chunk_words,
            target_words=target_words
        )
        print("\n===== SUMMARY =====\n")
        print(textwrap.fill(out, width=100))
    except Exception as e:
        print(f"[error] {e}")

if __name__ == "__main__":
    main()
