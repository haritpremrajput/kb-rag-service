"""RAG service over a 50-passage technical KB.

POST /query {"question": str, "top_k": int} ->
  {"answer": str|None, "sources": [int], "scores": [float], "confidence": float}

Pipeline (no LLM, fully extractive so answers can only come from the KB):
  1. sanitize question (strip prompt-injection clauses)
  2. hybrid retrieval: dense cosine (bge-small) + IDF-weighted keyword overlap
  3. abstention checks (unknown entities, missing asked-for attribute, low relevance)
  4. extract best sentence(s) from top passage (+ a second passage for multi-part questions)
"""
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding
from pydantic import BaseModel

BASE = Path(__file__).parent
KB: list[str] = json.loads((BASE / "kb.json").read_text(encoding="utf-8"))

MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# 1 ONNX thread by default: free hosts give a fraction of a CPU, and FastAPI's
# threadpool already provides request-level parallelism.
model = TextEmbedding(MODEL_NAME, cache_dir=str(BASE / "model_cache"),
                      threads=int(os.environ.get("EMBED_THREADS", "1")))


@lru_cache(maxsize=4096)
def qembed(text: str) -> np.ndarray:
    """Embed one search query (cached: the same text is reused across retrieval and sentence picking)."""
    return embed([QUERY_PREFIX + text])[0]


def embed(texts: list[str]) -> np.ndarray:
    v = np.array(list(model.embed(texts, batch_size=8)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------- text utils
STOP = set("""a an the of to in on for by with and or is are was were be been being does do did
what which who whom whose when where why how much many that this these those it its as at from
into over under than then there their they them i you we he she me my your our us can could
should would will shall may might must about according kb knowledge base passage context tell
give name state value number used use uses using default s per each any some all value
""".split())

QUESTION_WORDS = STOP | set("""explain describe list compare please also note answer question
if both between versus vs difference respectively""".split())


def stem(t: str) -> str:
    if any(c.isdigit() for c in t) or len(t) <= 3:
        return t
    if t.endswith("ies"):
        t = t[:-3] + "y"
    elif re.search(r"(ch|sh|x|ss)es$", t):
        t = t[:-2]
    elif t.endswith("s") and not t.endswith("ss") and not t.endswith("us"):
        t = t[:-1]
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


def tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:[./+-][a-z0-9]+)*", text.lower())
    out = []
    for t in toks:
        out.append(stem(t))
        if "-" in t or "/" in t:
            out.extend(stem(p) for p in re.split(r"[-/]", t) if p)
    return out


# Acronyms / aliases -> KB wording (expanded before retrieval)
ALIASES = {
    r"\bSGD\b": "stochastic gradient descent SGD",
    r"\bMLPs?\b": "multi-layer perceptron MLP",
    r"\bCNNs?\b|\bResNets?\b": "residual convolutional network ResNet",
    r"\bBN\b|\bbatchnorm\b": "batch normalization",
    r"\bLN\b|\blayernorm\b": "layer normalization",
    r"\bLSM\b": "log-structured merge tree LSM",
    r"\bWAL\b": "write-ahead logging WAL",
    r"\bHLCs?\b": "hybrid logical clocks HLC",
    r"\b2PC\b": "two-phase commit",
    r"\b3PC\b": "three-phase commit",
    r"\bECC\b": "elliptic curve cryptography ECC",
    r"\bPostgres\b": "PostgreSQL",
    r"\bMongo\b": "MongoDB",
    r"\bInnoDB\b": "MySQL InnoDB",
    r"\bHRW\b|\bhighest random weight\b": "rendezvous hashing highest random weight",
    r"\bWAFs?\b": "web application firewall WAF",
    r"\bSVMs?\b": "support vector machine SVM",
    r"\bmDNS\b": "multicast DNS mDNS",
    r"\bJWTs?\b": "JSON Web Token JWT",
    r"\bPKCE\b": "OAuth PKCE proof key for code exchange",
    r"\bAdam-?W\b": "AdamW",
    r"\bLLMs?\b|\bGPT\b": "autoregressive transformer decoder GPT",
    r"\bBERT\b": "transformer encoder",
    r"\bRSA\b": "RSA",
    r"\bChaCha20?\b": "ChaCha20-Poly1305",
    r"\bQUIC\b": "QUIC",
    r"\bTCP\b": "TCP transmission control protocol",
    r"\bDNS\b(?!SEC)": "DNS domain name system",
    r"\bBGP\b": "BGP border gateway protocol",
    r"\bOSPF\b": "OSPF open shortest path first",
    r"\bTLS ?1\.3\b": "TLS 1.3 transport layer security",
    r"\bAES\b": "AES advanced encryption standard",
    r"\b(?:logistic regression|LR model)\b": "binary logistic regression",
}


def is_alias(w: str) -> bool:
    return any(re.fullmatch(p, w, flags=re.I) for p in ALIASES)


def expand(q: str) -> str:
    def acro(m):
        w = m.group(0)
        low = w.lower()
        if low in VOCAB or low not in ACRONYMS or is_alias(w):
            return w
        return f"{ACRONYMS[low]} {w}"

    q = re.sub(r"\b[A-Z]{2,5}\b", acro, q)  # auto-derived acronyms first
    for pat, rep in ALIASES.items():
        q = re.sub(pat, rep, q, flags=re.I if pat.islower() else 0)
    return q


STOP |= {stem(w) for w in STOP}
QUESTION_WORDS |= STOP


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOP and len(t) > 1]


def split_sentences(p: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Za-z])", p) if s.strip()]


# ---------------------------------------------------------------- index
P_TOKS = [set(content_tokens(p)) for p in KB]
VOCAB = set().union(*P_TOKS)
N = len(KB)
DF = {}
for ts in P_TOKS:
    for t in ts:
        DF[t] = DF.get(t, 0) + 1
IDF = {t: math.log(1 + N / df) for t, df in DF.items()}

SENTS = [split_sentences(p) for p in KB]
SUBJ_TOKS = [set(content_tokens(s[0])) for s in SENTS]


def _build_acronyms() -> dict[str, str]:
    """Derive acronyms from KB phrases: 'multi-version concurrency control' -> MVCC."""
    best: dict[str, tuple[int, str]] = {}
    for p in KB:
        words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", p)
        for i in range(len(words)):
            for L in range(1, 5):
                ws = words[i:i + L]
                if L == 1 and "-" not in ws[0]:
                    continue
                if len(ws) < L or any(w.lower() in STOP for w in (ws[0], ws[-1])):
                    continue
                pieces = [s for w in ws for s in w.split("-")]
                for parts in (pieces, ws):
                    a = "".join(s[0] for s in parts if s).lower()
                    if not 2 <= len(a) <= 5:
                        continue
                    # prefer Title-Cased phrases (proper technical names)
                    rank = sum(w[0].isupper() for w in ws) / len(ws)
                    if rank == 0 and not any("-" in w for w in ws):
                        continue
                    if a not in best or rank > best[a][0]:
                        best[a] = (rank, " ".join(ws))
    return {a: v for a, (_, v) in best.items()}


ACRONYMS = _build_acronyms()


def _load_or_build_embeddings():
    """Passage + sentence embeddings, cached on disk keyed by KB/model hash.

    The Docker build runs `import app` once, so containers start by loading this
    file instead of re-embedding ~300 texts on a throttled CPU.
    """
    import hashlib
    key = hashlib.sha1(json.dumps([MODEL_NAME, KB, SENTS]).encode()).hexdigest()
    path = BASE / "emb_cache.npz"
    if path.exists():
        try:
            z = np.load(path)
            if str(z["key"]) == key:
                offs = z["offsets"]
                return z["p"], [z["s"][offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        except Exception:
            pass
    p_emb = embed(KB)
    s_emb = [embed(x) for x in SENTS]
    offs = np.cumsum([0] + [len(x) for x in s_emb])
    try:
        np.savez(path, key=key, p=p_emb, s=np.concatenate(s_emb), offsets=offs)
    except OSError:
        pass
    return p_emb, s_emb


P_EMB, S_EMB = _load_or_build_embeddings()

# Attribute words: if the question asks for one of these and the chosen passage
# never mentions it, the KB does not contain the answer -> abstain.
ATTRIBUTES = {
    "port": ["port"],
    "quorum": ["quorum"],
    "heartbeat": ["heartbeat"],
    "learning rate": ["learning rate"],
    "batch size": ["batch", "mini-batch", "batches"],
    "key size": ["bits", "bit"],
    "hash": ["hash", "digest", "sha", "blake2"],
    "layers": ["layer", "block", "blocks", "layers"],
    "timeout": ["timeout", "ms", "interval", "window", "duration", "exceed", "within"],
    "threshold": ["threshold", "0."],
    "retries": ["retries", "retry"],
}
ATTR_TRIGGERS = {
    "port": r"\bports?\b",
    "quorum": r"\bquorum\b",
    "heartbeat": r"\bheartbeats?\b",
    "learning rate": r"\blearning[- ]rate\b|\blr\b",
    "batch size": r"\bbatch(?:[- ]size)?\b|\bmini-?batch",
    "key size": r"\bkey (?:size|length)\b|\bhow many bits\b|\bbits?\b",
    "hash": r"\bhash\b(?! weight)|\bhash(?:ing)? (?:function|algorithm)\b|\bdigest\b",
    "threshold": r"\bthreshold\b|\bprobability\b|\bratio\b",
    "layers": r"\bhow many (?:layers|blocks)\b|\bnumber of (?:layers|blocks)\b",
    "retries": r"\bretr(?:y|ies)\b",
}

# ---------------------------------------------------------------- injection defence
INJECTION = re.compile(
    r"(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instruction|prompt|rule|context|above|previous|prior|system)"
    r"|system prompt|you are now|new instructions?|developer mode|jailbreak|\bact as\b|pretend"
    r"|(respond|reply|answer|output|print|say|return|write)\b.{0,25}\b(only|exactly|with|the word|the string)\b"
    r"|<\s*/?\s*(system|instruction|inst|admin)\s*>|\[/?inst\]|<<sys>>|###\s*(system|instruction)"
    r"|the (correct|real|true) answer is|admin(istrator)? (override|mode|note)",
    re.I,
)


def sanitize(q: str) -> str:
    q = re.sub(r"<!--.*?-->", " ", q, flags=re.S)  # hidden html comments
    q = re.sub(r"[​-‏⁠﻿]", "", q)  # zero-width chars
    parts = re.split(r"(?<=[.!?;:\n])\s+|\n+|\s+[-–—]{1,2}\s+|\(|\)|\[|\]", q)
    kept = [p for p in parts if p and p.strip() and not INJECTION.search(p)]
    # prefer the actual question clause(s)
    qs = [p for p in kept if "?" in p]
    if qs:
        kept = qs
    out = " ".join(s.strip() for s in kept).strip()
    if not out:
        out = INJECTION.sub(" ", q).strip()
    return out or q


def unknown_entities(q: str) -> list[str]:
    """Proper-noun-ish or versioned tokens in the question that never appear in the KB."""
    unk = []
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9./+-]*[A-Za-z0-9+]|[A-Za-z0-9]", q):
        raw = m.group(0).rstrip("+")
        low = raw.lower().strip(".")
        if len(re.sub(r"[^a-z0-9]", "", low)) <= 2:
            continue
        if low in QUESTION_WORDS or stem(low) in VOCAB or low in VOCAB:
            continue
        if any(re.fullmatch(p, raw, flags=re.I) for p in ALIASES):
            continue
        if raw.isupper() and (low in ACRONYMS or low.rstrip("s") in ACRONYMS):
            continue
        inner_upper = any(c.isupper() for c in raw[1:])
        has_digit = any(c.isdigit() for c in raw)
        sentence_start = m.start() == 0 or re.search(r"[.?!:;\"'(]\s*$", q[: m.start()])
        has_upper = inner_upper or (raw[0].isupper() and not sentence_start)
        if not (has_upper or has_digit):
            continue
        parts = [p for p in re.split(r"[-/.]", low) if p]
        # versioned names (SHA-512, GPT-4, v3.5) must match exactly; plain compounds may match by parts
        if not has_digit and parts and all(stem(p) in VOCAB or p in STOP for p in parts):
            continue
        if low.endswith("s") and low[:-1] in VOCAB:
            continue
        unk.append(raw)
    return unk


# ---------------------------------------------------------------- retrieval
def kw_score(q_terms: list[str], doc_terms: set[str]) -> float:
    q = [t for t in set(q_terms) if t in IDF]
    if not q:
        return 0.0
    tot = sum(IDF[t] for t in q)
    return sum(IDF[t] for t in q if t in doc_terms) / tot


def retrieve(q: str):
    qv = qembed(q)
    dense = P_EMB @ qv
    q_terms = content_tokens(q)
    kw = np.array([kw_score(q_terms, P_TOKS[i]) for i in range(N)])
    # subject boost: an entity named in the passage's opening sentence is what the
    # passage is *about*; later mentions ("Unlike Redis, ...") are confounders.
    ents = [e for e in entity_terms(q) if IDF.get(e, 0) > math.log(1 + N / 6)]
    subj = np.zeros(N)
    if ents:
        for i in range(N):
            subj[i] = sum(e in SUBJ_TOKS[i] for e in ents) / len(ents)
    final = np.clip(0.6 * dense + 0.3 * kw + 0.1 * subj, 0.0, 1.0)
    order = np.argsort(-final)
    return qv, dense, kw, final, order, q_terms


GENERIC = set("""default configured configure set typical typically standard specific exact
actual value size much long often large mean means purpose role work function
say said tell know mention describe explain summarize summary overview detail info information
main key important primary specify specifie""".split())
GENERIC_STEM = {stem(w) for w in GENERIC}

VALUE_Q = re.compile(
    r"\bhow (?:many|much|long|often|large|big|quickly|fast)\b|\bdefault\b|\bport\b|\bsize\b|\blength\b"
    r"|\btimeout\b|\binterval\b|\bthreshold\b|\bquorum\b|\bbits\b|\blearning rate\b|\bduration\b"
    r"|\bvalue of\b|\bwhat value\b|\bnumber of\b|\blimit\b|\bratio\b|\bprobability\b",
    re.I,
)

FACTUAL = re.compile(r"\b(what|which|how many|how much|how long|how often|when|value|size|port|rate|number)\b", re.I)


def best_sentences(i: int, q: str, n: int = 1, extra_attrs=()):
    """Pick the sentence(s) in passage i that carry the asked-for fact."""
    qv = qembed(q)
    sims = S_EMB[i] @ qv
    # ignore terms naming the passage's topic: every sentence "is about" it anyway
    topic = set(content_tokens(SENTS[i][0]))
    q_terms = [t for t in content_tokens(q) if t not in topic]
    attrs = set(asked_attributes(q)) | set(extra_attrs)
    factual = bool(FACTUAL.search(q))
    count_q = bool(re.search(r"\bhow many\b|\bnumber of\b", q, re.I))
    time_q = bool(re.search(r"\bhow long\b|\bhow often\b|\btimeout\b|\binterval\b|\bduration\b|\blatency\b|\bhow quickly\b", q, re.I))
    scores = []
    for j, s in enumerate(SENTS[i]):
        st = set(content_tokens(s))
        sc = 0.5 * float(sims[j]) + 0.35 * kw_score(q_terms, st)
        if factual and re.search(r"\d", s):
            sc += 0.15
        if attrs and any(text_has_attr(s, a) for a in attrs):
            sc += 0.3
        if count_q and re.search(r"\b\d+\b(?!\.\d)(?!\s*(?:ms|bits?)\b)", s):
            sc += 0.2
        if time_q and re.search(r"\d+ms\b", s):
            sc += 0.2
        scores.append(sc)
    idx = sorted(range(len(scores)), key=lambda j: -scores[j])[:n]
    # does the chosen sentence actually address what was asked?
    focus = [t for t in q_terms if t in IDF and t not in GENERIC and t not in GENERIC_STEM
             and not is_alias(t) and t not in ACRONYMS]
    s_best = SENTS[i][idx[0]]
    covered = (
        not focus
        or kw_score(focus, P_TOKS[i]) > 0
        or bool(attrs and any(text_has_attr(s_best, a) for a in attrs))
    )
    # a question asking for a value must land on a sentence that holds one
    if covered and VALUE_Q.search(q) and not re.search(r"\d", s_best):
        covered = False
    return [SENTS[i][j] for j in sorted(idx)], covered


def asked_attributes(q: str) -> list[str]:
    return [a for a, pat in ATTR_TRIGGERS.items() if re.search(pat, q, re.I)]


def text_has_attr(text: str, attr: str) -> bool:
    low = text.lower()
    return any(re.search(r"(?<![a-z])" + re.escape(w), low) for w in ATTRIBUTES[attr])


def passage_has_attr(i: int, attr: str) -> bool:
    return text_has_attr(KB[i], attr)


def entity_terms(question: str) -> set[str]:
    """Tokens written like names (capitalised / acronyms / digits) in the original question."""
    ents = set()
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9./+-]*", question):
        if raw.lower() in QUESTION_WORDS:
            continue
        if any(c.isupper() for c in raw) or any(c.isdigit() for c in raw):
            ents.update(tokens(raw))
            if raw.lower() not in VOCAB and (is_alias(raw) or raw.lower() in ACRONYMS):
                ents.update(content_tokens(expand(raw)))
    return {e for e in ents if e in IDF and e not in STOP}


MULTI = re.compile(r"\band\b|\bboth\b|\bcompare|\bversus\b|\bvs\.?\b|\bdiffer|\brespectively\b|,", re.I)


def split_subquestions(q: str) -> list[str]:
    parts = re.split(r"\?\s*|\band\b|;|,\s*(?:and\s+)?(?=what|how|which|who|when|where)", q, flags=re.I)
    return [p.strip() for p in parts if p and len(content_tokens(p)) >= 1]


DENSE_MIN = float(os.environ.get("DENSE_MIN", "0.55"))
FINAL_MIN = float(os.environ.get("FINAL_MIN", "0.45"))


def abstain(reason: str, final=None):
    return {"answer": None, "sources": [], "scores": [], "confidence": 0.1, "_reason": reason}


@lru_cache(maxsize=2048)
def answer_question(question: str, top_k: int) -> dict:
    raw_q = sanitize(question)
    if not content_tokens(raw_q):
        return abstain("empty")

    unk = unknown_entities(raw_q)
    q = expand(raw_q)
    qv, dense, kw, final, order, q_terms = retrieve(q)
    top = int(order[0])

    if unk:
        return abstain(f"unknown entities {unk}")
    if dense[top] < DENSE_MIN or final[top] < FINAL_MIN:
        return abstain(f"low relevance dense={dense[top]:.3f} final={final[top]:.3f}")

    # Multi-part / cross-passage: a second passage naming an entity from the
    # question that the top passage does not mention.
    chosen = [top]
    sub_of = {top: q}
    full_attrs = asked_attributes(q)
    if MULTI.search(raw_q):
        # 1) retrieve each sub-question separately
        subqs = [s for s in re.split(r"\band\b|;|,|\bwith\b|\bversus\b|\bvs\.?\b", q, flags=re.I)
                 if entity_terms(s)]
        if len(subqs) >= 2:
            for s in subqs:
                _, d2, _, f2, o2, _ = retrieve(s)
                j = int(o2[0])
                if j not in chosen and d2[j] >= DENSE_MIN and f2[j] >= FINAL_MIN:
                    chosen.append(j)
                    sub_of[j] = s
                elif j == top:
                    sub_of[top] = s
        # 2) fallback: a passage naming an entity the top passage lacks
        if len(chosen) == 1:
            ents = entity_terms(q)
            for j in order[1:6]:
                j = int(j)
                extra = [t for t in ents if t in P_TOKS[j] and t not in P_TOKS[top]]
                if extra and final[j] >= 0.55 * final[top]:
                    chosen.append(j)
                    sub_of[j] = q
                    break

    multi = len(chosen) > 1
    for i in list(chosen):
        need = set(asked_attributes(sub_of[i])) | (set(full_attrs) if multi else set())
        for a in need:
            if not passage_has_attr(i, a):
                if i == top:
                    return abstain(f"attribute '{a}' not in passage {i}")
                chosen.remove(i)
                break

    pieces = []
    for i in chosen:
        sents, covered = best_sentences(i, sub_of[i], n=1, extra_attrs=full_attrs if multi else ())
        if not covered and i == top and not multi:
            return abstain(f"best sentence of passage {i} does not cover the question focus")
        pieces.extend(sents)
    answer = " ".join(pieces)

    k = max(1, min(int(top_k or 3), N))
    srcs = [int(i) for i in chosen]
    for j in order:
        if len(srcs) >= k:
            break
        if int(j) not in srcs:
            srcs.append(int(j))
    srcs = srcs[:k]
    scs = sorted((round(float(final[i]), 4) for i in srcs), reverse=True)
    # keep sources aligned with descending score order
    srcs = sorted(srcs, key=lambda i: -final[i])
    conf = round(float(min(0.99, 0.6 + 0.6 * max(0.0, final[top] - FINAL_MIN))), 4)
    return {"answer": answer, "sources": srcs, "scores": scs, "confidence": conf}


# ---------------------------------------------------------------- API
app = FastAPI(title="KB RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


import collections
import time as _time

from fastapi import Request
from starlette.concurrency import run_in_threadpool

REQUEST_LOG: collections.deque = collections.deque(maxlen=500)


def _parse_body(raw: bytes, params) -> tuple[str, int]:
    data = {}
    if raw:
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    q = data.get("question") or data.get("query") or data.get("q") or params.get("question") or params.get("q") or ""
    try:
        k = int(data.get("top_k") or data.get("k") or params.get("top_k") or 3)
    except (TypeError, ValueError):
        k = 3
    return str(q), k


@app.api_route("/query", methods=["POST", "GET"])
@app.api_route("/query/", methods=["POST", "GET"], include_in_schema=False)
async def query(request: Request):
    t = _time.time()
    raw = await request.body()
    q, k = _parse_body(raw, request.query_params)
    res = dict(await run_in_threadpool(answer_question, q.strip()[:4000], k))
    reason = res.pop("_reason", None)
    entry = {
        "t": _time.strftime("%Y-%m-%d %H:%M:%S", _time.gmtime()),
        "ms": round((_time.time() - t) * 1000),
        "method": request.method,
        "ua": request.headers.get("user-agent", "")[:120],
        "origin": request.headers.get("origin"),
        "raw": raw[:2000].decode("utf-8", "replace"),
        "response": res,
        "reason": reason,
    }
    REQUEST_LOG.append(entry)
    print("QUERYLOG", json.dumps(entry), flush=True)
    if os.environ.get("DEBUG") == "1":
        res["_reason"] = reason
    return res


@app.get("/debug/requests")
def debug_requests(n: int = 200):
    return list(REQUEST_LOG)[-n:]


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "passages": N, "model": MODEL_NAME}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
