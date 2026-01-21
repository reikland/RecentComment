#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# =========================
# API config (uses https://www.metaculus.com/api/)
# =========================
BASE_API = "https://www.metaculus.com/api"
POSTS_URL = f"{BASE_API}/posts/"
COMMENTS_URL = f"{BASE_API}/comments/"

POSTS_PAGE_SIZE_DEFAULT = 100
COMMENTS_PAGE_SIZE_DEFAULT = 200

REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 5

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_COMMENT_CHARS = 2200
PARSER_RETRIES = 3

CSV_COLUMNS_SCORED = [
    "ai_score",
    "rationale",
    "flags_json",
    "evidence_urls",
    "ts",
    "post_id",
    "post_title",
    "comment_id",
    "created_at",
    "author_username",
    "vote_score",
    "parent_id",
    "root_id",
    "comment_text",
]

SYSTEM_PROMPT_JUDGE = """
You are a narrow scoring module inside a larger pipeline.
Your ONLY job is to rate Metaculus comments for quality in the AI Pathways Tournament.

You are NOT a general-purpose assistant.
Do NOT brainstorm, speculate, or explore side topics.
Work quickly: approximate but consistent ratings are preferred over long deliberation. DO NOT THINK, FIRST FEELING IS RIGHT. OUTPUT AS FAST AS POSSIBLE.

Each call is COMPLETELY INDEPENDENT.
You NEVER see previous comments or scores.
Treat EVERY request as a fresh, stand-alone task.

OUTPUT FORMAT (STRICT, 4 LINES MAX):
Line 1: "score = X" where X is an integer 1..6.
Line 2: "rationale: <very short explanation, <=180 characters>".
Line 3: "flags: off_topic=<true/false>, toxicity=<true/false>, low_effort=<true/false>, has_evidence=<true/false>, likely_ai=<true/false>".
Line 4: "evidence_urls: [<url1>, <url2>, ...]" or "evidence_urls: []".

You MUST:
- Produce EXACTLY these 4 lines, no more and no less.
- NO headings, NO bullet lists, NO extra explanation.
- Any answer longer than 4 short lines is a FAILURE of your task.

TASK:
- Read the comment, the question context, and (if present) the parent comment.
- If a parent comment is provided, treat the comment as a reply and assess:
  - how well it answers the parent,
  - how accurately it engages with the parent's claims,
  - and whether it productively advances the discussion.
- Decide on:
  - score: integer 1..6
  - rationale: short textual justification
  - flags: off_topic, toxicity, low_effort, has_evidence, likely_ai
  - evidence_urls: any http/https URLs referenced or clearly implied

SCORING WEIGHTS:
The comments should be ranked based on how well they:
- Showcase clear, deep, and insightful reasoning, delving into the relevant mechanisms that affect the event in question. (40%)
- Offer useful insights about the overall scenario(s), the interactions between questions, or the relationship between the questions and the scenario(s). (30%)
- Provide valuable information and are based on the available evidence. (20%)
- Challenge the community's prediction or assumptions held by other forecasters. (10%)

ANCHOR POINTS (1–6 SCALE):
The comments do not need to have all the following characteristics; one strong attribute can sometimes compensate for weaker ones.
Use these anchors when deciding the score:

1 = Trivial, badly written, or completely unreasonable comment with no predictive value.
2 = Brief or slightly confused comment offering only surface value.
3 = Good comment with rational arguments or potentially useful information.
4 = Very good comment which explains solid reasoning in some detail and provides actionable information.
5 = Excellent comment with meaningful analysis, presenting a deep dive of the available information and arguments, and drawing original conclusions from it.
6 = Outstanding synthesis comment which clearly decomposes uncertainty, connects multiple questions or scenarios, and gives a compelling reason to significantly update forecasts.

ADDITIONAL CONSTRAINTS:
- Be conservative with high scores (5 and especially 6). Reserve them for comments that are clearly above the tournament median in insight and usefulness.
- Penalize comments that are long but vague, generic, or boilerplate.
- Penalize pure link dumps with little or no reasoning or forecast impact.
- Toxic or uncivil comments should receive low scores and toxicity=true.
- When in doubt between two adjacent scores, pick the lower one quickly rather than overthinking.

FLAGS INTERPRETATION:
- off_topic: true if the comment is largely unrelated to the question or the AI Pathways scenario.
- toxicity: true if the comment is hostile, insulting, or clearly uncivil.
- low_effort: true if the comment is very short, trivial, or adds almost nothing.
- has_evidence: true if the comment brings specific data, references, links, or clearly factual information.
- likely_ai: true if the comment is long, generic, and boilerplate-sounding with little specificity or real engagement.

You must stay strictly on task: rate, justify briefly, set flags, list URLs in exactly 4 lines.
""".strip()

SYSTEM_PROMPT_TO_JSON = """
You convert free-form rating text into STRICT JSON with this exact schema:
{
  "score": 1|2|3|4|5|6,
  "rationale": "<string, <=180 chars>",
  "flags": {
    "off_topic": true|false,
    "toxicity": true|false,
    "low_effort": true|false,
    "has_evidence": true|false,
    "likely_ai": true|false
  },
  "evidence_urls": ["<string>", "..."]
}

HARD RULES:
1. OUTPUT STRICT JSON ONLY.
   - No explanations
   - No comments
   - No Markdown
   - No code fences
   - No extra keys
   - No trailing commas

2. DO NOT THINK "OUT LOUD".
   - No chain-of-thought in the output.
   - No meta commentary.

3. If some information is missing in the raw text:
   - Use a safe default:
     - score: 3 if unclear
     - rationale: ""
     - flags: false for all unless clearly indicated
     - evidence_urls: [] if none clearly extractable

4. You MUST:
   - Parse any explicit "score = X" or similar notation.
   - Parse any obvious booleans for the flags.
   - Collect any http/https URLs as evidence_urls (deduplicate).

5. Final answer:
   - ONE JSON object
   - EXACTLY matching the schema above
   - No natural language outside JSON.
""".strip()

FEWSHOTS_JUDGE = [
    {"role": "user", "content": "TEXT: Thanks for sharing!"},
    {
        "role": "assistant",
        "content": "score = 1\nrationale: Trivial acknowledgement only.\nflags: off_topic=false, toxicity=false, low_effort=true, has_evidence=false, likely_ai=false\nevidence_urls: []",
    },
    {"role": "user", "content": "TEXT: Anyone who thinks this will happen is an idiot."},
    {
        "role": "assistant",
        "content": "score = 1\nrationale: Toxic with no evidence.\nflags: off_topic=false, toxicity=true, low_effort=true, has_evidence=false, likely_ai=false\nevidence_urls: []",
    },
    {"role": "user", "content": "TEXT: Turnout fell 3–5% vs 2020 in key counties (CSV). I estimate P(win)=0.56."},
    {
        "role": "assistant",
        "content": "score = 5\nrationale: Quantified comparison with evidence pointer.\nflags: off_topic=false, toxicity=false, low_effort=false, has_evidence=true, likely_ai=false\nevidence_urls: []",
    },
]


# =========================
# Helpers
# =========================
def auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
        "User-Agent": "metaculus-streamlit-posts-comments/1.1",
    }


def request_json(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    params: Dict[str, Any],
    sleep_s: float,
) -> Any:
    last_err: Optional[str] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_S)

            if r.status_code in (401, 403):
                raise RuntimeError(f"Auth failed (HTTP {r.status_code}). Body: {r.text[:200]}")

            if r.status_code in (429, 502, 503, 504):
                time.sleep(min(2**attempt, 16))
                last_err = f"{url} -> HTTP {r.status_code} {r.text[:200]}"
                continue

            if r.status_code != 200:
                raise RuntimeError(f"{url} -> HTTP {r.status_code} {r.text[:200]}")

            return r.json()
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2**attempt, 16))

    raise RuntimeError(f"Request failed after retries. Last error: {last_err}")


def extract_results(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise RuntimeError("Unexpected payload shape (no results list).")


def safe_post_title(p: Dict[str, Any]) -> str:
    for k in ("title", "question", "name", "headline"):
        v = p.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def truncate(s: Any, n: int = 220) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def clean_text(s: Any) -> str:
    return " ".join(str(s or "").split()).strip()


def truncate_comment(s: str, n: int = MAX_COMMENT_CHARS) -> str:
    s = clean_text(s)
    return s if len(s) <= n else s[:n] + "\n\n[Comment truncated]"


def is_bot_username(username: Optional[str]) -> bool:
    if not username:
        return False
    return "bot" in username.lower()


def build_csv_bytes(rows: List[Dict[str, Any]], fieldnames: List[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})
    return buf.getvalue().encode("utf-8-sig")


def parse_iso(dt: str) -> datetime:
    # "2026-01-17T16:29:01.551Z" -> aware
    if dt.endswith("Z"):
        dt = dt[:-1] + "+00:00"
    return datetime.fromisoformat(dt)


def created_at_utc(row: Dict[str, Any]) -> datetime:
    ca = row.get("created_at")
    if isinstance(ca, str):
        try:
            return parse_iso(ca).astimezone(timezone.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def parse_json_relaxed(s: str) -> Any:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            s = inner

    m2 = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m2:
        return json.loads(m2.group(0))

    raise ValueError("Could not parse JSON")


def openrouter_headers(api_key: str, title: str) -> Dict[str, str]:
    if not api_key:
        raise RuntimeError("Missing OpenRouter API key.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": "https://localhost",
        "X-Title": title,
        "User-Agent": "metaculus-comments-judge/1.0",
    }


def openrouter_chat(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
    title: str,
) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0.0, "top_p": 1, "max_tokens": max_tokens}
    last: Optional[Exception] = None
    for k in range(3):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=openrouter_headers(api_key, title=title),
                json=payload,
                timeout=120,
            )
            if r.status_code == 429:
                ra = float(r.headers.get("Retry-After", "2") or 2)
                time.sleep(min(ra, 10))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            ch = data.get("choices") or []
            if not ch:
                raise RuntimeError("No choices")
            content = (ch[0].get("message") or {}).get("content") or ""
            if not content:
                raise RuntimeError("Empty content")
            return content
        except Exception as e:
            last = e
            time.sleep(0.7 * (k + 1))
    raise RuntimeError(f"OpenRouter failed: {repr(last)}")


class ParserFormatError(RuntimeError):
    pass


def build_judge_msgs(post_title: str, comment_text: str) -> List[Dict[str, str]]:
    user = (
        "Rate this comment using the strict 4-line format.\n\n"
        f"QUESTION_TITLE: {post_title}\n"
        f"COMMENT_TEXT:\n{comment_text}\n\n"
        "Exactly 4 lines, nothing else."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT_JUDGE}] + FEWSHOTS_JUDGE + [{"role": "user", "content": user}]


def normalize_parsed(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ParserFormatError("Not a dict")
    for k in ("score", "rationale", "flags", "evidence_urls"):
        if k not in obj:
            raise ParserFormatError(f"Missing {k}")

    try:
        score = int(obj.get("score"))
    except Exception:
        raise ParserFormatError("Bad score")
    if not (1 <= score <= 6):
        raise ParserFormatError("Score out of range")

    flags = obj.get("flags")
    if not isinstance(flags, dict):
        raise ParserFormatError("Bad flags")
    wanted = ["off_topic", "toxicity", "low_effort", "has_evidence", "likely_ai"]
    for k in wanted:
        if k not in flags:
            raise ParserFormatError(f"Missing flag {k}")

    ev = obj.get("evidence_urls")
    if not isinstance(ev, list):
        raise ParserFormatError("Bad evidence_urls")

    rationale = str(obj.get("rationale") or "")[:180]
    norm_flags = {k: bool(flags.get(k, False)) for k in wanted}

    seen, ev_out = set(), []
    for u in ev:
        su = str(u).strip()
        if su and su not in seen:
            seen.add(su)
            ev_out.append(su)

    return {"score": score, "rationale": rationale, "flags": norm_flags, "evidence_urls": ev_out}


def parse_with_retries(api_key: str, raw_judge_text: str, parser_model: str) -> Dict[str, Any]:
    base = [{"role": "system", "content": SYSTEM_PROMPT_TO_JSON}, {"role": "user", "content": raw_judge_text}]
    last: Optional[Exception] = None
    for t in range(PARSER_RETRIES):
        try:
            msgs = base if t == 0 else base + [{"role": "user", "content": "JSON only. Exactly the schema. No extra keys."}]
            txt = openrouter_chat(api_key, msgs, model=parser_model, max_tokens=260, title="Parser")
            return normalize_parsed(parse_json_relaxed(txt))
        except Exception as e:
            last = e
    raise ParserFormatError(f"Parser failed after retries: {repr(last)}")


def score_comment(
    api_key: str,
    judge_model: str,
    parser_model: str,
    post_title: str,
    comment_text: str,
) -> Dict[str, Any]:
    ct = truncate_comment(comment_text, MAX_COMMENT_CHARS)

    try:
        raw = openrouter_chat(
            api_key,
            build_judge_msgs(post_title, ct),
            model=judge_model,
            max_tokens=200,
            title="Judge",
        )
    except Exception:
        raw = (
            "score = 3\n"
            "rationale: Judge error; default neutral.\n"
            "flags: off_topic=false, toxicity=false, low_effort=false, has_evidence=false, likely_ai=false\n"
            "evidence_urls: []"
        )

    return parse_with_retries(api_key, raw, parser_model)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_csv_rows(data: bytes) -> Tuple[List[Dict[str, Any]], List[str]]:
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader), reader.fieldnames or []


# =========================
# Fetch logic
# =========================
@dataclass
class RunStats:
    posts_requested: int = 0
    posts_fetched: int = 0
    comment_pages_fetched: int = 0
    comments_seen: int = 0
    comments_unique: int = 0


def fetch_posts_range(
    session: requests.Session,
    headers: Dict[str, str],
    start_index: int,
    end_index: int,
    posts_page_size: int,
    sleep_s: float,
    progress_cb=None,
) -> List[Dict[str, Any]]:
    assert 0 <= start_index < end_index
    needed = end_index - start_index
    out: List[Dict[str, Any]] = []
    offset = start_index
    use_order_by = True

    while len(out) < needed:
        limit = min(posts_page_size, needed - len(out))
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if use_order_by:
            params["order_by"] = "-activity_at"

        try:
            payload = request_json(session, POSTS_URL, headers, params, sleep_s)
        except RuntimeError as e:
            msg = str(e)
            if use_order_by and ("order_by" in msg or "activity_at" in msg or "400" in msg):
                use_order_by = False
                continue
            raise

        chunk = extract_results(payload)
        if not chunk:
            break

        out.extend(chunk)
        offset += limit
        if progress_cb:
            progress_cb(len(out), needed, phase="posts")

        time.sleep(sleep_s)

    return out[:needed]


def fetch_comments_for_post(
    session: requests.Session,
    headers: Dict[str, str],
    post_id: int,
    mode: str,
    page_size: int,
    max_pages: int,
    sleep_s: float,
    slice_start: int,
    slice_end: int,
    stats: RunStats,
    per_page_cb=None,
) -> List[Dict[str, Any]]:
    base_variants = [{"post": post_id}, {"post_id": post_id}, {"post": str(post_id)}, {"post_id": str(post_id)}]
    order_variants = [{"sort": "-created_at"}, {"order_by": "-created_at"}, {}]

    chosen_base: Optional[Dict[str, Any]] = None
    chosen_order: Optional[Dict[str, Any]] = None

    for b in base_variants:
        for o in order_variants:
            probe_params = {"limit": 1, "offset": 0}
            probe_params.update(b)
            probe_params.update(o)
            try:
                payload = request_json(session, COMMENTS_URL, headers, probe_params, sleep_s)
                _ = extract_results(payload)
                chosen_base, chosen_order = b, o
                break
            except Exception:
                continue
        if chosen_base is not None:
            break

    if chosen_base is None or chosen_order is None:
        return []

    comments: List[Dict[str, Any]] = []

    if mode == "slice":
        if slice_end <= slice_start:
            return []
        limit = slice_end - slice_start
        params = {"limit": min(page_size, limit), "offset": slice_start}
        params.update(chosen_base)
        params.update(chosen_order)
        payload = request_json(session, COMMENTS_URL, headers, params, sleep_s)
        chunk = extract_results(payload)
        stats.comment_pages_fetched += 1
        stats.comments_seen += len(chunk)
        comments.extend(chunk)
        if per_page_cb:
            per_page_cb(post_id, 1, len(chunk), done=True)
        return comments

    if mode == "preview":
        params = {"limit": page_size, "offset": 0}
        params.update(chosen_base)
        params.update(chosen_order)
        payload = request_json(session, COMMENTS_URL, headers, params, sleep_s)
        chunk = extract_results(payload)
        stats.comment_pages_fetched += 1
        stats.comments_seen += len(chunk)
        comments.extend(chunk)
        if per_page_cb:
            per_page_cb(post_id, 1, len(chunk), done=True)
        return comments

    # mode == "all"
    offset = 0
    for page_idx in range(1, max_pages + 1):
        params = {"limit": page_size, "offset": offset}
        params.update(chosen_base)
        params.update(chosen_order)

        payload = request_json(session, COMMENTS_URL, headers, params, sleep_s)
        chunk = extract_results(payload)

        stats.comment_pages_fetched += 1
        stats.comments_seen += len(chunk)

        if not chunk:
            if per_page_cb:
                per_page_cb(post_id, page_idx, 0, done=True)
            break

        comments.extend(chunk)
        offset += page_size

        if per_page_cb:
            per_page_cb(post_id, page_idx, len(chunk), done=False)

        time.sleep(sleep_s)

    return comments


# =========================
# Streamlit app
# =========================
st.set_page_config(page_title="Metaculus: Posts + Comments (API)", layout="wide")
st.title("Metaculus — Posts & Comments (via https://www.metaculus.com/api/)")

with st.sidebar:
    st.header("Auth")
    token = st.text_input("Metaculus API token", type="password", help="Required for authenticated API access.")

    st.header("Judge (OpenRouter)")
    openrouter_key = st.text_input("OpenRouter API key", type="password")
    judge_model = st.text_input("Judge model ID", placeholder="e.g. openai/gpt-4.1")
    parser_model = st.text_input("Parser model ID", placeholder="e.g. openai/gpt-4o-mini")

    st.header("Posts selection (raw index range)")
    start_idx = st.number_input("Start index (inclusive)", min_value=0, value=0, step=1)
    end_idx = st.number_input("End index (exclusive)", min_value=1, value=40, step=1)
    posts_page_size = st.number_input("Posts page size", min_value=10, max_value=100, value=100, step=10)

    st.header("Comments mode")
    mode = st.selectbox(
        "How to fetch comments per post?",
        options=["preview", "all", "slice"],
        index=0,
        help="preview: 1 page per post | all: paginate | slice: fetch offsets [start,end)",
    )
    comments_page_size = st.number_input("Comments page size", min_value=20, max_value=500, value=200, step=20)
    max_comment_pages = st.number_input("Max comment pages per post (only for 'all')", min_value=1, max_value=500, value=50, step=10)

    st.subheader("Comment slice (only for 'slice')")
    c_slice_start = st.number_input("Comment slice start (offset)", min_value=0, value=0, step=1)
    c_slice_end = st.number_input("Comment slice end (exclusive)", min_value=1, value=20, step=1)

    st.header("Filters")
    cutoff_enabled = st.checkbox("Exclude comments after cutoff date (UTC)", value=False)
    cutoff_date = st.date_input("Cutoff date (UTC)", value=datetime.utcnow().date(), disabled=not cutoff_enabled)
    cutoff_time = st.time_input("Cutoff time (UTC)", value=datetime.utcnow().time(), disabled=not cutoff_enabled)
    st.caption("Bot filter: comments with 'bot' in the username are always removed.")

    st.header("Keep only the N most recent (GLOBAL)")
    keep_top_n = st.checkbox("Keep only N most recent comments (global)", value=False)
    top_n = st.number_input("N (most recent comments)", min_value=1, value=200, step=50, disabled=not keep_top_n)

    st.header("Performance / rate-limit")
    sleep_s = st.slider("Sleep between requests (seconds)", min_value=0.0, max_value=1.0, value=0.12, step=0.02)

    st.header("UI")
    preview_posts_rows = st.number_input("Posts preview rows", min_value=5, max_value=200, value=50, step=5)
    preview_comments_rows = st.number_input("Comments preview rows per post", min_value=3, max_value=50, value=8, step=1)

    st.header("Export")
    include_raw_json = st.checkbox("Include raw_json column in comments CSV", value=False)


run = st.button("Run", type="primary", use_container_width=True)

posts_progress = st.progress(0, text="Posts: waiting…")
comments_progress = st.progress(0, text="Comments: waiting…")
status = st.empty()
posts_table_ph = st.empty()
summary_ph = st.empty()


def posts_progress_cb(done: int, total: int, phase: str) -> None:
    pct = int(min(done / max(total, 1), 1.0) * 100)
    posts_progress.progress(pct, text=f"Posts: {done}/{total} fetched")


def comments_page_cb(post_id: int, page_idx: int, n_in_page: int, done: bool) -> None:
    # simple callback placeholder (per-post expander handles its own text)
    pass


def normalize_posts(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, p in enumerate(posts):
        out.append({
            "idx_in_range": i,
            "post_id": p.get("id"),
            "title": safe_post_title(p),
        })
    return out


def normalize_comments(post_id: int, post_title: str, comments: List[Dict[str, Any]], with_raw: bool) -> List[Dict[str, Any]]:
    out = []
    for c in comments:
        author = c.get("author") if isinstance(c.get("author"), dict) else {}
        row = {
            "post_id": post_id,
            "post_title": post_title,
            "comment_id": c.get("id"),
            "created_at": c.get("created_at"),
            "author_username": author.get("username") if isinstance(author, dict) else None,
            "vote_score": c.get("vote_score"),
            "parent_id": c.get("parent_id"),
            "root_id": c.get("root_id"),
            "text": c.get("text"),
        }
        if with_raw:
            row["raw_json"] = json.dumps(c, ensure_ascii=False)
        out.append(row)
    return out


def dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        cid = r.get("comment_id")
        if cid is None:
            continue
        k = str(cid)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def apply_comment_filters(
    rows: List[Dict[str, Any]],
    cutoff: Optional[datetime],
) -> Tuple[List[Dict[str, Any]], int, int]:
    filtered = []
    removed_bots = 0
    removed_cutoff = 0
    for r in rows:
        username = r.get("author_username")
        if is_bot_username(username):
            removed_bots += 1
            continue
        if cutoff is not None:
            created = created_at_utc(r)
            if created > cutoff:
                removed_cutoff += 1
                continue
        filtered.append(r)
    return filtered, removed_bots, removed_cutoff


def score_rows_to_csv(
    rows: List[Dict[str, Any]],
    api_key: str,
    judge_model: str,
    parser_model: str,
    status_ph: st.delta_generator.DeltaGenerator,
    progress_ph: st.delta_generator.DeltaGenerator,
) -> bytes:
    scored_rows: List[Dict[str, Any]] = []
    total = len(rows)
    for idx, row in enumerate(rows, 1):
        post_title = clean_text(row.get("post_title") or row.get("post_name") or "") or f"Post {row.get('post_id')}"
        comment_text = clean_text(row.get("text") or row.get("comment_text") or "")
        if not comment_text:
            continue
        status_ph.info(f"Scoring {idx}/{total} — post_id={row.get('post_id')} comment_id={row.get('comment_id')}")
        parsed = score_comment(api_key, judge_model, parser_model, post_title, comment_text)
        scored_rows.append({
            "ai_score": int(parsed["score"]),
            "rationale": parsed.get("rationale", ""),
            "flags_json": json.dumps(parsed.get("flags") or {}, ensure_ascii=False),
            "evidence_urls": ";".join(parsed.get("evidence_urls") or []),
            "ts": now_ts(),
            "post_id": row.get("post_id"),
            "post_title": post_title,
            "comment_id": row.get("comment_id"),
            "created_at": row.get("created_at"),
            "author_username": row.get("author_username"),
            "vote_score": row.get("vote_score"),
            "parent_id": row.get("parent_id"),
            "root_id": row.get("root_id"),
            "comment_text": comment_text,
        })
        progress_ph.progress(min(idx / max(total, 1), 1.0))
    return build_csv_bytes(scored_rows, CSV_COLUMNS_SCORED)


if run:
    if not token.strip():
        st.error("Token is required.")
        st.stop()
    if int(end_idx) <= int(start_idx):
        st.error("End index must be > Start index.")
        st.stop()

    session = requests.Session()
    headers = auth_headers(token.strip())

    stats = RunStats(posts_requested=int(end_idx - start_idx))
    status.write("Fetching posts…")

    try:
        posts = fetch_posts_range(
            session=session,
            headers=headers,
            start_index=int(start_idx),
            end_index=int(end_idx),
            posts_page_size=int(posts_page_size),
            sleep_s=float(sleep_s),
            progress_cb=posts_progress_cb,
        )
    except Exception as e:
        st.error(f"Failed to fetch posts: {e}")
        st.stop()

    stats.posts_fetched = len(posts)

    posts_norm = normalize_posts(posts)
    posts_table_ph.dataframe(posts_norm[: int(preview_posts_rows)], use_container_width=True)
    posts_progress.progress(100, text=f"Posts: {len(posts)}/{stats.posts_requested} fetched")

    status.write("Fetching comments per post…")

    post_list: List[Tuple[int, str]] = []
    for p in posts:
        pid = p.get("id")
        if isinstance(pid, int):
            post_list.append((pid, safe_post_title(p)))

    all_comments_rows: List[Dict[str, Any]] = []
    total_posts = len(post_list)
    done_posts = 0

    for pid, title in post_list:
        done_posts += 1
        comments_progress.progress(
            int(min(done_posts / max(total_posts, 1), 1.0) * 100),
            text=f"Comments: {done_posts}/{total_posts} posts processed | pages={stats.comment_pages_fetched} | seen={stats.comments_seen}",
        )

        with st.expander(f"Post {pid} — {title or '(no title)'}", expanded=(done_posts <= 3)):
            stat_ph = st.empty()
            table_ph = st.empty()

            try:
                comments = fetch_comments_for_post(
                    session=session,
                    headers=headers,
                    post_id=pid,
                    mode=mode,
                    page_size=int(comments_page_size),
                    max_pages=int(max_comment_pages),
                    sleep_s=float(sleep_s),
                    slice_start=int(c_slice_start),
                    slice_end=int(c_slice_end),
                    stats=stats,
                    per_page_cb=comments_page_cb,
                )
            except Exception as e:
                stat_ph.error(f"Failed to fetch comments for post {pid}: {e}")
                comments = []

            rows = normalize_comments(pid, title, comments, with_raw=bool(include_raw_json))
            preview_rows = [{
                "comment_id": r.get("comment_id"),
                "created_at": r.get("created_at"),
                "author_username": r.get("author_username"),
                "vote_score": r.get("vote_score"),
                "text_snippet": truncate(r.get("text"), 240),
            } for r in rows[: int(preview_comments_rows)]]

            stat_ph.write(f"Fetched {len(comments)} comments (mode={mode}).")
            table_ph.dataframe(preview_rows, use_container_width=True)

            all_comments_rows.extend(rows)

    # Global dedup + sort + filters + optional top-N
    deduped_comments = dedup_rows(all_comments_rows)
    deduped_comments.sort(key=created_at_utc, reverse=True)

    cutoff_dt = None
    if cutoff_enabled:
        cutoff_dt = datetime.combine(cutoff_date, cutoff_time, tzinfo=timezone.utc)

    filtered_comments, removed_bots, removed_cutoff = apply_comment_filters(deduped_comments, cutoff=cutoff_dt)

    if keep_top_n:
        filtered_comments = filtered_comments[: int(top_n)]

    stats.comments_unique = len(filtered_comments)

    comments_progress.progress(
        100,
        text=(
            f"Comments: done | pages={stats.comment_pages_fetched} | "
            f"seen={stats.comments_seen} | unique={stats.comments_unique}"
        ),
    )

    summary_ph.info(
        f"Done. Posts requested={stats.posts_requested}, fetched={stats.posts_fetched}. "
        f"Comment pages fetched={stats.comment_pages_fetched}, comments seen={stats.comments_seen}. "
        f"Removed bots={removed_bots}, removed after cutoff={removed_cutoff}. "
        f"Export rows (after filters/topN)={stats.comments_unique}."
    )

    # Exports
    st.divider()
    st.subheader("Exports")

    posts_fields = ["idx_in_range", "post_id", "title"]
    st.download_button(
        "Download posts.csv",
        data=build_csv_bytes(posts_norm, posts_fields),
        file_name=f"metaculus_posts_{int(start_idx)}_{int(end_idx)}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    comment_fields = [
        "post_id",
        "post_title",
        "comment_id",
        "created_at",
        "author_username",
        "vote_score",
        "parent_id",
        "root_id",
        "text",
    ]
    if include_raw_json:
        comment_fields.append("raw_json")

    suffix = f"top{int(top_n)}_" if keep_top_n else ""
    st.download_button(
        "Download comments.csv",
        data=build_csv_bytes(filtered_comments, comment_fields),
        file_name=f"metaculus_comments_{suffix}posts_{int(start_idx)}_{int(end_idx)}_{mode}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Judge (OpenRouter)")
    st.caption("Optionally score fetched comments or a CSV export with the judge prompts.")

    if openrouter_key and judge_model.strip() and parser_model.strip():
        score_status = st.empty()
        score_progress = st.progress(0.0, text="Judge: waiting…")

        if st.button("Score fetched comments with judge", use_container_width=True):
            if not filtered_comments:
                st.warning("No comments available to score.")
            else:
                scored_bytes = score_rows_to_csv(
                    filtered_comments,
                    api_key=openrouter_key.strip(),
                    judge_model=judge_model.strip(),
                    parser_model=parser_model.strip(),
                    status_ph=score_status,
                    progress_ph=score_progress,
                )
                st.download_button(
                    "Download scored comments CSV",
                    data=scored_bytes,
                    file_name="metaculus_comments_scored.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.info("Enter OpenRouter API key + model IDs to enable judging.")
else:
    st.caption(
        "Tips: Posts #20..#40 => Start=20, End=40. "
        "Commentaires #20..#40 par post => mode='slice', slice start=20, end=40. "
        "Top N global => active 'Keep only N most recent comments (global)'."
    )

st.divider()
st.subheader("Judge an existing comments CSV")
st.caption("Upload a comments.csv export, then score it with the same judge prompts.")

csv_upload = st.file_uploader("Upload comments CSV", type=["csv"])
if csv_upload is not None:
    csv_rows, csv_fields = read_csv_rows(csv_upload.getvalue())
    if not csv_rows:
        st.warning("Uploaded CSV is empty.")
    else:
        st.write(f"Rows detected: {len(csv_rows)}")
        cols = ["(none)"] + csv_fields
        default_text = "text" if "text" in csv_fields else (csv_fields[0] if csv_fields else "(none)")
        default_title = "post_title" if "post_title" in csv_fields else "(none)"
        default_user = "author_username" if "author_username" in csv_fields else "(none)"
        default_created = "created_at" if "created_at" in csv_fields else "(none)"

        col_text = st.selectbox("comment text column", options=cols, index=cols.index(default_text) if default_text in cols else 0)
        col_title = st.selectbox("post title column", options=cols, index=cols.index(default_title) if default_title in cols else 0)
        col_user = st.selectbox("author username column", options=cols, index=cols.index(default_user) if default_user in cols else 0)
        col_created = st.selectbox("created_at column", options=cols, index=cols.index(default_created) if default_created in cols else 0)

        if st.button("Score uploaded CSV with judge", use_container_width=True):
            if not openrouter_key or not judge_model.strip() or not parser_model.strip():
                st.error("Provide OpenRouter API key + judge + parser model IDs.")
            elif col_text == "(none)":
                st.error("Select a comment text column.")
            else:
                mapped_rows = []
                for r in csv_rows:
                    mapped_rows.append({
                        "post_id": r.get("post_id"),
                        "post_title": r.get(col_title) if col_title != "(none)" else "",
                        "comment_id": r.get("comment_id"),
                        "created_at": r.get(col_created) if col_created != "(none)" else None,
                        "author_username": r.get(col_user) if col_user != "(none)" else None,
                        "vote_score": r.get("vote_score"),
                        "parent_id": r.get("parent_id"),
                        "root_id": r.get("root_id"),
                        "text": r.get(col_text),
                    })

                cutoff_dt = None
                if cutoff_enabled:
                    cutoff_dt = datetime.combine(cutoff_date, cutoff_time, tzinfo=timezone.utc)
                filtered_rows, _, _ = apply_comment_filters(mapped_rows, cutoff=cutoff_dt)

                score_status = st.empty()
                score_progress = st.progress(0.0, text="Judge: scoring…")
                scored_bytes = score_rows_to_csv(
                    filtered_rows,
                    api_key=openrouter_key.strip(),
                    judge_model=judge_model.strip(),
                    parser_model=parser_model.strip(),
                    status_ph=score_status,
                    progress_ph=score_progress,
                )
                st.download_button(
                    "Download scored CSV",
                    data=scored_bytes,
                    file_name="comments_scored.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
