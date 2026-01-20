#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import io
import json
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

    # Global dedup + sort + optional top-N
    deduped_comments = dedup_rows(all_comments_rows)
    deduped_comments.sort(key=created_at_utc, reverse=True)

    if keep_top_n:
        deduped_comments = deduped_comments[: int(top_n)]

    stats.comments_unique = len(deduped_comments)

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
        f"Export rows (after dedup/sort/topN)={stats.comments_unique}."
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
        data=build_csv_bytes(deduped_comments, comment_fields),
        file_name=f"metaculus_comments_{suffix}posts_{int(start_idx)}_{int(end_idx)}_{mode}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.caption(
        "Tips: Posts #20..#40 => Start=20, End=40. "
        "Commentaires #20..#40 par post => mode='slice', slice start=20, end=40. "
        "Top N global => active 'Keep only N most recent comments (global)'."
    )

