#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import streamlit as st

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore


# -----------------------------
# Config (single endpoints)
# -----------------------------
BASE = "https://www.metaculus.com"
POSTS_URL = f"{BASE}/api/posts/"
COMMENTS_URL = f"{BASE}/api/comments/"

POSTS_PAGE_SIZE = 100

COMMENTS_PAGE_SIZE = 200   # pagination size for /api/comments
MAX_COMMENT_PAGES_PER_POST = 200  # safety cap: 200 * 200 = 40k comments/post max

REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 5
SLEEP_S = 0.12


# -----------------------------
# Helpers
# -----------------------------
def auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
        "User-Agent": "metaculus-scan-n-posts-then-sort/1.0",
    }


def parse_iso(dt: str) -> datetime:
    # "2026-01-17T16:29:01.551Z" -> aware
    if dt.endswith("Z"):
        dt = dt[:-1] + "+00:00"
    return datetime.fromisoformat(dt)


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def request_json(session: requests.Session, url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Dict[str, Any]:
    last_err: Optional[str] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_S)

            if r.status_code in (401, 403):
                raise RuntimeError(f"Auth failed (HTTP {r.status_code}). Check token. Body: {r.text[:200]}")

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


def get_post_id(post_obj: Dict[str, Any]) -> Optional[int]:
    v = post_obj.get("id")
    return v if isinstance(v, int) else None


def normalize_comment_row(post_id: int, c: Dict[str, Any]) -> Dict[str, Any]:
    author = c.get("author") if isinstance(c.get("author"), dict) else {}
    return {
        "post_id": post_id,
        "comment_id": c.get("id"),
        "created_at": c.get("created_at"),
        "author_id": author.get("id") if isinstance(author, dict) else None,
        "author_username": author.get("username") if isinstance(author, dict) else None,
        "vote_score": c.get("vote_score"),
        "parent_id": c.get("parent_id"),
        "root_id": c.get("root_id"),
        "text": c.get("text"),
        "raw_json": json.dumps(c, ensure_ascii=False),
    }


def build_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    fieldnames = [
        "post_id",
        "comment_id",
        "created_at",
        "author_id",
        "author_username",
        "vote_score",
        "parent_id",
        "root_id",
        "text",
        "raw_json",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})
    return buf.getvalue().encode("utf-8-sig")


# -----------------------------
# Core logic
# -----------------------------
@dataclass
class Stats:
    posts_target: int = 0
    posts_collected: int = 0
    posts_scanned_for_comments: int = 0
    comment_pages_fetched: int = 0
    comments_seen: int = 0
    comments_unique: int = 0


def collect_n_post_ids(session: requests.Session, headers: Dict[str, str], n_posts: int) -> List[int]:
    post_ids: List[int] = []
    offset = 0

    while len(post_ids) < n_posts:
        payload = request_json(session, POSTS_URL, headers, {"limit": POSTS_PAGE_SIZE, "offset": offset})
        posts = extract_results(payload)
        if not posts:
            break

        for p in posts:
            pid = get_post_id(p)
            if pid is not None:
                post_ids.append(pid)
                if len(post_ids) >= n_posts:
                    break

        offset += POSTS_PAGE_SIZE
        time.sleep(SLEEP_S)

    # de-dup while preserving order (defensive)
    seen: Set[int] = set()
    out: List[int] = []
    for pid in post_ids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out[:n_posts]


def fetch_all_comments_for_post(
    session: requests.Session,
    headers: Dict[str, str],
    post_id: int,
    stats: Stats,
) -> List[Dict[str, Any]]:
    all_comments: List[Dict[str, Any]] = []
    offset = 0

    for _ in range(MAX_COMMENT_PAGES_PER_POST):
        params = {
            "post": post_id,
            "limit": COMMENTS_PAGE_SIZE,
            "offset": offset,
            "sort": "-created_at",      # stable and efficient; still we will sort globally later
            "is_private": "false",
        }
        payload = request_json(session, COMMENTS_URL, headers, params)
        stats.comment_pages_fetched += 1

        chunk = extract_results(payload)
        if not chunk:
            break

        all_comments.extend(chunk)
        stats.comments_seen += len(chunk)

        offset += COMMENTS_PAGE_SIZE
        time.sleep(SLEEP_S)

    return all_comments


def apply_cutoff(comments: List[Tuple[int, Dict[str, Any]]], cutoff_utc: Optional[datetime]) -> List[Tuple[int, Dict[str, Any]]]:
    if cutoff_utc is None:
        return comments

    out: List[Tuple[int, Dict[str, Any]]] = []
    for post_id, c in comments:
        ca = c.get("created_at")
        if not isinstance(ca, str):
            continue
        try:
            created_utc = to_utc(parse_iso(ca))
        except Exception:
            continue
        if created_utc >= cutoff_utc:
            out.append((post_id, c))
    return out


def sort_by_created_desc(comments: List[Tuple[int, Dict[str, Any]]]) -> List[Tuple[int, Dict[str, Any]]]:
    def key_fn(item: Tuple[int, Dict[str, Any]]) -> datetime:
        ca = item[1].get("created_at")
        if isinstance(ca, str):
            try:
                return to_utc(parse_iso(ca))
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)
        return datetime.min.replace(tzinfo=timezone.utc)

    return sorted(comments, key=key_fn, reverse=True)


def scan_n_posts_then_sort(
    token: str,
    target_n_comments: int,
    cutoff_utc: Optional[datetime],
    progress_cb=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Stats]:
    """
    Algorithm requested:
    1) collect exactly N post_ids (N = desired comments count)
    2) fetch ALL comments for each of these posts (pagination)
    3) aggregate comments, de-dup by comment_id
    4) sort globally by created_at desc
    5) outputs:
       - top_n (after optional cutoff): min(N, #after_cutoff) most recent
       - after_cutoff_all: all comments after cutoff (or all if cutoff disabled), sorted desc
    """
    headers = auth_headers(token)
    session = requests.Session()
    stats = Stats(posts_target=target_n_comments)

    post_ids = collect_n_post_ids(session, headers, n_posts=target_n_comments)
    stats.posts_collected = len(post_ids)

    # Fetch all comments for each post
    paired: List[Tuple[int, Dict[str, Any]]] = []
    for idx, pid in enumerate(post_ids, start=1):
        stats.posts_scanned_for_comments += 1
        try:
            comments = fetch_all_comments_for_post(session, headers, pid, stats)
        except Exception:
            time.sleep(SLEEP_S)
            comments = []
        for c in comments:
            paired.append((pid, c))

        if progress_cb:
            progress_cb(idx, len(post_ids), stats)

    # De-dup by comment_id (sitewide)
    seen_cids: Set[int] = set()
    deduped: List[Tuple[int, Dict[str, Any]]] = []
    for pid, c in paired:
        cid = c.get("id")
        if not isinstance(cid, int):
            continue
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        deduped.append((pid, c))
    stats.comments_unique = len(deduped)

    # Cutoff then sort
    after_cutoff = apply_cutoff(deduped, cutoff_utc)
    after_cutoff_sorted = sort_by_created_desc(after_cutoff)

    # Top-N after cutoff (or top-N overall if cutoff disabled)
    top_n_sorted = after_cutoff_sorted[:target_n_comments]

    # Normalize rows for CSV
    top_n_rows = [normalize_comment_row(pid, c) for pid, c in top_n_sorted]
    after_cutoff_rows = [normalize_comment_row(pid, c) for pid, c in after_cutoff_sorted]

    return top_n_rows, after_cutoff_rows, stats


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Metaculus — Scan N posts, fetch all comments, sort", layout="wide")
st.title("Metaculus: scan N posts → fetch all comments per post → sort by date → export CSV")

with st.sidebar:
    st.header("Inputs")
    token = st.text_input("Metaculus API token", type="password")

    n_comments = st.number_input(
        "N (desired comments)",
        min_value=1,
        max_value=5000,
        value=400,
        step=50,
        help="This app will scan exactly N posts, then fetch all comments for each scanned post.",
    )

    st.subheader("Cutoff (keep created_at >= cutoff)")
    enable_cutoff = st.checkbox("Enable cutoff", value=True)

    tz_name = st.selectbox("Timezone for cutoff input", options=["America/New_York", "Europe/Paris", "UTC"], index=0)
    cutoff_d = st.date_input("Cutoff date", value=date.today())
    cutoff_t = st.time_input("Cutoff time", value=dtime(0, 0))


def compute_cutoff_utc() -> Optional[datetime]:
    if not enable_cutoff:
        return None
    dt_naive = datetime.combine(cutoff_d, cutoff_t)
    if tz_name == "UTC":
        return dt_naive.replace(tzinfo=timezone.utc)
    if ZoneInfo is None:
        return dt_naive.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(tz_name)
    return dt_naive.replace(tzinfo=tz).astimezone(timezone.utc)


cutoff_utc = compute_cutoff_utc()
if cutoff_utc is None:
    st.caption("Cutoff: disabled (no filtering). Output ‘after cutoff’ == all scanned comments.")
else:
    st.caption(f"Cutoff (UTC): {cutoff_utc.isoformat()}")

run = st.button("Run", type="primary", use_container_width=True)
progress = st.progress(0)
status = st.empty()

if "top_n_rows" not in st.session_state:
    st.session_state.top_n_rows = []
if "after_cutoff_rows" not in st.session_state:
    st.session_state.after_cutoff_rows = []
if "stats" not in st.session_state:
    st.session_state.stats = None


def progress_cb(done_posts: int, total_posts: int, stats: Stats) -> None:
    frac = min(done_posts / max(total_posts, 1), 1.0)
    progress.progress(int(frac * 100))
    status.write(
        f"posts {done_posts}/{total_posts} | "
        f"comment pages fetched={stats.comment_pages_fetched} | "
        f"comments seen={stats.comments_seen} | unique={stats.comments_unique or '…'}"
    )


if run:
    if not token.strip():
        st.error("Token is required.")
    else:
        progress.progress(0)
        status.write("Starting…")

        try:
            top_n_rows, after_cutoff_rows, stats = scan_n_posts_then_sort(
                token=token.strip(),
                target_n_comments=int(n_comments),
                cutoff_utc=cutoff_utc,
                progress_cb=progress_cb,
            )

            st.session_state.top_n_rows = top_n_rows
            st.session_state.after_cutoff_rows = after_cutoff_rows
            st.session_state.stats = stats

            progress.progress(100)
            st.success("Done.")
        except Exception as e:
            st.error(f"Run failed: {e}")

stats = st.session_state.stats
if stats is not None:
    st.write(
        f"Stats — posts_target={stats.posts_target}, posts_collected={stats.posts_collected}, "
        f"posts_scanned_for_comments={stats.posts_scanned_for_comments}, "
        f"comment_pages_fetched={stats.comment_pages_fetched}, comments_seen={stats.comments_seen}, "
        f"comments_unique={stats.comments_unique}"
    )

top_n_rows = st.session_state.top_n_rows or []
after_cutoff_rows = st.session_state.after_cutoff_rows or []

if top_n_rows or after_cutoff_rows:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # CSV 1: Top N most recent (after cutoff if enabled)
    top_csv = build_csv_bytes(top_n_rows)
    top_name = f"metaculus_topN_{len(top_n_rows)}_{ts}.csv"
    st.download_button(
        "Download CSV — Top N most recent (after cutoff if enabled)",
        data=top_csv,
        file_name=top_name,
        mime="text/csv",
        use_container_width=True,
    )

    # CSV 2: All after cutoff (or all scanned if cutoff disabled)
    all_csv = build_csv_bytes(after_cutoff_rows)
    all_name = f"metaculus_after_cutoff_{len(after_cutoff_rows)}_{ts}.csv"
    st.download_button(
        "Download CSV — All comments after cutoff (or all scanned if cutoff disabled)",
        data=all_csv,
        file_name=all_name,
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Preview (first 20 rows of Top N)"):
        st.json(top_n_rows[:20], expanded=False)
else:
    st.info("No output yet. Click Run.")


