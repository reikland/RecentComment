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

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore


# =========================
# Config (Metaculus API v1)
# =========================
BASE_API = "https://www.metaculus.com/api"
COMMENTS_URL = f"{BASE_API}/comments/"

REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 6

DEFAULT_PAGE_SIZE = 200
MAX_PAGE_SIZE = 500


# =========================
# Helpers
# =========================
def auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
        "User-Agent": "metaculus-streamlit-recent-comments/1.1",
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


def parse_iso(dt: str) -> datetime:
    # "2026-01-17T16:29:01.551Z" -> aware
    s = (dt or "").strip()
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def comment_created_at_utc(c: Dict[str, Any]) -> datetime:
    ca = c.get("created_at") or c.get("createdAt")
    if isinstance(ca, str):
        d = parse_iso(ca)
        return d.astimezone(timezone.utc) if d.tzinfo else datetime.min.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def clean_text(s: Any) -> str:
    return " ".join(str(s or "").split()).strip()


def truncate(s: str, n: int = 220) -> str:
    s = clean_text(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def is_bot_username(username: Optional[str]) -> bool:
    return bool(username) and ("bot" in str(username).lower())


def build_csv_bytes(rows: List[Dict[str, Any]], fieldnames: List[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})
    return buf.getvalue().encode("utf-8-sig")


def pick_author_username(c: Dict[str, Any]) -> str:
    a = c.get("author")
    if isinstance(a, dict):
        u = a.get("username") or a.get("name")
        if isinstance(u, str) and u.strip():
            return u.strip()
    for k in ("author_username", "username", "user", "created_by_username"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def pick_post_id(c: Dict[str, Any]) -> Optional[int]:
    for k in ("post", "post_id", "postId"):
        v = c.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
        if isinstance(v, dict):
            pid = v.get("id")
            if isinstance(pid, int):
                return pid
    return None


def pick_comment_id(c: Dict[str, Any]) -> Optional[int]:
    v = c.get("id")
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def pick_vote_score(c: Dict[str, Any]) -> Optional[int]:
    for k in ("vote_score", "score", "votes"):
        v = c.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip():
            try:
                return int(float(v.strip()))
            except Exception:
                pass
    return None


def pick_parent_id(c: Dict[str, Any]) -> Optional[int]:
    for k in ("parent_id", "parent", "parent_comment", "in_reply_to"):
        v = c.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
    return None


def pick_root_id(c: Dict[str, Any]) -> Optional[int]:
    for k in ("root_id", "root", "root_comment_id"):
        v = c.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
    return None


# =========================
# Fetcher: global most recent comments until N or cutoff
# =========================
@dataclass
class FetchStats:
    pages_fetched: int = 0
    comments_seen: int = 0
    comments_kept: int = 0
    removed_bots: int = 0
    removed_cutoff: int = 0
    ordering_unreliable: bool = False


def choose_comments_order_params(
    session: requests.Session,
    headers: Dict[str, str],
    sleep_s: float,
) -> Dict[str, Any]:
    """
    Probe to find an ordering parameter that the deployment accepts.
    """
    candidates = [
        {"sort": "-created_at"},
        {"order_by": "-created_at"},
        {"ordering": "-created_at"},
        {},
    ]
    for cand in candidates:
        try:
            payload = request_json(
                session,
                COMMENTS_URL,
                headers,
                params={"limit": 3, "offset": 0, **cand},
                sleep_s=sleep_s,
            )
            chunk = extract_results(payload)
            if isinstance(chunk, list):
                return cand
        except Exception:
            continue
    return {}


def _chunk_is_desc_by_created_at(chunk: List[Dict[str, Any]]) -> bool:
    dts = [comment_created_at_utc(c) for c in chunk]
    # If any timestamp is missing/unparseable, do not trust ordering
    if any(dt.year <= 1901 for dt in dts):
        return False
    return all(dts[i] >= dts[i + 1] for i in range(len(dts) - 1))


def fetch_recent_comments_global(
    session: requests.Session,
    headers: Dict[str, str],
    n_wanted: int,
    cutoff_utc: Optional[datetime],
    page_size: int,
    sleep_s: float,
    exclude_bots: bool,
    progress_cb=None,
    hard_page_cap: int = 5000,
) -> Tuple[List[Dict[str, Any]], FetchStats]:
    """
    Fetch comments newest-first (global). Keep collecting until:
      - have N comments, OR
      - we are confident we've crossed the cutoff in a descending stream, OR
      - no more results.

    IMPORTANT: cutoff is enforced strictly as a filter (never include older-than-cutoff comments),
    and the "stop when crossing cutoff" optimization is only used if ordering looks reliable.
    """
    stats = FetchStats()
    order_params = choose_comments_order_params(session, headers, sleep_s=sleep_s)

    collected: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    offset = 0

    page_size = max(20, min(int(page_size), MAX_PAGE_SIZE))
    n_wanted = max(1, int(n_wanted))

    while len(collected) < n_wanted and stats.pages_fetched < hard_page_cap:
        params: Dict[str, Any] = {"limit": page_size, "offset": offset}
        params.update(order_params)

        payload = request_json(session, COMMENTS_URL, headers, params=params, sleep_s=sleep_s)
        chunk = extract_results(payload)

        stats.pages_fetched += 1
        stats.comments_seen += len(chunk)

        if not chunk:
            break

        # Detect weird pagination loops
        ids_in_chunk = [pick_comment_id(c) for c in chunk if pick_comment_id(c) is not None]
        if ids_in_chunk and all(int(cid) in seen_ids for cid in ids_in_chunk if cid is not None):
            break

        # Ordering reliability (for cutoff early stop only)
        is_desc = _chunk_is_desc_by_created_at(chunk)
        if not is_desc:
            stats.ordering_unreliable = True

        # Process chunk, always enforcing cutoff as a filter (never include older)
        oldest_dt_in_chunk = None
        for c in chunk:
            cid = pick_comment_id(c)
            if cid is None:
                continue
            if int(cid) in seen_ids:
                continue
            seen_ids.add(int(cid))

            created = comment_created_at_utc(c)
            if oldest_dt_in_chunk is None or created < oldest_dt_in_chunk:
                oldest_dt_in_chunk = created

            if cutoff_utc is not None and created < cutoff_utc:
                stats.removed_cutoff += 1
                continue  # strict filter: never include

            author_username = pick_author_username(c)
            if exclude_bots and is_bot_username(author_username):
                stats.removed_bots += 1
                continue

            collected.append(c)
            stats.comments_kept += 1
            if len(collected) >= n_wanted:
                break

        if progress_cb:
            progress_cb(stats, len(collected), n_wanted)

        # Early stop only if ordering looks reliable and the oldest element in chunk is < cutoff
        if cutoff_utc is not None and is_desc and oldest_dt_in_chunk is not None and oldest_dt_in_chunk < cutoff_utc:
            break

        # Next page
        offset += len(chunk)
        time.sleep(sleep_s)

    return collected[:n_wanted], stats


# =========================
# Streamlit app
# =========================
st.set_page_config(page_title="Metaculus — Recent Comments Fetcher", layout="wide")
st.title("Metaculus — Recent Comments Fetcher (global, newest-first)")

# Stable defaults (avoid auto-updating cutoff time on rerun)
if "cutoff_enabled" not in st.session_state:
    st.session_state["cutoff_enabled"] = True
if "cutoff_tz" not in st.session_state:
    st.session_state["cutoff_tz"] = "Europe/Paris"
if "cutoff_date" not in st.session_state:
    st.session_state["cutoff_date"] = datetime.utcnow().date()
if "cutoff_time" not in st.session_state:
    # freeze the initial default; user edits persist thereafter
    st.session_state["cutoff_time"] = datetime.utcnow().time().replace(microsecond=0)
if "exclude_bots" not in st.session_state:
    st.session_state["exclude_bots"] = True


def compute_cutoff_utc(enabled: bool, d, t, tz_choice: str) -> Optional[datetime]:
    if not enabled:
        return None
    naive = datetime.combine(d, t)
    if tz_choice == "UTC":
        return naive.replace(tzinfo=timezone.utc)
    # Europe/Paris
    if ZoneInfo is None:
        # Cannot do correct tz conversion without zoneinfo; fallback to UTC interpretation
        return naive.replace(tzinfo=timezone.utc)
    try:
        loc = naive.replace(tzinfo=ZoneInfo("Europe/Paris"))  # type: ignore[arg-type]
        return loc.astimezone(timezone.utc)
    except Exception:
        return naive.replace(tzinfo=timezone.utc)


with st.sidebar:
    st.header("Auth")
    token = st.text_input("Metaculus API token", type="password")

    st.header("Target")
    n_wanted = st.number_input("N most recent comments to collect", min_value=1, max_value=5000, value=200, step=50)

    st.header("Cutoff (strict filter + stop when crossed)")
    cutoff_enabled = st.checkbox("Enable cutoff", key="cutoff_enabled")
    cutoff_tz = st.selectbox(
        "Cutoff timezone",
        options=["UTC", "Europe/Paris"],
        key="cutoff_tz",
        disabled=not cutoff_enabled,
        help="Cutoff is converted to UTC before filtering (if zoneinfo is available).",
    )

    cutoff_date = st.date_input("Cutoff date", key="cutoff_date", disabled=not cutoff_enabled)
    cutoff_time = st.time_input("Cutoff time", key="cutoff_time", disabled=not cutoff_enabled)

    if cutoff_enabled and cutoff_tz == "Europe/Paris" and ZoneInfo is None:
        st.warning("zoneinfo unavailable in this environment; cutoff will be interpreted as UTC (approximation).")

    st.header("Pagination / rate-limit")
    page_size = st.number_input("Page size", min_value=20, max_value=MAX_PAGE_SIZE, value=DEFAULT_PAGE_SIZE, step=20)
    sleep_s = st.slider("Sleep between requests (seconds)", min_value=0.0, max_value=1.0, value=0.12, step=0.02)

    st.header("Filters")
    exclude_bots = st.checkbox("Exclude usernames containing 'bot'", key="exclude_bots", value=True)


run = st.button("Fetch recent comments", type="primary", use_container_width=True)

prog = st.progress(0, text="Waiting…")
status = st.empty()
preview_ph = st.empty()
summary_ph = st.empty()


def normalize_comment_row(c: Dict[str, Any]) -> Dict[str, Any]:
    created_dt = comment_created_at_utc(c)
    created_s = created_dt.isoformat().replace("+00:00", "Z") if created_dt.tzinfo else str(c.get("created_at") or "")
    author_username = pick_author_username(c)
    text = clean_text(c.get("text") or c.get("comment_text") or "")
    return {
        "comment_id": pick_comment_id(c),
        "created_at": created_s,
        "author_username": author_username,
        "vote_score": pick_vote_score(c),
        "post_id": pick_post_id(c),
        "parent_id": pick_parent_id(c),
        "root_id": pick_root_id(c),
        "comment_text": text,
        "snippet": truncate(text, 240),
        "raw_json": json.dumps(c, ensure_ascii=False),
    }


if run:
    if not token.strip():
        st.error("Token is required.")
        st.stop()

    cutoff_utc = compute_cutoff_utc(cutoff_enabled, cutoff_date, cutoff_time, cutoff_tz)

    # Display cutoff in both tz + UTC for sanity
    if cutoff_enabled:
        naive = datetime.combine(cutoff_date, cutoff_time)
        if cutoff_tz == "Europe/Paris" and ZoneInfo is not None:
            local = naive.replace(tzinfo=ZoneInfo("Europe/Paris"))  # type: ignore[arg-type]
            local_s = local.isoformat()
        elif cutoff_tz == "UTC":
            local_s = naive.replace(tzinfo=timezone.utc).isoformat()
        else:
            local_s = naive.isoformat() + " (naive)"
        status.info(f"Cutoff local: {local_s} | cutoff UTC: {cutoff_utc.isoformat().replace('+00:00','Z') if cutoff_utc else '(none)'}")

    session = requests.Session()
    headers = auth_headers(token.strip())

    def progress_cb(stats: FetchStats, kept: int, target: int) -> None:
        pct = int(min(kept / max(1, target), 1.0) * 100)
        warn = " | ordering_unreliable=true" if stats.ordering_unreliable else ""
        prog.progress(
            pct,
            text=(
                f"Kept {kept}/{target} | pages={stats.pages_fetched} seen={stats.comments_seen} "
                f"bots_removed={stats.removed_bots} cutoff_removed={stats.removed_cutoff}{warn}"
            ),
        )

    try:
        comments, stats = fetch_recent_comments_global(
            session=session,
            headers=headers,
            n_wanted=int(n_wanted),
            cutoff_utc=cutoff_utc,
            page_size=int(page_size),
            sleep_s=float(sleep_s),
            exclude_bots=bool(exclude_bots),
            progress_cb=progress_cb,
        )
    except Exception as e:
        st.error(f"Fetch failed: {e}")
        st.stop()

    rows = [normalize_comment_row(c) for c in comments]

    # Extra safety: strict post-filter for cutoff (in case of any upstream parse anomalies)
    if cutoff_utc is not None:
        kept2 = []
        for r in rows:
            dt = parse_iso(str(r.get("created_at") or ""))
            if dt.astimezone(timezone.utc) >= cutoff_utc:
                kept2.append(r)
        rows = kept2

    prog.progress(100, text=f"Done. Kept {len(rows)}/{int(n_wanted)} (or hit cutoff / exhausted).")

    cutoff_txt = cutoff_utc.isoformat().replace("+00:00", "Z") if cutoff_utc else "(none)"
    summary_ph.info(
        f"Result: kept={len(rows)} target={int(n_wanted)} | cutoff_utc={cutoff_txt} | "
        f"pages={stats.pages_fetched} seen={stats.comments_seen} | "
        f"bots_removed={stats.removed_bots} cutoff_removed={stats.removed_cutoff} | "
        f"ordering_unreliable={stats.ordering_unreliable}"
    )

    st.subheader("Preview")
    preview_cols = ["comment_id", "created_at", "author_username", "vote_score", "post_id", "snippet"]
    preview_ph.dataframe([{k: r.get(k) for k in preview_cols} for r in rows[:200]], use_container_width=True, height=420)

    st.subheader("Downloads")
    csv_fields = ["comment_id", "created_at", "author_username", "vote_score", "post_id", "parent_id", "root_id", "comment_text"]
    st.download_button(
        "Download comments.csv",
        data=build_csv_bytes(rows, csv_fields),
        file_name="metaculus_recent_comments.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "Download comments.json",
        data=json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="metaculus_recent_comments.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Raw (first 5 comments)", expanded=False):
        st.code("\n\n".join(r.get("raw_json", "") for r in rows[:5]) or "(empty)")
else:
    st.caption(
        "Cutoff inputs are persisted in session_state (they do not auto-reset to current time on reruns). "
        "Cutoff is enforced as a strict filter; early-stop on cutoff is only used when ordering looks reliable."
    )

