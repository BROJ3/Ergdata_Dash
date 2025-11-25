import requests
import time
from bs4 import BeautifulSoup
import sqlite3
import json
from datetime import datetime, timedelta
import config
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import random
from threading import Semaphore
from collections import deque

BATCH_SIZE = 10

# API paging
PER_PAGE = 250
PAGE_SLEEP_SEC = 0.25  # polite delay between result pages

# Stroke fetching throttles
STROKE_CONCURRENCY = 3
STROKE_SEM = Semaphore(STROKE_CONCURRENCY)

JITTER_MIN, JITTER_MAX = 0.05, 0.20

# season 25
DEFAULT_FROM = "2025-10-21"
DEFAULT_TO   = "2026-03-01"

# Interleaved work processing
ROWER_PARALLELISM = 2     # max simultaneous requests per single rower
GLOBAL_WORKERS     = 16   # total threads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("concept2")

# =========================
access_token = config.access_token

def make_session() -> requests.Session:
    """Create a requests.Session with auth, keep-alive pooling, and retries."""
    s = requests.Session()
    s.headers.update({'Authorization': f'Bearer {access_token}'})
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

# Global session for main thread light calls
session = make_session()

# =========================
# DB setup
# =========================
def setup_db(conn: sqlite3.Connection) -> sqlite3.Cursor:
    cur = conn.cursor()
    cur.execute('PRAGMA foreign_keys = ON')

    # Note: using MySQL-like DDL is tolerated by SQLite but AUTOINCREMENT is the canonical keyword.
    cur.execute('''
    CREATE TABLE IF NOT EXISTS crnjakt_rowers (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        partner_id VARCHAR(255) UNIQUE NOT NULL,
        name       VARCHAR(255) NOT NULL
    );
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS crnjakt_workouts (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        rower_id       VARCHAR(255) NOT NULL,
        workout_id     VARCHAR(255) UNIQUE NOT NULL,
        name           VARCHAR(200),
        type           VARCHAR(30),
        distance       INT,
        date           DATE,
        weekday        VARCHAR(30),
        hour           TIME,
        time           INT,
        timezone       VARCHAR(255),
        date_utc       DATETIME,
        heart_rate     INT,
        calories_total INT,
        stroke_data    JSON,
        FOREIGN KEY (rower_id) REFERENCES crnjakt_rowers(partner_id)
    );
    ''')

    cur.execute("CREATE INDEX IF NOT EXISTS idx_workouts_rower_date ON crnjakt_workouts(rower_id, date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_workouts_workout_id ON crnjakt_workouts(workout_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rowers_partner_id ON crnjakt_rowers(partner_id)")
    return cur

def insert_rower(conn, cursor, rower):
    sql = '''
    INSERT INTO crnjakt_rowers (partner_id, name)
    VALUES (?, ?)
    ON CONFLICT(partner_id) DO UPDATE SET
        name = excluded.name
    '''
    with conn:
        cursor.execute(sql, (rower['partner_id'], rower['name']))

def insert_workouts_chunked(conn, cursor, rows: list[tuple], chunk: int = 500) -> int:
    """Chunked executemany insert that matches the schema/tuple order exactly."""
    if not rows:
        return 0
    total = 0
    sql = '''
    INSERT INTO crnjakt_workouts (
        rower_id, workout_id, name, type, distance, date, weekday, hour,
        time, timezone, date_utc, heart_rate, calories_total, stroke_data
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(workout_id) DO NOTHING
    '''
    for i in range(0, len(rows), chunk):
        batch = rows[i:i+chunk]
        with conn:
            cursor.executemany(sql, batch)
        total += len(batch)
    return total

def get_latest_workout_date(cursor, partner_id: str):
    cursor.execute("SELECT MAX(date) FROM crnjakt_workouts WHERE rower_id = ?", (partner_id,))
    row = cursor.fetchone()
    return row[0] if row else None

def get_existing_workout_ids(cursor, partner_id: str) -> set[str]:
    cursor.execute("SELECT workout_id FROM crnjakt_workouts WHERE rower_id = ?", (partner_id,))
    return {r[0] for r in cursor.fetchall()}

# =========================
# Web / scraping
# =========================
def login_and_get_rowers(session) -> dict:
    login_url = "https://log.concept2.com/login"
    partners_url = "https://log.concept2.com/team/" + config.my_team
    resp = session.post(login_url, data=config.login_payload, allow_redirects=True)
    if not resp.ok:
        log.error("Login failed. status=%s", resp.status_code)
        return {}

    log.info("Login successful.")
    resp = session.get(partners_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="table js-tablesort")
    if not table or not table.tbody:
        log.error("Could not find team table on page.")
        return {}

    rowers = {}
    for tr in table.tbody.find_all("tr"):
        name_cell = tr.find("td")
        if not name_cell:
            continue
        a = name_cell.find("a", href=True)
        if not a or "profile" not in a["href"]:
            continue
        partner_id = a["href"].split("/")[-1]
        name = a.get_text(strip=True)
        rowers[partner_id] = {"partner_id": partner_id, "name": name}
    log.info("Active rowers this season: %d", len(rowers))
    return rowers

def fetch_data(api_endpoint: str, sess: requests.Session | None = None):
    s = sess or session
    try:
        r = s.get(api_endpoint, timeout=(5, 30))  # (connect, read)
        if r.status_code == 200:
            return r.json()
        log.error("HTTP %s on %s: %s", r.status_code, api_endpoint, r.text[:200])
    except requests.RequestException:
        log.exception("HTTP error on %s", api_endpoint)
    return None


def fetch_all_results_paginated(session: requests.Session, rower_id: str, date_from: str, date_to: str) -> list[dict]:
    base = (f"https://log.concept2.com/api/users/{rower_id}/results"
            f"?from={date_from}&to={date_to}&number={PER_PAGE}")

    all_results: list[dict] = []


    page = 1
    next_url = f"{base}&page={page}"

    while next_url:
        data = fetch_data(next_url, sess=session)
        if not data or "data" not in data:
            break

        results_this_page = data["data"] or []
        all_results.extend(results_this_page)
        meta = (data.get("meta") or {}).get("pagination") or {}
        current = meta.get("current_page", page)
        total_pages = meta.get("total_pages")
        links = meta.get("links") or {}
        next_link = links.get("next")
        
        if next_link:
            next_url = next_link
            page = current + 1
        else:
            if total_pages and current < total_pages:
                page = current + 1
                next_url = f"{base}&page={page}"
            else:
                next_url = None

        if next_url:
            time.sleep(PAGE_SLEEP_SEC + random.uniform(JITTER_MIN, JITTER_MAX))

    return all_results

# =========================
# Helper transforms
# =========================
def build_date_range(latest_date: str | None) -> tuple[str, str]:
    """
    Start the next fetch the day after the latest saved date for that rower,
    clamped to DEFAULT_FROM..DEFAULT_TO.
    """
    if latest_date:
        try:
            start = (datetime.fromisoformat(str(latest_date)) + timedelta(days=1)).date().isoformat()
        except Exception:
            start = DEFAULT_FROM
        start = max(start, DEFAULT_FROM)
    else:
        start = DEFAULT_FROM
    return (start, DEFAULT_TO)

def maybe_fetch_strokes(sess: requests.Session, rower_id: str, training_id: str, stroke_data_flag: bool) -> str | None:
    if not stroke_data_flag:
        return None
    url = f"https://log.concept2.com/api/users/{rower_id}/results/{training_id}/strokes"
    with STROKE_SEM:
        time.sleep(0.2 + random.uniform(JITTER_MIN, JITTER_MAX))  # respect rate limits
        single = fetch_data(url, sess=sess)
        return json.dumps(single) if single else None

def build_workout_tuple(rower_id: str, name: str, w: dict, stroke_json: str | None) -> tuple:
    date_str = w.get("date", "1970-01-01 00:00:00")
    parts = date_str.split(" ")
    date_only = parts[0] if len(parts) > 0 else "1970-01-01"
    time_only = parts[1] if len(parts) > 1 else "00:00:00"

    # weekday
    try:
        day_of_week = datetime.strptime(date_only, "%Y-%m-%d").strftime("%A")
    except Exception:
        day_of_week = "Unknown"

    # duration to seconds
    time_text = w.get("time", "00:00:00")
    try:
        h, m, s = map(int, time_text.split(":"))
        time_seconds = h * 3600 + m * 60 + s
    except Exception:
        time_seconds = 0


    # build a stable stroke blob with optional intervals meta
    sd_wrapped = None
    if stroke_json:
        try:
            strokes_list = json.loads(stroke_json) or []
        except Exception:
            strokes_list = []
        wk = (w.get("workout") or {})
        intervals_meta = wk.get("intervals") or wk.get("splits") or []
        # normalize to a list of dicts we can index by interval
        if isinstance(intervals_meta, dict):
            intervals_meta = []
        sd_wrapped = json.dumps({
            "data": strokes_list,
            "intervals_meta": intervals_meta
        })
    else:
        sd_wrapped = None


    return (
        rower_id,
        w.get("id"),
        name,
        w.get("type"),
        w.get("distance", 0),
        date_only,
        day_of_week,
        time_only,
        time_seconds,
        w.get("timezone", "UTC"),
        w.get("date_utc"),
        (w.get("heart_rate") or {}).get("average", 0),
        w.get("calories_total", 0),
        sd_wrapped,
    )

# =========================
# Interleaved work planning
# =========================
def round_robin_interleave(dict_of_lists: dict) -> list[dict]:
    """
    dict_of_lists: {rower_id: [workout_dict, ...]}
    returns a single list interleaving: r1w1, r2w1, r3w1, r1w2, r2w2, ...
    """
    queues = {k: deque(v) for k, v in dict_of_lists.items() if v}
    result: list[dict] = []
    while queues:
        for rid in list(queues.keys()):
            q = queues[rid]
            if q:
                result.append(q.popleft())
            if not q:
                queues.pop(rid, None)
    return result

def list_new_workouts_for_rower(sess: requests.Session, rower: dict, date_from: str, date_to: str, existing_ids: set[str]) -> list[dict]:
    # enumerate list (cheap); no stroke fetch here
    items = fetch_all_results_paginated(sess, rower["partner_id"], date_from, date_to) or []
    fresh = [w for w in items if w.get("id") and w["id"] not in existing_ids and "distance" in w]
    for w in fresh:
        w["_rower_id"] = rower["partner_id"]
        w["_rower_name"] = rower["name"]
    return fresh

def plan_global_worklist(sess: requests.Session, rowers: dict, cursor) -> list[dict]:
    per_rower_lists: dict[str, list[dict]] = {}
    for _, rower in rowers.items():
        latest_date = get_latest_workout_date(cursor, rower["partner_id"])
        date_from, date_to = build_date_range(latest_date)
        existing_ids = get_existing_workout_ids(cursor, rower["partner_id"])
        per_rower_lists[rower["partner_id"]] = list_new_workouts_for_rower(sess, rower, date_from, date_to, existing_ids)
    global_worklist = round_robin_interleave(per_rower_lists)
    return global_worklist

def build_rower_semaphores(rowers: dict) -> dict[str, Semaphore]:
    return {r["partner_id"]: Semaphore(ROWER_PARALLELISM) for _, r in rowers.items()}

def fetch_one_workout(session_factory, rower_sems: dict[str, Semaphore], workout_dict: dict) -> tuple | None:
    # one session per thread (good keep-alive reuse)
    sess = session_factory()
    rid = workout_dict["_rower_id"]
    sem = rower_sems[rid]
    try:
        with sem:
            stroke_json = maybe_fetch_strokes(sess, rid, workout_dict["id"], workout_dict.get("stroke_data", False))
        return build_workout_tuple(rid, workout_dict["_rower_name"], workout_dict, stroke_json)
    finally:
        try:
            sess.close()
        except Exception:
            pass

def process_global_worklist(global_worklist: list[dict], rowers: dict) -> list[tuple]:
    rower_sems = build_rower_semaphores(rowers)

    def session_factory():
        s = make_session()
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=Retry(total=3, backoff_factor=0.3))
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    results: list[tuple] = []
    with ThreadPoolExecutor(max_workers=GLOBAL_WORKERS) as ex:
        futs = [ex.submit(fetch_one_workout, session_factory, rower_sems, w) for w in global_worklist]
        for fut in as_completed(futs):
            try:
                tup = fut.result()
                if tup:
                    results.append(tup)
            except Exception as e:
                log.exception("workout fetch failed: %s", e)
    return results

# =========================
# Main
# =========================
connection = sqlite3.connect('team_data.db')
cursor = setup_db(connection)

def main():
    start_time = time.time()
    try:
        # 1) Login + get rowers
        rowers = login_and_get_rowers(session)
        if not rowers:
            raise SystemExit("No rowers found; aborting.")

        # Upsert rowers
        for _, rower in rowers.items():
            insert_rower(connection, cursor, rower)

        # 2) PLAN (enumerate only new workouts per rower, no strokes)
        global_worklist = plan_global_worklist(session, rowers, cursor)
        log.info("Planned %d new workouts (interleaved across %d rowers).", len(global_worklist), len(rowers))
        if not global_worklist:
            log.info("No new workouts to fetch. Done.")
            return

        # 3) EXECUTE (fetch strokes interleaved across rowers)
        tuples = process_global_worklist(global_worklist, rowers)
        log.info("Fetched %d workouts.", len(tuples))

        # 4) INSERT (chunked)
        inserted = insert_workouts_chunked(connection, cursor, tuples, chunk=500)
        if inserted > 0:
            log.info("✅ Inserted %d workouts.", inserted)
        else:
            log.info("No new rows inserted (all duplicates).")

        elapsed = time.time() - start_time
        log.info("Script ran successfully! Elapsed time: %.2fs", elapsed)

    finally:
        try:
            connection.close()
            log.info("SQLite connection closed.")
        except Exception:
            log.exception("Error while closing SQLite connection")

        try:
            session.close()
            log.info("HTTP session closed.")
        except Exception:
            log.exception("Error while closing HTTP session")




if __name__ == "__main__":
    main()
