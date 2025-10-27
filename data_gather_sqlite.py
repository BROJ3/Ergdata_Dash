import requests
import time
from bs4 import BeautifulSoup
import sqlite3
import json
from datetime import datetime
import config
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import random
from threading import Semaphore


# overall configuration

BATCH_SIZE = 20

# API paging
PER_PAGE = 250
PAGE_SLEEP_SEC = 0.25  # polite delay between result pages

STROKE_CONCURRENCY = 3
STROKE_SEM = Semaphore(3)

JITTER_MIN, JITTER_MAX = 0.05,0.20

#season 2024
#DEFAULT_FROM = "2024-11-01"
#DEFAULT_TO   = "2025-03-01"

#season 25
DEFAULT_FROM = "2025-10-21"
DEFAULT_TO   = "2026-03-01"


#set up logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("concept2")


access_token = config.access_token  

#Create a requests.Session with auth, retries, and backoff
def make_session() -> requests.Session:
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

# global session for main thread use
session = make_session()


#  Login to Concept2 and scrape the team roster table
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
    
    #get all rowers from table
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

        # Key by partner_id to avoid collisions on same-name rowers
        rowers[partner_id] = {"partner_id": partner_id, "name": name}
    return rowers



def build_date_range(latest_date: str | None) -> tuple[str, str]:
    # Keep your current hardcoded windows, but centralized.
    return (DEFAULT_FROM, DEFAULT_TO)

def maybe_fetch_strokes(session, rower_id: str, training_id: str, stroke_data_flag: bool) -> str | None:
    if not stroke_data_flag:
        return None
    single_url = f"https://log.concept2.com/api/users/{rower_id}/results/{training_id}/strokes"

    with STROKE_SEM:
        time.sleep(0.2+random.uniform(JITTER_MIN, JITTER_MAX))  # respect rate limits
    
    
        single = fetch_data(single_url, sess=session)
        return json.dumps(single) if single else None



def build_workout_tuple(rower_id: str, name: str, w: dict, stroke_json: str | None) -> tuple:
    date_str = w.get("date", "1970-01-01 00:00:00")
    parts = date_str.split(" ")
    date_only = parts[0] if len(parts) > 0 else "1970-01-01"
    time_only = parts[1] if len(parts) > 1 else "00:00:00"

    date_obj = datetime.strptime(date_only, "%Y-%m-%d")
    day_of_week = date_obj.strftime("%A")

    time_text = w.get("time", "00:00:00")

    try:
        h, m, s = map(int, time_text.split(":"))
        time_seconds = h * 3600 + m * 60 + s
    except Exception:
        time_seconds = 0 
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
        w.get("heart_rate", {}).get("average", 0),
        w.get("calories_total", 0),
        stroke_json,
    )


# Connect to SQLite database
def setup_db(conn: sqlite3.Connection) -> sqlite3.Cursor:
    cur = conn.cursor()
    cur.execute('PRAGMA foreign_keys = ON')

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

#use upsert
def insert_rower(conn, cursor, rower):

    sql = '''
    INSERT INTO crnjakt_rowers (partner_id, name)
    VALUES (?, ?)
    ON CONFLICT(partner_id) DO UPDATE SET
        name = excluded.name
    '''
    
    with conn:
        cursor.execute(sql, (rower['partner_id'], rower['name']))


def insert_workout(conn, cursor, workouts):
  
    if not workouts:
        return 0
    sql = '''
    INSERT INTO crnjakt_workouts (
        rower_id, workout_id, name, type, distance, date, weekday, hour,
        time, timezone, date_utc, heart_rate, calories_total, stroke_data
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(workout_id) DO NOTHING
    '''
    # track how many were inserted
    before = conn.total_changes
    with conn:  
        cursor.executemany(sql, workouts)
    return conn.total_changes - before


def get_latest_workout_date(cursor, partner_id):
    sql = "SELECT MAX(date) as latest_date FROM crnjakt_workouts WHERE rower_id = ?"
    cursor.execute(sql, (partner_id,))
    result = cursor.fetchone()
    return result[0]


access_token = config.access_token 

def fetch_data(api_endpoint: str, sess: requests.Session | None = None):
    """GET JSON with a hardened session (auth, retries, timeouts)."""
    s = sess or session
    try:
        r = s.get(api_endpoint, timeout=(5, 30))  # (connect, read)
        if r.status_code == 200:
            return r.json()
        log.error("HTTP %s on %s: %s", r.status_code, api_endpoint, r.text[:200])
    except requests.RequestException:
        log.exception("HTTP error on %s", api_endpoint)
    return None

#fetch all workouts
def fetch_all_results_paginated(session: requests.Session, rower_id: str, date_from: str, date_to: str) -> list[dict]:

    base = (f"https://log.concept2.com/api/users/{rower_id}/results"
            f"?from={date_from}&to={date_to}&number={PER_PAGE}")
    
    all_results:list[dict] = []
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

        #iterating through the pages if result is too large
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


#fetch distinct workouts with strokes
def fetch_and_build_for_rower(rower: dict, date_from: str, date_to: str) -> List[Tuple]:

    workout_rows: List[Tuple] = []
    worker_session = make_session()

    try:
        rid = rower["partner_id"]
        all_results = fetch_all_results_paginated(worker_session, rid, date_from, date_to)

        if not all_results:
            log.warning("No workouts found for %s", rower["name"])
            return workout_rows

        for w in all_results:
            tid = w.get("id")
            if not tid or "distance" not in w:
                continue
            stroke_json = maybe_fetch_strokes(worker_session, rid, tid, w.get("stroke_data", False))
            workout_rows.append(build_workout_tuple(rid, rower["name"], w, stroke_json))
        return workout_rows
    except Exception:
        log.exception("Worker failed for %s", rower.get("name"))
        return workout_rows
    finally:
        try:
            worker_session.close()
        except Exception:
            log.exception("Error closing worker session")

connection = sqlite3.connect('team_data.db')
cursor = setup_db(connection)

def main():

    start_time = time.time()
    try:
        # 1) Login + get rowers
        rowers = login_and_get_rowers(session)  # dict keyed by partner_id
        if not rowers:
            raise SystemExit("No rowers found; aborting.")

        # 2) Upsert rowers first so FK inserts are valid
        for _, rower in rowers.items():
            insert_rower(connection, cursor, rower)

        date_from, date_to = build_date_range(None)  # or hardcode your window

        MAX_WORKERS = 8 # tune 4–8 based on rate limits / network

        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for _, rower in rowers.items():

                latest_date = get_latest_workout_date(cursor, rower["partner_id"])
                log.info("%s has latest record on: %s", rower["name"], latest_date)
                futures[ex.submit(fetch_and_build_for_rower, rower, date_from, date_to)] = rower

            for fut in as_completed(futures):
                rower = futures[fut]
                name = rower["name"]
                try:
                    rows = fut.result()  
                except Exception:
                    log.exception("Worker future failed for %s", name)
                    continue

                if not rows:
                    continue

                batch = []
                for t in rows:
                    batch.append(t)
                    if len(batch) >= BATCH_SIZE:
                        inserted = insert_workout(connection, cursor, batch)
                        log.info("✅ Batch %s (new: %s) for %s", len(batch), inserted, name)
                        batch.clear()

                if batch:
                    inserted = insert_workout(connection, cursor, batch)
                    log.info("✅ Remaining %s (new: %s) for %s", len(batch), inserted, name)

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