import time
import json
import random
import logging
from datetime import datetime, timedelta
import sqlite3
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter                      
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed #Allows you to download multiple workouts at the same time.
from threading import Semaphore                                 #Limits how many downloads happen simultaneously.
from collections import deque
import config

access_token = config.access_token

BATCH_SIZE = 10                                                 #insert 10 workouts at a time into db
PER_PAGE = 250                                                  #better get too many than make too many requests
PAGE_SLEEP_SEC = 0.25                                           #pause between requests
JITTER_MIN, JITTER_MAX = 0.05, 0.20                             #humanizes requests
ROWER_PARALLELISM = 2                                           #ensures we don't download more than 2 workouts at a time for the same rower at a time
GLOBAL_WORKERS     = 16                                         #16 workers in threadpoolexecutor
STROKE_CONCURRENCY = 3                                          # three stroke API's at the same time
STROKE_SEM = Semaphore(STROKE_CONCURRENCY)                      #regulates stroke request traffic

DEFAULT_FROM = "2025-10-21"                            
DEFAULT_TO   = "2026-03-01"

#logging setup - prints with timestamps
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("concept2")

 #establish a reusable internet connection
def make_session():         

    session = requests.Session()                                         
    session.headers.update({'Authorization': f'Bearer {access_token}'})
    retry_settings = Retry( total=5, backoff_factor=0.5,                                          #wait twice as much as previous time before trying again
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"), respect_retry_after_header=True )
    adapter = HTTPAdapter(max_retries=retry_settings)                                             #place the retry logic into the session
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

#DATABASE LOGIC
def setup_database(conn):
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

def insert_rower(conn,cursor,rower):
    sql = '''
    INSERT INTO crnjakt_rowers (partner_id, name)
    VALUES (?, ?)
    ON CONFLICT(partner_id) DO UPDATE SET
        name = excluded.name
    '''
    cursor.execute(sql, (rower['partner_id'], rower['name']))
    conn.commit()


def login_and_get_rowers(session):
    login_url = "https://log.concept2.com/login"
    partners_url = "https://log.concept2.com/team/" + config.my_team

    resp = session.post(login_url, data=config.login_payload, allow_redirects=True)

    if not resp.ok:
        log.error("Login failed. status=%s", resp.status_code)
        return {}
    
    log.info("Login successful.")
    resp = session.get(partners_url)

    #scrape from website
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

#get a workout
def get_api_json(api_endpoint, session=None):
    s = session
    try:
        r = session.get(api_endpoint,timeout=(5,30))
        if r.status_code == 200:
            return r.json()
        log.error("HTTP %s on %s: %s", r.status_code, api_endpoint, r.text[:200])
    except requests.RequestException:
        log.exception("HTTP error on %s", api_endpoint)
    return None


def get_all_workout_summaries(session, rower_id, date_from, date_to):

    base = (f"https://log.concept2.com/api/users/{rower_id}/results"
            f"?from={date_from}&to={date_to}&number={PER_PAGE}")
    
    all_results: list[dict] = []

    page = 1
    next_url = f"{base}&page={page}"

    while next_url:
            data = get_api_json(next_url, sess=session)
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


def get_existing_workout_ids(cursor,rower_id):
    cursor.execute( "SELECT workout_id FROM crnjakt_workouts WHERE rower_id = ?",
                   (rower_id))
    rows = cursor.fetchall()
    return {row[0] for row in rows}


def get_latest_workout_date(cursor, rower_id):
    cursor.execute("SELECT MAX(date) FROM crnjakt_workouts WHERE rower_id = ?",
        (rower_id,))
    result = cursor.fetchone()
    return result[0] 

def build_date_range(latest_date):
    if latest_date:
        start=(datetime.fromisoformat(latest_date)).strftime("%Y-%m-%d")
        end = DEFAULT_TO
    else:
        start=DEFAULT_FROM
        end=DEFAULT_TO
    return start, end

def list_new_workouts_for_rower(session,cursor,rower_id,rower_name):
    latest_date = get_latest_workout_date(cursor,rower_id)
    start,end = build_date_range(latest_date)

    all_workouts = get_all_workout_summaries(session,rower_id,start,end)
    existing_ids = get_existing_workout_ids(cursor,rower_id)

    new_workouts = []

    for workout in all_workouts:
        workout_id = str(workout.get("id"))
        if workout_id not in existing_ids:
            workout["_rower_id"]=rower_id
            workout["_rower_name"] = rower_name
            new_workouts.append(workout)

    return new_workouts


def round_robiN_interleave(session,cursor,rowers):
    rower_to_new={}
    
    for rower_id, info in rowers.items():
        name=info['name']
        
        new_for_rower=list_new_workouts_for_rower(session,cursor,rower_id)