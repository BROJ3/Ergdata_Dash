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

# API paging
PER_PAGE = 250
PAGE_SLEEP_SEC = 0.25  # polite delay between result pages

access_token = 'Cu8hdZo8VTmwy8EeOzSGkE5X8GCMCcZd9dnWPzgd' 

JITTER_MIN, JITTER_MAX = 0.05, 0.20


login_payload = {
    "username": "ToniCrnjak",  
    "password": "Rafael0!3500"   
}

my_team='19989' 

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


def login_and_get_rowers(session) -> dict:
    login_url = "https://log.concept2.com/login"
    partners_url = "https://log.concept2.com/team/" + config.my_team
    resp = session.post(login_url, data=config.login_payload, allow_redirects=True)
    if not resp.ok:
        print("Login failed. status=%s", resp.status_code)
        return {}

    print("Login successful.")
    resp = session.get(partners_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="table js-tablesort")
    if not table or not table.tbody:
        print("Could not find team table on page.")
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
    print("Active rowers this season: %d", len(rowers))
    return rowers


def fetch_data(api_endpoint: str, sess: requests.Session | None = None):
    s = sess or session
    try:
        r = s.get(api_endpoint, timeout=(5, 30))  # (connect, read)
        if r.status_code == 200:
            return r.json()
        print("HTTP %s on %s: %s", r.status_code, api_endpoint, r.text[:200])
    except requests.RequestException:
        print("HTTP error on %s", api_endpoint)
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


def maybe_fetch_strokes(sess: requests.Session, rower_id: str, training_id: str, stroke_data_flag: bool) -> str | None:
    if not stroke_data_flag:
        return None
    url = f"https://log.concept2.com/api/users/{rower_id}/results/{training_id}/strokes"

    time.sleep(0.2 + random.uniform(JITTER_MIN, JITTER_MAX))  # respect rate limits
    single = fetch_data(url, sess=sess)

    return json.dumps(single) if single else None



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


def list_new_workouts_for_rower(sess: requests.Session, rower: dict, date_from: str, date_to: str, existing_ids: set[str]) -> list[dict]:
    # enumerate list (cheap); no stroke fetch here
    items = fetch_all_results_paginated(sess, rower["partner_id"], date_from, date_to) or []
    fresh = [w for w in items if w.get("id") and w["id"] not in existing_ids and "distance" in w]
    for w in fresh:
        w["_rower_id"] = rower["partner_id"]
        w["_rower_name"] = rower["name"]
    return fresh



DEFAULT_FROM = "2025-10-21"
DEFAULT_TO   = "2026-03-01"


def main():

    rowers = login_and_get_rowers(session)
    #print(rowers)

    per_rower_lists: dict[str, list[dict]] = {}

    for _, rower in rowers.items():
        date_from, date_to = DEFAULT_FROM, DEFAULT_TO

        base = (f"https://log.concept2.com/api/users/{_}/results"
            f"?from={date_from}&to={date_to}&number={PER_PAGE}")

        all_results: list[dict] = []

        data = fetch_data(base, sess=session)
        results_this_page = data["data"] or []
        all_results.extend(results_this_page)


#main()