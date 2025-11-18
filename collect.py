import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from pathlib import Path
from typing import Dict, Any, List

# import the helpers that already live in review_data.py
from review_data import (
    make_session,
    login_and_get_rowers,
    fetch_all_results_paginated,
    maybe_fetch_strokes,
)

JITTER_MIN, JITTER_MAX = 0.05, 0.20

def fetch_result_detail(sess, rower_id: str, workout_id: int) -> dict | None:
    url = f"https://log.concept2.com/api/users/{rower_id}/results/{workout_id}"
    try:
        r = sess.get(url, timeout=(5, 30))
        if r.status_code == 200:
            return r.json()
        print(f"[detail] HTTP {r.status_code} id={workout_id} text={r.text[:200]}")
    except Exception as e:
        print(f"[detail] error id={workout_id}: {e}")
    return None

def build_workout_record(summary_obj: dict, detail_obj: dict | None, strokes_json: str | None, rower_id: str) -> dict:
    out = dict(summary_obj)
    out["_rower_id"] = rower_id              # ensure we can route it back to the rower
    out["detail"] = detail_obj if detail_obj is not None else None
    out["strokes"] = json.loads(strokes_json) if strokes_json else None
    return out

def _per_workout_job(sess_factory, rower_id: str, summary_obj: dict, sem: Semaphore) -> dict:
    sess = sess_factory()
    try:
        with sem:
            detail = fetch_result_detail(sess, rower_id, summary_obj["id"])
            time.sleep(0.15 + random.uniform(JITTER_MIN, JITTER_MAX))
            stroke_flag = bool(summary_obj.get("stroke_data", False))
            strokes_json = None
            if stroke_flag:
                strokes_json = maybe_fetch_strokes(sess, rower_id, summary_obj["id"], stroke_flag)
        return build_workout_record(summary_obj, detail, strokes_json, rower_id)
    finally:
        try:
            sess.close()
        except Exception:
            pass

def collect_team_workouts(date_from: str, date_to: str) -> Dict[str, Any]:
    start = time.time()
    root = {"date_from": date_from, "date_to": date_to, "rowers": []}

    listing_session = make_session()
    rowers = login_and_get_rowers(listing_session) or {}

    per_rower_sems: Dict[str, Semaphore] = {rid: Semaphore(2) for rid in rowers.keys()}
    per_rower_workouts: Dict[str, List[dict]] = {rid: [] for rid in rowers.keys()}

    futures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for rid, info in rowers.items():
            listings = fetch_all_results_paginated(listing_session, rid, date_from, date_to) or []
            for summary_obj in listings:
                if not summary_obj.get("id"):
                    continue
                futures.append(
                    pool.submit(_per_workout_job, make_session, rid, summary_obj, per_rower_sems[rid])
                )

        for f in as_completed(futures):
            try:
                rec = f.result()
                rid = rec.get("_rower_id")
                if rid:
                    per_rower_workouts.setdefault(rid, []).append(rec)
            except Exception as e:
                print(f"[worker] error collecting result: {e}")

    for rid, info in rowers.items():
        root["rowers"].append({
            "partner_id": rid,
            "name": info["name"],
            "workouts": per_rower_workouts.get(rid, [])
        })

    print(f"Collected in {time.time()-start:.1f}s for {date_from} → {date_to}; rowers={len(root['rowers'])}")
    return root

def main():
    date_from = "2025-10-27"
    date_to   = "2025-11-01"
    data = collect_team_workouts(date_from, date_to)
    out = Path(f"team_workouts_{date_from}_to_{date_to}.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Wrote {out.resolve()}")

if __name__ == "__main__":
    main()
