# audit_strokes.py
import sqlite3, json, sys, pathlib
import pandas as pd

DB = sys.argv[1] if len(sys.argv) > 1 else "team_data.db"
out_dir = pathlib.Path("audit_out"); out_dir.mkdir(exist_ok=True)

def parse_json_maybe(x):
    if x is None: return None
    if isinstance(x, (dict, list)): return x
    if isinstance(x, (bytes, bytearray)): x = x.decode("utf-8", "ignore")
    if isinstance(x, str):
        try: return json.loads(x)
        except json.JSONDecodeError: return x   # leave as raw string if broken
    return x

con = sqlite3.connect(DB)
df = pd.read_sql_query("""
    SELECT rowid AS id, name, date, stroke_data
    FROM crnjakt_workouts
    ORDER BY date DESC
""", con)

# Normalize just enough to inspect
df["sd"] = df["stroke_data"].apply(parse_json_maybe)

# Flags so we can count shapes
def classify(sd):
    if sd is None: return "None"
    if isinstance(sd, str): return "STRING_sd"
    if isinstance(sd, dict):
        d = sd.get("data")
        if isinstance(d, list): return "DICT_list"
        if isinstance(d, str):  return "DICT_STRING_data"
        return f"DICT_{type(d).__name__}"
    if isinstance(sd, list): return "LIST_root"
    return type(sd).__name__

df["shape"] = df["sd"].apply(classify)

print("\n==== stroke_data shape counts ====")
print(df["shape"].value_counts(dropna=False), "\n")

# Write samples of each shape to JSONL so you can eyeball them
for shape, sub in df.groupby("shape"):
    sample_path = out_dir / f"samples_{shape}.jsonl"
    with sample_path.open("w", encoding="utf-8") as f:
        for _, r in sub.head(10).iterrows():   # 10 examples per shape
            rec = {
                "id": int(r["id"]),
                "name": r["name"],
                "date": r["date"],
                "stroke_data": r["sd"],        # already parsed when possible
            }
            try:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except TypeError:
                # If something’s still not JSON-serializable, dump as string
                rec["stroke_data"] = str(r["sd"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {min(10, len(sub))} examples for {shape} -> {sample_path}")

# Also dump a full list (IDs + quick summary) for fast scanning
def quick(sd):
    if sd is None: return None
    if isinstance(sd, str): return f"STRING({len(sd)} chars)"
    if isinstance(sd, dict):
        d = sd.get("data")
        if isinstance(d, list): return f"DICT(list, {len(d)} pts)"
        if isinstance(d, str):  return f"DICT(STRING data, {len(d)} chars)"
        return f"DICT({type(d).__name__})"
    if isinstance(sd, list): return f"LIST_root({len(sd)} items)"
    return type(sd).__name__

summary_cols = df[["id","name","date","shape"]].copy()
summary_cols["quick"] = df["sd"].apply(quick)
summary_path = out_dir / "stroke_data_summary.csv"
summary_cols.to_csv(summary_path, index=False)
print(f"\nSummary -> {summary_path}")
