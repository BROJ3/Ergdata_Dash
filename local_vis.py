import sqlite3
import plotly.express as px
import dash
from dash import dcc, html, no_update, ctx
from dash.dependencies import Input, Output, State
from datetime import datetime
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go




COACHES = {"Toni Crnjak", "Boris Jukic", "Martijn Ronk"}

def apply_filters(df: pd.DataFrame, flt: dict, *, force_allwinter: bool | None = None) -> pd.DataFrame:
    sub = df

    # ----- date range -----
    if flt.get("date"):
        end = pd.to_datetime(flt["date"][1]) if flt["date"][1] else pd.Timestamp.today().normalize()

        # if caller forces behavior, use that; otherwise use the stored flag
        aw = force_allwinter if force_allwinter is not None else flt.get("allwinter", False)

        if aw:
            start = pd.to_datetime(flt["date"][0])        # picker start (“winter start”)
        else:
            start = (pd.to_datetime(end) - pd.Timedelta(days=14)).normalize()

        sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]

    # ----- rower names -----
    if flt.get("name"):
        sub = sub[sub["name"].isin(flt["name"])]

    # ----- include coaches toggle (keep your existing logic if you have one) -----
    # if not flt.get("include_coaches", False):
    #     sub = sub[~sub["name"].isin(COACHES)]

    return sub.copy()



#loading into dataframe
connection = sqlite3.connect('team_data.db')

df = pd.read_sql_query( 
    """
    SELECT name, distance, date, weekday, hour, time, stroke_data
    FROM crnjakt_workouts
    """, 
    connection,
    parse_dates=["date"]
)

df["distance"] = df["distance"].astype(float)
df["hour"]=pd.to_datetime(df["hour"], format = "%H:%M:%S").dt.hour
df["time"] = df["time"].astype(int)


# After df is created and cleaned
ALL_ROWERS = sorted(df["name"].unique())

# pick a qualitative palette
BASE_COLORS = px.colors.qualitative.Plotly #Plotly  # or .Bold, .Safe, etc.

ROWER_COLOR_MAP = {
    name: BASE_COLORS[i % len(BASE_COLORS)]
    for i, name in enumerate(ALL_ROWERS)
}


#parse the json if workour has stroke data
def parse_json_maybe(js):
    try:
        return json.loads(js) if js else None
    except Exception:
        return None

df["stroke_data"] = df["stroke_data"].apply(parse_json_maybe)

df=df.sort_values(["name","date"])


df["cumulative_distance"] = df.groupby("name")["distance"].cumsum()
#cumulative_df = df[["name", "date", "cumulative_distance"]]

def tod(h):
    if 5<= h < 11:
        return "Morning"
    if 11<= h < 17:
        return "Midday"
    if 17<= h < 23:
        return "Evening"
    return "Night"

df["time_of_day"] = df["hour"].apply(tod)


weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
df["weekday"] = pd.Categorical(df["weekday"], categories=weekday_order, ordered=True)

'''distance_by_weekday = (
    df.groupby("weekday", as_index=False, observed=False)["distance"]
      .sum()
      .sort_values("weekday")
)
'''

START = pd.Timestamp("2024-11-01")
df["week_num"] = ((df["date"] - START).dt.days // 7) + 1



app = dash.Dash(__name__)
server = app.server

rowers_with_strokes = (
    df[df["stroke_data"].notna()]["name"].drop_duplicates()
    .sort_values()
    .tolist()
    )
    
rower_options = [{'label': r, 'value': r} for r in rowers_with_strokes]


app.layout = html.Div(
    className="dashboard-container",
    children=[

        dcc.Store(
            id="filters",
            data={
                "date": [
                    df["date"].min().date().isoformat(),
                    df["date"].max().date().isoformat()
                ],
                "name": []   # empty means select all
            }
        ),

        html.H1("Clarkson Crew Winter Meters", className="dashboard-header"),

        html.Div(
    className="filter-row",
    style={"display": "flex", "gap": "16px", "alignItems": "flex-end", "flexWrap": "wrap"},
    children=[
        html.Div([
            html.Label("Start date"),
            dcc.DatePickerSingle(
                id="date-start",
                date=df["date"].min().date(),
                display_format="YYYY-MM-DD"
            )
        ]),
        html.Div([
            html.Label("End date"),
            dcc.DatePickerSingle(
                id="date-end",
                date=df["date"].max().date(),
                display_format="YYYY-MM-DD"
            )
        ]),
        html.Div(style={"minWidth": "280px"}, children=[
            html.Label("Filter rowers (multi)"),
            dcc.Dropdown(
                id="global-rower-filter",
                options=[{"label": n, "value": n} for n in sorted(df["name"].unique())],
                value=[],  # empty = all rowers
                multi=True,
                placeholder="All rowers"
            )
        ]),
        html.Button(
            "Clear filters",
            id="clear-filters",
            n_clicks=0,
            style={"height": "38px"}
        ),
                        html.Div(
                style={"margin":"8px 0 16px 0"},


                

                
                children=dcc.Checklist(
                    id="include-coaches",
                    options=[{"label": "Include coaches", "value": "yes"}],
                    value=[],  # empty -> coaches excluded by default
                    inputStyle={"marginRight":"6px"}
                        ),
                        
            ),

            html.Div([
            dcc.Checklist(
                id="show-all-winter",
                options=[{"label": "Show all winter", "value": "ALL"}],
                value=[],
                inputStyle={"marginRight": "6px"},
                style={"marginTop": "20px"}
            )
        ]),
            
    ]
),


        dcc.Graph(id='cumulative-distance-graph', className="graph", animate=True,animation_options={"frame": {"duration": 1200}, "transition": {"duration": 1200}}),
        html.Div(
            
            className="charts-row",
            children=[
                dcc.Graph(id='time-of-day-pie-chart', className="chart",animate=True,animation_options={"frame": {"duration": 1200}, "transition": {"duration": 1200}}),
                dcc.Graph(id='weekday-bar-chart', className="chart",animate=True,animation_options={"frame": {"duration": 1200}, "transition": {"duration": 1200}}),
            ]
        ),
        html.H2("Leaderboard", className="leaderboard-header"),



        html.Div(id="leaderboard-table", className="leaderboard-container"),


        html.Div(
            className="dropdown-row",
            children=[
                html.Div(
                    className="dropdown",
                    children=[
                        html.Label("Select Rower:"),
                        dcc.Dropdown(
                            id='rower-dropdown',
                            options=[{'label': r, 'value': r} for r in (
                                df[df["stroke_data"].notna()]["name"].drop_duplicates().sort_values().tolist()
                            )],
                            value=None
                        ),
                    ]
                ),
                html.Div(
                    className="dropdown",
                    children=[
                        html.Label("Select Workout:"),
                        dcc.Dropdown(id='workout-dropdown'),
                    ]
                ),
                
            html.Div(className="dropdown", children=[
                html.Label("Select Interval:"),
                dcc.Dropdown(id="interval-dropdown", placeholder="Interval 1", clearable=False)
            ])
            ]
        ),
        dcc.Graph(id='workout-stroke-graph'),
        html.Div(
            style={"width": "80%", "margin": "20px auto 40px"},
            children=[
                html.Label("Filter interval by distance (m):"),
                dcc.RangeSlider(
                    id="distance-range",
                    min=0,
                    max=1000,
                    step=50,
                    value=[0, 1000],
                    allowCross=False,
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="mouseup",
                    marks=None
                ),
            ],
        ),
        html.Div(
            id="summary-table",
            style={"width": "80%", "margin": "0 auto 40px"},
        ),
    ]
)

def _strokes_from_sd(sd: dict):

    if not isinstance(sd, dict):
        return []
    slot = sd.get("data", [])
    if isinstance(slot, dict) and isinstance(slot.get("data"), list):
        return slot["data"]
    return slot if isinstance(slot, list) else []


def _intervals_meta_from_sd(sd: dict):

    if not isinstance(sd, dict):
        return []
    meta = sd.get("intervals_meta")
    if meta:
        return meta
    slot = sd.get("data")
    if isinstance(slot, dict):
        return slot.get("intervals_meta") or slot.get("meta") or []
    return []


def split_strokes_by_time_reset(strokes):

    if not strokes:
        return []

    t_vals = [s.get("t", 0) for s in strokes]

    boundaries = [0]
    for i in range(len(t_vals) - 1):
        if t_vals[i + 1] < t_vals[i]:
            boundaries.append(i + 1)
    boundaries.append(len(strokes))

    segments = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        seg = strokes[a:b]
        if seg:
            segments.append(seg)
    return segments


def segments_for_sd(sd: dict):
    """
    Given a stroke_data payload (sd), return:
      segments: list of stroke segments (one per interval-like piece)
      intervals_meta: metadata list for intervals (if present)

    Special case:
    - Some single continuous pieces (e.g. 2000m with 500m splits) have:
        * strokes: one continuous time segment
        * intervals_meta: multiple split entries (500m, 500m, 500m, 500m)
      In that case we want to treat the whole thing as ONE interval, so we
      collapse the meta list into a single total (time + distance).
    """
    strokes = _strokes_from_sd(sd)
    segments = split_strokes_by_time_reset(strokes)
    intervals_meta = _intervals_meta_from_sd(sd)

    # If there is only one continuous stroke segment but multiple meta entries,
    # assume it's a continuous piece with splits and collapse the meta.
    if (
        isinstance(segments, list)
        and len(segments) == 1
        and isinstance(intervals_meta, list)
        and len(intervals_meta) > 1
    ):
        total_time = 0.0
        total_dist = 0.0
        has_time = False
        has_dist = False

        for m in intervals_meta:
            if not isinstance(m, dict):
                continue
            if "time" in m and m["time"] is not None:
                has_time = True
                total_time += float(m["time"])
            if "distance" in m and m["distance"] is not None:
                has_dist = True
                total_dist += float(m["distance"])

        combined = {}
        if has_time:
            combined["time"] = total_time
        if has_dist:
            combined["distance"] = total_dist

        # Replace with a single combined meta entry
        intervals_meta = [combined] if combined else []

    return segments, intervals_meta


def build_interval_dropdown_options(segments, intervals_meta):

    def fmt_time_ds(ds):
        if ds is None:
            return None
        sec = float(ds) / 10.0
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        else:
            return f"{m}:{s:02d}"

    options = []

    for i, seg in enumerate(segments):
        label = f"Interval {i + 1}"

        if isinstance(intervals_meta, list) and i < len(intervals_meta):
            meta = intervals_meta[i]  

            if isinstance(meta, dict):
                t = meta.get("time")       # deciseconds
                d = meta.get("distance")   # meters

                parts = []

                if t is not None:
                    t_formatted = fmt_time_ds(t)
                    if t_formatted:
                        parts.append(t_formatted)

                # Format distance
                if d is not None:
                    parts.append(f"{int(d)}m")

                if parts:
                    label += " (" + ", ".join(parts) + ")"

        options.append({"label": label, "value": i})

    return options



def format_pace(sec):
    if sec <= 0 or not np.isfinite(sec):
        return "-"
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:04.1f}"


def format_time_hms(sec):
    if sec is None or sec <= 0 or not np.isfinite(sec):
        return "-"
    sec = float(sec)
    h = int(sec // 3600)
    rem = sec - h * 3600
    m = int(rem // 60)
    s = rem - m * 60
    if h > 0:
        return f"{h}:{m:02d}:{s:04.1f}"
    else:
        return f"{m}:{s:04.1f}"


def prepare_segment_data(segment, interval_meta):
    """
    Same trimming/scaling logic as build_stroke_figure, but instead of
    building a figure we just return arrays for summaries & slider logic.
    Returns dict with keys: x, t, pace_s, hr, spm
    """
    if not segment:
        empty = np.array([], dtype=float)
        return dict(x=empty, t=empty, pace_s=empty, hr=empty, spm=empty)

    # ----- raw arrays -----
    t = np.array([s.get("t", np.nan) for s in segment], float)
    hr = np.array([s.get("hr", 0) for s in segment], float)
    spm = np.array([s.get("spm", 0) for s in segment], float)
    d = np.array([s.get("d", np.nan) for s in segment], float)
    p_raw = np.array([s.get("p", 0) for s in segment], float)
    pace_seconds = p_raw / 10.0

    # ----- trim to programmed work time, if present -----
    work_raw = None
    if isinstance(interval_meta, dict) and ("time" in interval_meta):
        work_raw = float(interval_meta["time"])

    if work_raw is not None and np.isfinite(work_raw):
        t0 = t[0] if np.isfinite(t[0]) else 0.0
        t_rel = t - t0
        mask = np.isfinite(t_rel) & (t_rel <= work_raw)
        if mask.any():
            last = int(np.max(np.nonzero(mask)[0]))
            t = t[: last + 1]
            hr = hr[: last + 1]
            spm = spm[: last + 1]
            d = d[: last + 1]
            pace_seconds = pace_seconds[: last + 1]

    # ----- trim very slow edge strokes (same idea as graph) -----
    valid_idx = np.where(np.isfinite(pace_seconds) & (pace_seconds > 0))[0]
    if valid_idx.size:
        valid_pace = pace_seconds[valid_idx]
        thr = float(np.nanpercentile(valid_pace, 98))  # KNOB for aggressiveness

        good = (
            np.isfinite(pace_seconds)
            & (pace_seconds > 0)
            & (pace_seconds <= thr)
        )

        if good.any():
            first_good = int(np.argmax(good))
            last_good = len(pace_seconds) - 1 - int(np.argmax(good[::-1]))
            if first_good > 0 or last_good < len(pace_seconds) - 1:
                sl = slice(first_good, last_good + 1)
                t = t[sl]
                hr = hr[sl]
                spm = spm[sl]
                d = d[sl]
                pace_seconds = pace_seconds[sl]

    # ----- distance axis (decimeters -> meters, start at 0) -----
    if np.isfinite(d).any():
        d_m = d / 10.0
        x = d_m - d_m[0]
    else:
        x = np.arange(len(t))

    return dict(x=x, t=t, pace_s=pace_seconds, hr=hr, spm=spm)


def compute_pace_from_time_distance(time_s, dist_m):
    if time_s is None or dist_m is None or time_s <= 0 or dist_m <= 0:
        return np.nan
    return (float(time_s) / float(dist_m)) * 500.0


def compute_watts_from_pace(pace_500_s):
    """
    Concept2 formula: watts = 2.8 / (pace/500)^3
    """
    if pace_500_s is None or pace_500_s <= 0 or not np.isfinite(pace_500_s):
        return np.nan
    return 2.8 / (pace_500_s / 500.0) ** 3


def compute_cal_per_hour(calories_total, time_s):
    if (calories_total is None or time_s is None or
            time_s <= 0 or not np.isfinite(time_s)):
        return np.nan
    return (float(calories_total) * 3600.0) / float(time_s)


def make_summary_row(label, time_s, dist_m, calories, stroke_rate, hr_avg):
    pace_500 = compute_pace_from_time_distance(time_s, dist_m)
    watts = compute_watts_from_pace(pace_500)
    cal_hr = compute_cal_per_hour(calories, time_s)

    return {
        "label": label,
        "time": format_time_hms(time_s),
        "meters": int(round(dist_m)) if dist_m is not None and np.isfinite(dist_m) else "-",
        "pace": format_pace(pace_500) if np.isfinite(pace_500) else "-",
        "spm": int(round(stroke_rate)) if stroke_rate is not None and np.isfinite(stroke_rate) else "-",
        "hr": int(round(hr_avg)) if hr_avg is not None and np.isfinite(hr_avg) else "-",
    }


def compute_summary_rows_local(selected_workout_str, which_interval, distance_range):
    """
    Build summary rows for local_vis:
      - Workout total (from df row, with fallback to intervals_meta)
      - One row per interval (from intervals_meta)
      - Current view (slider-filtered part of selected interval)
    """
    rows = []

    if not selected_workout_str:
        return rows

    # --- parse stroke_data JSON and get segments + interval meta ---
    sd = json.loads(selected_workout_str)
    segments, intervals_meta = segments_for_sd(sd)

    # --- find the df row for this workout (value is json.dumps(stroke_data)) ---
    mask = df["stroke_data"].notna() & df["stroke_data"].apply(
        lambda x: json.dumps(x) == selected_workout_str
    )
    wrow = df[mask].iloc[0] if mask.any() else None

    # ---------- WORKOUT TOTAL: time & distance ----------
    w_time = None
    w_dist = None

    if wrow is not None:
        # time from DB (may be seconds or deciseconds or missing)
        raw_w_time = wrow.get("time")
        if raw_w_time is not None:
            val = float(raw_w_time)
            # if it looks like deciseconds (very large), convert
            if val > 6 * 3600:  # > 6 hours of rowing is unlikely
                val = val / 10.0
            if val > 0:
                w_time = val

        # distance from DB
        raw_w_dist = wrow.get("distance")
        if raw_w_dist is not None:
            val = float(raw_w_dist)
            if val > 0:
                w_dist = val

    # totals from intervals_meta (time in deciseconds, distance in meters)
    total_ds = 0.0
    total_dist = 0.0
    if isinstance(intervals_meta, list):
        for m in intervals_meta:
            if not isinstance(m, dict):
                continue
            if m.get("time") is not None:
                total_ds += float(m["time"])
            if m.get("distance") is not None:
                total_dist += float(m["distance"])

    if (w_time is None or w_time <= 0 or not np.isfinite(w_time)) and total_ds > 0:
        w_time = total_ds / 10.0  # deciseconds → seconds
    if (w_dist is None or w_dist <= 0 or not np.isfinite(w_dist)) and total_dist > 0:
        w_dist = total_dist

    if w_time is None:
        w_time = 0.0
    if w_dist is None:
        w_dist = 0.0

    # ---------- WORKOUT TOTAL: avg SPM & HR ----------
    spm_total = None
    hr_total = None
    if segments:
        all_spm = []
        all_hr = []
        for seg in segments:
            for s in seg:
                s_spm = s.get("spm")
                s_hr = s.get("hr")
                if s_spm is not None and s_spm > 0:
                    all_spm.append(float(s_spm))
                if s_hr is not None and s_hr > 0:
                    all_hr.append(float(s_hr))
        if all_spm:
            spm_total = float(np.mean(all_spm))
        if all_hr:
            hr_total = float(np.mean(all_hr))

    # workout total row (now with S/M and HR, but same time/pace as before)
    rows.append(make_summary_row("Workout total", w_time, w_dist, None, spm_total, hr_total))

    # ---------- INTERVAL ROWS ----------
    for i, meta in enumerate(intervals_meta or []):
        if not isinstance(meta, dict):
            continue

        # time is in deciseconds → convert to seconds
        raw_t = meta.get("time")
        t_s = float(raw_t) / 10.0 if raw_t is not None else 0.0

        raw_d = meta.get("distance")
        d_m = float(raw_d) if raw_d is not None else 0.0

        c_tot = meta.get("calories_total")
        if c_tot is not None:
            c_tot = float(c_tot)

        sr = meta.get("stroke_rate")
        if sr is not None:
            sr = float(sr)

        hr_avg = None
        hr_meta = meta.get("heart_rate")
        if isinstance(hr_meta, dict):
            hr_avg = hr_meta.get("average")

        rows.append(
            make_summary_row(f"Interval {i + 1}", t_s, d_m, c_tot, sr, hr_avg)
        )

    # ---------- CURRENT VIEW ----------
    if segments:
        if which_interval is None or which_interval < 0 or which_interval >= len(segments):
            which_interval = 0

        seg = segments[which_interval]
        meta = (
            intervals_meta[which_interval]
            if isinstance(intervals_meta, list)
            and which_interval < len(intervals_meta)
            and isinstance(intervals_meta[which_interval], dict)
            else {}
        )

        data = prepare_segment_data(seg, meta)
        x = data["x"]
        t = data["t"]
        hr = data["hr"]
        spm = data["spm"]

        label = f"Selected Interval (Interval {which_interval + 1})"

        if x.size >= 2:
            if distance_range and len(distance_range) == 2:
                low, high = distance_range
                mask = (x >= low) & (x <= high)
                if mask.sum() >= 2:
                    x = x[mask]
                    t = t[mask]
                    hr = hr[mask]
                    spm = spm[mask]

            if x.size >= 2:
                dist_m = float(x[-1] - x[0])
                # t is in deciseconds → seconds
                time_s = float(t[-1] - t[0]) / 10.0
                hr_valid = hr[hr > 0]
                spm_valid = spm[spm > 0]
                hr_avg = float(np.mean(hr_valid)) if hr_valid.size > 0 else None
                spm_avg = float(np.mean(spm_valid)) if spm_valid.size > 0 else None

                rows.insert(
                    0,
                    make_summary_row(label, time_s, dist_m, None, spm_avg, hr_avg)
                )
    return rows



def build_stroke_figure(segment, title, interval_meta):
    """
    segment = list of strokes for this interval
    interval_meta = metadata for this interval (contains programmed work time).
    """
    data = prepare_segment_data(segment, interval_meta)
    x = data["x"]
    t = data["t"]
    pace_seconds = data["pace_s"]
    hr = data["hr"]
    spm = data["spm"]

    if x.size == 0:
        return go.Figure(layout_title_text=title)

    pace_labels = [format_pace(v) for v in pace_seconds]

    fig = go.Figure()

    # Pace trace
    fig.add_trace(go.Scatter(
        x=x, y=pace_seconds,
        name="Pace/500m",
        mode="lines",
        yaxis="y",
        line=dict(width=0.8),
        customdata=pace_labels,
        hovertemplate="Distance=%{x:.0f} m<br>Pace=%{customdata}<extra></extra>"
    ))

    # HR trace
    fig.add_trace(go.Scatter(
        x=x, y=hr,
        name="Heart Rate (bpm)",
        line=dict(width=0.8),
        mode="lines",
        yaxis="y2",
        hovertemplate="Distance=%{x:.0f} m<br>HR=%{y:.0f} bpm<extra></extra>"
    ))

    # SPM trace
    fig.add_trace(go.Scatter(
        x=x, y=spm,
        name="SPM",
        mode="lines",
        yaxis="y3",
        line=dict(width=0.8),
        hovertemplate="Distance=%{x:.0f} m<br>SPM=%{y:.0f}<extra></extra>"
    ))

    # ----- pace axis ticks (same logic you already had) -----
    idx_valid = np.where(np.isfinite(pace_seconds) & (pace_seconds > 0))[0]

    if idx_valid.size:
        if idx_valid.size > 6:
            core_idx = idx_valid[2:-2]
        else:
            core_idx = idx_valid

        core_pace = pace_seconds[core_idx]

        med = float(np.nanmedian(core_pace))
        p10 = float(np.nanpercentile(core_pace, 10))
        p90 = float(np.nanpercentile(core_pace, 90))

        half_span = max((p90 - p10), 12.0)
        pad = 5.0

        p_lo = med - half_span - pad
        p_hi = med + half_span + pad

        p_lo = max(p_lo, 80.0)
        p_hi = min(p_hi, 180.0)

        span = p_hi - p_lo

        nice = np.arange(80.0, 181.0, 1.0)
        ticks = nice[(nice >= p_lo) & (nice <= p_hi)]

        if ticks.size < 3:
            step = 1.0 if span <= 60.0 else 1.0
            ticks = np.arange(
                np.floor(p_lo / step) * step,
                np.ceil(p_hi / step) * step + 0.1,
                step
            )

        tick_text = [format_pace(v) for v in ticks]
    else:
        ticks = []
        tick_text = []

    fig.update_layout(
        title=title,
        xaxis=dict(title="Distance (m)"),
        yaxis=dict(
            title="Pace (mm:ss)",
            autorange="reversed",
            tickvals=ticks,
            ticktext=tick_text
        ),
        yaxis2=dict(
            title="Heart Rate (bpm)",
            overlaying="y",
            side="right",
            range=[0, 220],
            showgrid=False
        ),
        yaxis3=dict(
            title="SPM",
            overlaying="y",
            side="right",
            position=0.97,
            range=[0, 50],
            showgrid=False,
            showticklabels=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0
        ),
        margin=dict(t=60, r=160),
        height=600
    )

    return fig


def get_distance_axis_for_segment(segment, interval_meta, title="tmp"):
    """
    Build a temporary figure with build_stroke_figure and extract the
    distance axis (x) from the first trace. This guarantees the slider
    uses the *exact* same x as the plot (after trimming / scaling).
    """
    tmp_fig = build_stroke_figure(segment, title, interval_meta)
    if not tmp_fig.data:
        return np.array([0.0, 1.0])

    x_vals = np.asarray(tmp_fig.data[0].x, dtype=float)

    if not np.isfinite(x_vals).any():
        return np.array([0.0, 1.0])

    return x_vals



#when date changes, update filters.date
@app.callback(
    Output("filters", "data", allow_duplicate=True),
    Input("date-start", "date"),
    Input("date-end", "date"),
    Input("show-all-winter", "value"),
    State("filters", "data"),
    prevent_initial_call=True
)
def set_date(start, end, allwinter_value, flt):
    flt = dict(flt or {})
    if not start or not end:
        return no_update
    flt["date"] = [start, end]
    # store a simple boolean flag
    flt["allwinter"] = ("ALL" in (allwinter_value or []))
    return flt

#when rower select changes -> update filters.name
@app.callback(
    Output("filters", "data", allow_duplicate=True),
    Input("global-rower-filter", "value"),
    State("filters", "data"),
    prevent_initial_call=True,
    debounce=True
)
def _set_rowers(names, flt):
    names = [] if not names else list(names)
    flt = dict(flt or {})
    flt["name"] = names
    return flt


@app.callback(
    Output("filters", "data", allow_duplicate=True),
    Output("date-start", "date"),
    Output("date-end", "date"),
    Output("global-rower-filter", "value"),
    Output("rower-dropdown", "value"),
    Input("clear-filters", "n_clicks"),
    prevent_initial_call=True
)
def clear_filters(n):
    # defaults based on your data
    start_default = df["date"].min().date().isoformat()
    end_default   = df["date"].max().date().isoformat()

    # filters store defaults
    filters_default = {
        "date": [start_default, end_default],
        "name": []   # empty = all rowers
    }

    # UI defaults:
    # - reset start/end date pickers
    # - clear global rower multi-select
    # - clear the stroke rower (so it doesn’t override top-charts)
    return (
        filters_default,
        start_default,
        end_default,
        [],
        None
    )


@app.callback(
    Output('workout-dropdown', 'options'),
    Output('workout-dropdown', 'value'),
    Input('rower-dropdown', 'value')
)


def update_workout_dropdown(selected_rower):
    if not selected_rower:
        return [], None

    # filter for this rower’s workouts with stroke data
    sub = df[(df["name"] == selected_rower) & (df["stroke_data"].notna())].copy()

    # build dropdown label and value
    sub["label"] = sub["date"].dt.strftime("%Y-%m-%d") + " - " + sub["distance"].astype(int).astype(str) + "m"
    sub["value"] = sub["stroke_data"].apply(json.dumps)

    # sort so most recent workout is first
    sub = sub.sort_values("date", ascending=False)

    # pick first workout as default value (latest)
    default_value = sub.iloc[0]["value"] if not sub.empty else None

    return sub[["label", "value"]].to_dict("records"), default_value




@app.callback(
    Output("interval-dropdown", "options"),
    Output("interval-dropdown", "value"),
    Input("workout-dropdown", "value"),
)
def _update_interval_dropdown(selected_workout):
    if not selected_workout:
        return [], None

    sd = json.loads(selected_workout)

    # use the same logic as single_workout_vis: split by time reset
    segments, intervals_meta = segments_for_sd(sd)

    options = build_interval_dropdown_options(segments, intervals_meta)
    value = options[0]["value"] if options else None
    return options, value


@app.callback(
    Output("distance-range", "min"),
    Output("distance-range", "max"),
    Output("distance-range", "value"),
    Input("workout-dropdown", "value"),
    Input("interval-dropdown", "value"),
)
def _update_distance_slider(selected_workout, which_interval):
    """
    Whenever workout or interval changes, recompute the distance axis
    and reset the slider to cover the full interval.
    """
    if not selected_workout:
        return 0.0, 100.0, [0.0, 100.0]

    sd = json.loads(selected_workout)
    segments, intervals_meta = segments_for_sd(sd)

    if not segments:
        return 0.0, 100.0, [0.0, 100.0]

    if which_interval is None or which_interval < 0 or which_interval >= len(segments):
        which_interval = 0

    seg = segments[which_interval]
    meta = (
        intervals_meta[which_interval]
        if isinstance(intervals_meta, list)
        and which_interval < len(intervals_meta)
        and isinstance(intervals_meta[which_interval], dict)
        else {}
    )

    # use the same distance axis as the figure
    x_vals = get_distance_axis_for_segment(seg, meta, title="tmp")

    x_min = float(np.nanmin(x_vals))
    x_max = float(np.nanmax(x_vals))

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        x_min, x_max = 0.0, 100.0

    if x_max <= x_min:
        x_max = x_min + 1.0

    slider_min = float(np.floor(x_min))
    slider_max = float(np.ceil(x_max))

    return slider_min, slider_max, [slider_min, slider_max]



@app.callback(
    Output('cumulative-distance-graph', 'figure'),
    Output('time-of-day-pie-chart', 'figure'),
    Output('weekday-bar-chart', 'figure'),
    Output('leaderboard-table', 'children'),
    Input('filters', 'data'),
    Input('include-coaches', 'value')
)


def _update_top_section(flt, include_coaches):



    # 1) apply filters common to all charts
    sub = apply_filters(df, flt or {})

    # 2) apply coaches toggle globally (charts + leaderboard)
    include_flag = (include_coaches or [])
    if "yes" not in include_flag:
        sub = sub[~sub["name"].isin(COACHES)].copy()

    # leaderboards use the same 'sub'
    sub_lb = sub.copy()

    # 2) cumulative by rower
    cum = sub.sort_values(["name","date"]).copy()
    cum["cumulative_distance"] = cum.groupby("name")["distance"].cumsum()
    cum = cum[["name","date","cumulative_distance"]]

    fig_cum = px.line(
        cum,
        x="date",
        y="cumulative_distance",
        color="name",
        title="Cumulative Distance by Rower - Last 14 days",
        labels={
            "cumulative_distance": "Cumulative Distance",
            "date": "Date",
            "name": "Rower",
        },
        color_discrete_map=ROWER_COLOR_MAP,      # <- stable colors
        category_orders={"name": ALL_ROWERS},    # <- stable legend order
    )


    # 3) pie by time-of-day (force stable categories + order)
    tod_order = ["Morning", "Midday", "Evening"]
    tod_tbl = (sub[sub["time_of_day"].isin(tod_order)]
                .groupby("time_of_day", as_index=False)["distance"].sum())

    # reindex to ensure all slices always exist
    tod_tbl = tod_tbl.set_index("time_of_day").reindex(tod_order, fill_value=0).reset_index()

    fig_tod = px.pie(
        tod_tbl,
        names="time_of_day",
        values="distance",
        title="Distance by Time of Day",
    )
    # stability hints
    fig_tod.update_traces(sort=False)                      # don't re-order slices
    fig_tod.update_layout(transition_duration=400, uirevision="stay")


    # 4) bar by weekday
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    sub["weekday"] = pd.Categorical(sub["weekday"], categories=weekday_order, ordered=True)
    wday_tbl = (sub.groupby("weekday", as_index=False, observed=False)["distance"]
                .sum().sort_values("weekday"))

    fig_wday = px.bar(
        wday_tbl, x="weekday", y="distance",
        title="Distance by Weekday",
        labels={'weekday': 'Weekday', 'distance': 'Distance'}
    )

    # add headroom so bars don’t “kiss” the top (esp. when coaches included)
    ymax = float(wday_tbl["distance"].max()) if not wday_tbl.empty else 0.0
    padding = 1.15  # 15% headroom
    fig_wday.update_yaxes(range=[0, ymax * padding], tickformat=",")

    ymax_cum = float(cum["cumulative_distance"].max()) if not cum.empty else 0.0
    fig_cum.update_yaxes(range=[0, ymax_cum * 1.05], tickformat=",")
    fig_cum.update_layout(legend=dict(orientation="v", x=1.02, y=1))
    fig_cum.update_yaxes(tickformat=",")
    fig_wday.update_yaxes(tickformat=",")




    # 5) leaderboard (now computed from sub_lb)
    START = pd.Timestamp("2025-10-21")
    sub_lb["week_num"] = ((sub_lb["date"] - START).dt.days // 7) + 1

    # team weekly totals
    weekly_totals = sub_lb.groupby("week_num", as_index=False)["distance"].sum()

    # **per-rower weekly totals**, then pick the top rower each week
    weekly_by_rower = (
        sub_lb.groupby(["week_num", "name"], as_index=False)["distance"].sum()
    )

    idx = weekly_by_rower.groupby("week_num")["distance"].idxmax()
    weekly_winner = (
        weekly_by_rower.loc[idx, ["week_num", "name"]]
                    .rename(columns={"name": "most_rowed"})
    )

    leaderboard = weekly_totals.merge(weekly_winner, on="week_num", how="left")


    table = html.Table(
        children=[
            html.Tr([html.Th("Week"), html.Th("Team's meters rowed"), html.Th("Most Rowed")])
        ] + [
            html.Tr([
                html.Td(int(r.week_num)),
                html.Td(f"{int(r.distance):,}" if pd.notna(r.distance) else "0"),
                html.Td("" if pd.isna(r.most_rowed) else r.most_rowed)
            ])
            for r in leaderboard.itertuples(index=False)
        ]
    )

    fig_cum.update_layout(transition_duration=400)
    fig_tod.update_layout(transition_duration=400)
    fig_wday.update_layout(transition_duration=400)


    return fig_cum, fig_tod, fig_wday, table

@app.callback(
    Output('workout-stroke-graph', 'figure'),
    Input('workout-dropdown', 'value'),
    Input('interval-dropdown', 'value'),
    Input('distance-range', 'value'),
)
def update_workout_graph(selected_workout, which_interval, distance_range):
    if not selected_workout:
        return go.Figure(layout_title_text="No Data Available")

    sd = json.loads(selected_workout)
    segments, intervals_meta = segments_for_sd(sd)

    if not segments:
        return go.Figure(layout_title_text="No strokes in this workout")

    if which_interval is None or which_interval < 0 or which_interval >= len(segments):
        which_interval = 0

    seg = segments[which_interval]
    meta = (
        intervals_meta[which_interval]
        if isinstance(intervals_meta, list)
        and which_interval < len(intervals_meta)
        and isinstance(intervals_meta[which_interval], dict)
        else {}
    )

    title = f"Interval {which_interval + 1}"

    fig = build_stroke_figure(seg, title, meta)

    # Apply the slider window as an x-axis range
    if distance_range and len(distance_range) == 2:
        low, high = distance_range
        fig.update_xaxes(range=[low, high])

    return fig



@app.callback(
    Output("summary-table", "children"),
    Input("workout-dropdown", "value"),
    Input("interval-dropdown", "value"),
    Input("distance-range", "value"),
)
def update_summary_table(selected_workout, which_interval, distance_range):
    rows = compute_summary_rows_local(selected_workout, which_interval, distance_range)

    if not rows:
        return html.Div("No data")

    header = html.Tr(
        [
            html.Th("Label"),
            html.Th("Time"),
            html.Th("Meters"),
            html.Th("Pace"),
            html.Th("S/M"),
            html.Th("HR"),
        ],
        className="intervals-header-row",
    )

    body = []
    for r in rows:
        row_classes = ["interval-row"]
        if r["label"].startswith("Selected Interval"):
            row_classes.append("interval-row-selected")
        if r["label"].startswith("Workout total"):
            row_classes.append("interval-row-total")

        body.append(
            html.Tr(
                [
                    html.Td(r["label"], className="interval-cell label-cell"),
                    html.Td(r["time"], className="interval-cell numeric-cell"),
                    html.Td(r["meters"], className="interval-cell numeric-cell"),
                    html.Td(r["pace"], className="interval-cell numeric-cell"),
                    html.Td(r["spm"], className="interval-cell numeric-cell"),
                    html.Td(r["hr"], className="interval-cell numeric-cell"),
                ],
                className=" ".join(row_classes),
            )
        )

    return html.Div(
        [
            html.H4("Intervals", className="intervals-title"),
            html.Table(
                [header] + body,
                className="intervals-table",
            ),
        ]
    )


if __name__ == '__main__':
    app.run_server(debug=True)

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8050)
