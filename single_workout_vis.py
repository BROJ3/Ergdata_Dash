import json
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

JSON_PATH = "one_man_workous.json"


# =========================================================
#                LOAD JSON
# =========================================================

def load_data():
    with open(JSON_PATH, "r") as f:
        return json.load(f)

DATA = load_data()
ROWER = DATA["rowers"][0]        # only one rower in this file
WORKOUTS = ROWER["workouts"]


# =========================================================
#                INTERVAL SPLITTING
# =========================================================

def split_strokes_by_time_reset(strokes):
    """
    Splits stroke list into segments when the Concept2 time counter resets.
    """
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


def strokes_for_workout(idx):
    """
    Returns (segments, intervals_meta, workout_dict) for workout idx.

    Robust to workouts that have no stroke data (strokes is None or missing).
    """
    w = WORKOUTS[idx]

    # Safely pull strokes list
    strokes_obj = w.get("strokes")
    strokes = []
    if isinstance(strokes_obj, dict):
        strokes = strokes_obj.get("data") or []
    # if strokes_obj is None or not a dict → leave strokes = []

    # Build segments only if we have any strokes
    segments = split_strokes_by_time_reset(strokes) if strokes else []

    # Interval metadata (if any)
    intervals_meta = w.get("workout", {}).get("intervals", []) or []

    return segments, intervals_meta, w



def build_interval_dropdown_options(segments, intervals_meta):
    options = []
    for i, seg in enumerate(segments):
        label = f"Interval {i + 1}"
        if i < len(intervals_meta):
            meta = intervals_meta[i]
            t = meta.get("time")
            d = meta.get("distance")
            parts = []
            if t is not None:
                parts.append(f"{t}s")
            if d is not None:
                parts.append(f"{d}m")
            if parts:
                label += " (" + ", ".join(parts) + ")"
        options.append({"label": label, "value": i})
    return options


# =========================================================
#               HELPER: FORMATTING
# =========================================================

def format_pace(sec):
    """
    Format pace seconds per 500m as M:SS.s
    """
    if sec <= 0 or not np.isfinite(sec):
        return "-"
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:04.1f}"


def format_time_hms(sec):
    """
    Format total time in seconds as H:MM:SS.s (or M:SS.s if < 1h)
    """
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


# =========================================================
#     HELPER: PREPARE SEGMENT ARRAYS (for plot + summaries)
# =========================================================

def prepare_segment_data(segment, interval_meta):
    """
    Given a list of stroke dicts and interval metadata, return arrays:
      x       : distance axis in meters (0 at start)
      t       : stroke time stamps (seconds)
      pace_s  : pace per 500m in seconds
      hr      : heart rate
      spm     : stroke rate
    with the same trimming / scaling logic as the plot.
    """
    if not segment:
        empty = np.array([], dtype=float)
        return dict(x=empty, t=empty, pace_s=empty, hr=empty, spm=empty)

    # ---- raw arrays ----
    t = np.array([s.get("t", np.nan) for s in segment], float)
    hr = np.array([s.get("hr", 0) for s in segment], float)
    spm = np.array([s.get("spm", 0) for s in segment], float)
    d = np.array([s.get("d", np.nan) for s in segment], float)
    p_raw = np.array([s.get("p", 0) for s in segment], float)
    pace_s = p_raw / 10.0  # Concept2 pace field in tenths of a second

    # ---- trim to programmed work time, if present ----
    work_raw = None
    if interval_meta and ("time" in interval_meta):
        work_raw = float(interval_meta["time"])  # already seconds

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
            pace_s = pace_s[: last + 1]

    # ---- distance axis scaling (m) ----
    x = np.arange(len(t), dtype=float)
    if np.isfinite(d).any():
        dt = np.diff(t)
        dd = np.diff(d)
        valid = (dt > 0) & np.isfinite(dt) & np.isfinite(dd)
        d_m = d.copy()
        if valid.any():
            med_v = np.median(dd[valid] / dt[valid])
            # heuristic: units check on d / t
            if med_v > 25:
                d_m = d / 100.0
            elif med_v > 8:
                d_m = d / 10.0
        x = d_m - d_m[0]

    return dict(x=x, t=t, pace_s=pace_s, hr=hr, spm=spm)


# =========================================================
#                 BUILD STROKE FIGURE
# =========================================================

def build_stroke_figure(segment, title, interval_meta, distance_range=None):
    """
    Build the stroke graph figure. If distance_range is provided,
    apply it as x-axis zoom window.
    """
    data = prepare_segment_data(segment, interval_meta)
    x = data["x"]
    t = data["t"]
    pace_s = data["pace_s"]
    hr = data["hr"]
    spm = data["spm"]

    if x.size == 0:
        return go.Figure(layout_title_text=title)

    pace_labels = [format_pace(v) for v in pace_s]

    fig = go.Figure()

    # Pace trace
    fig.add_trace(go.Scatter(
        x=x, y=pace_s,
        name="Pace/500m",
        mode="lines",
        yaxis="y",
        customdata=pace_labels,
        hovertemplate="Distance=%{x:.0f} m<br>Pace=%{customdata}<extra></extra>"
    ))

    # HR trace
    fig.add_trace(go.Scatter(
        x=x, y=hr,
        name="Heart Rate (bpm)",
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
        hovertemplate="Distance=%{x:.0f} m<br>SPM=%{y:.0f}<extra></extra>"
    ))

    # pace axis ticks
    y_min = float(np.nanmin(pace_s))
    y_max = float(np.nanmax(pace_s))
    ticks = np.arange(np.floor(y_min / 10) * 10,
                      np.ceil(y_max / 10) * 10 + 1,
                      10)
    tick_text = [format_pace(v) for v in ticks]

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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60, r=160),
        height=600
    )

    # Apply slider window if provided
    if distance_range and len(distance_range) == 2:
        fig.update_xaxes(range=[distance_range[0], distance_range[1]])

    return fig


# =========================================================
#          SUMMARY ROW COMPUTATION (INTERVAL / WORKOUT)
# =========================================================

def compute_pace_from_time_distance(time_s, dist_m):
    if time_s is None or dist_m is None or time_s <= 0 or dist_m <= 0:
        return np.nan
    return (float(time_s) / float(dist_m)) * 500.0


def compute_watts_from_pace(pace_500_s):
    """
    Concept2 formula: watts = 2.8 / (pace/500)^3
    where pace is seconds per 500m.
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
        "watts": int(round(watts)) if np.isfinite(watts) else "-",
        "cal_hr": int(round(cal_hr)) if np.isfinite(cal_hr) else "-",
        "spm": int(round(stroke_rate)) if stroke_rate is not None and np.isfinite(stroke_rate) else "-",
        "hr": int(round(hr_avg)) if hr_avg is not None and np.isfinite(hr_avg) else "-",
    }


def compute_summary_rows(workout_idx, interval_idx, distance_range):
    """
    Build list of summary rows: workout total, each interval, current view.
    """
    if workout_idx is None or workout_idx < 0 or workout_idx >= len(WORKOUTS):
        return []

    segments, intervals_meta, w = strokes_for_workout(workout_idx)

    rows = []

    # --- workout total row ---
    w_time = float(w.get("time", 0) or 0)
    w_dist = float(w.get("distance", 0) or 0)
    w_cal = float(w.get("calories_total", 0) or 0)
    w_smr = float(w.get("stroke_rate", 0) or 0)
    w_hr_avg = None
    if isinstance(w.get("heart_rate"), dict):
        w_hr_avg = w["heart_rate"].get("average")

    rows.append(make_summary_row("Workout total", w_time, w_dist, w_cal, w_smr, w_hr_avg))

    # --- per-interval rows (from intervals_meta) ---
    for i, meta in enumerate(intervals_meta):
        if not isinstance(meta, dict):
            continue
        t_s = float(meta.get("time", 0) or 0)
        d_m = float(meta.get("distance", 0) or 0)
        c_tot = meta.get("calories_total")
        if c_tot is not None:
            c_tot = float(c_tot)
        sr = meta.get("stroke_rate")
        if sr is not None:
            sr = float(sr)
        hr_avg = None
        if isinstance(meta.get("heart_rate"), dict):
            hr_avg = meta["heart_rate"].get("average")

        rows.append(
            make_summary_row(f"Interval {i + 1}", t_s, d_m, c_tot, sr, hr_avg)
        )

    # --- current view row (slider-filtered inside selected interval) ---
    if segments:
        if interval_idx is None or interval_idx < 0 or interval_idx >= len(segments):
            interval_idx = 0

        seg = segments[interval_idx]
        meta = intervals_meta[interval_idx] if interval_idx < len(intervals_meta) else {}

        data = prepare_segment_data(seg, meta)
        x = data["x"]
        t = data["t"]
        hr = data["hr"]
        spm = data["spm"]

        label = f"Current view (Interval {interval_idx + 1})"

        if x.size >= 2:
            # apply slider window if given
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
                time_s = float(t[-1] - t[0])
                hr_valid = hr[hr > 0]
                spm_valid = spm[spm > 0]
                hr_avg = float(np.mean(hr_valid)) if hr_valid.size > 0 else None
                spm_avg = float(np.mean(spm_valid)) if spm_valid.size > 0 else None

                rows.append(
                    make_summary_row(label, time_s, dist_m, None, spm_avg, hr_avg)
                )

    return rows


# =========================================================
#                       DASH APP
# =========================================================

WORKOUT_OPTIONS = [
    {
        "label": f"{i}: {w['date']} — {w['distance']} m — {w.get('workout_type', '')}",
        "value": i,
    }
    for i, w in enumerate(WORKOUTS)
]

app = Dash(__name__)

app.layout = html.Div([
    html.H3("Stroke Graph Viewer"),

    dcc.Dropdown(
        id="workout-select",
        options=WORKOUT_OPTIONS,
        value=0,
        clearable=False,
        style={"width": "70%"}
    ),

    dcc.Dropdown(
        id="interval-select",
        options=[],
        value=0,
        clearable=False,
        style={"width": "50%", "marginTop": "10px"}
    ),

    dcc.Graph(id="stroke-graph"),

    # custom distance range slider
    html.Div(
        style={"width": "80%", "margin": "20px auto 10px"},
        children=[
            html.Label("Filter interval by distance (m):"),
            dcc.RangeSlider(
                id="distance-range",
                min=0,
                max=1000,          # placeholder; updated by callback
                step=50,
                value=[0, 1000],   # placeholder; updated by callback
                allowCross=False,
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
                marks=None,
            ),
        ],
    ),

    # data summary table
    html.Div(
        id="summary-table",
        style={"width": "80%", "margin": "0 auto 40px"},
    ),
])


# =========================================================
#                   CALLBACKS
# =========================================================

@app.callback(
    Output("interval-select", "options"),
    Output("interval-select", "value"),
    Input("workout-select", "value"),
)
def update_interval_dropdown(workout_idx):
    segments, intervals_meta, w = strokes_for_workout(workout_idx)
    options = build_interval_dropdown_options(segments, intervals_meta)
    value = 0 if options else None
    return options, value


@app.callback(
    Output("distance-range", "min"),
    Output("distance-range", "max"),
    Output("distance-range", "value"),
    Input("workout-select", "value"),
    Input("interval-select", "value"),
)
def update_distance_slider(workout_idx, interval_idx):
    segments, intervals_meta, w = strokes_for_workout(workout_idx)

    if not segments:
        return 0.0, 100.0, [0.0, 100.0]

    if interval_idx is None or interval_idx < 0 or interval_idx >= len(segments):
        interval_idx = 0

    seg = segments[interval_idx]
    meta = intervals_meta[interval_idx] if interval_idx < len(intervals_meta) else {}

    data = prepare_segment_data(seg, meta)
    x = data["x"]

    if x.size == 0 or not np.isfinite(x).any():
        return 0.0, 100.0, [0.0, 100.0]

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if x_max <= x_min:
        x_max = x_min + 1.0

    slider_min = float(np.floor(x_min))
    slider_max = float(np.ceil(x_max))

    return slider_min, slider_max, [slider_min, slider_max]


@app.callback(
    Output("stroke-graph", "figure"),
    Input("workout-select", "value"),
    Input("interval-select", "value"),
    Input("distance-range", "value"),
)
def update_graph(workout_idx, interval_idx, distance_range):
    segments, intervals_meta, w = strokes_for_workout(workout_idx)

    if not segments:
        return go.Figure()

    if interval_idx is None or interval_idx < 0 or interval_idx >= len(segments):
        interval_idx = 0

    seg = segments[interval_idx]
    meta = intervals_meta[interval_idx] if interval_idx < len(intervals_meta) else None

    title = (
        f"{ROWER['name']} — {w['date']} — {w['distance']} m — "
        f"Interval {interval_idx + 1}"
    )

    return build_stroke_figure(seg, title, meta, distance_range)


@app.callback(
    Output("summary-table", "children"),
    Input("workout-select", "value"),
    Input("interval-select", "value"),
    Input("distance-range", "value"),
)
def update_summary_table(workout_idx, interval_idx, distance_range):
    rows = compute_summary_rows(workout_idx, interval_idx, distance_range)

    if not rows:
        return html.Div("No data")

    header = html.Tr([
        html.Th("Label"),
        html.Th("Time"),
        html.Th("Meters"),
        html.Th("Pace"),
        html.Th("Watts"),
        html.Th("Cal/Hr"),
        html.Th("S/M"),
        html.Th("HR"),
    ])

    body = []
    for r in rows:
        style = {}
        if r["label"].startswith("Current view"):
            style = {"backgroundColor": "#e7f3ff"}  # light blue
        body.append(html.Tr([
            html.Td(r["label"]),
            html.Td(r["time"]),
            html.Td(r["meters"]),
            html.Td(r["pace"]),
            html.Td(r["watts"]),
            html.Td(r["cal_hr"]),
            html.Td(r["spm"]),
            html.Td(r["hr"]),
        ], style=style))

    return html.Div([
        html.H4("Intervals", style={"marginTop": "10px"}),
        html.Table(
            [header] + body,
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "marginTop": "5px",
            },
        ),
    ])


# =========================================================
#                     ENTRY POINT
# =========================================================

if __name__ == "__main__":
    app.run_server(debug=True)
