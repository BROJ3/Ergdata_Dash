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
    """
    w = WORKOUTS[idx]
    strokes = w["strokes"]["data"]
    segments = split_strokes_by_time_reset(strokes)
    intervals_meta = w.get("workout", {}).get("intervals", [])
    return segments, intervals_meta, w


def build_interval_dropdown_options(segments, intervals_meta):
    options = []
    for i, seg in enumerate(segments):
        label = f"Interval {i+1}"
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
#               HELPER: FORMAT PACE
# =========================================================

def format_pace(sec):
    if sec <= 0 or not np.isfinite(sec):
        return "-"
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:04.1f}"



# =========================================================
#   HELPER: COMPUTE DISTANCE AXIS FOR A SEGMENT (for slider)
# =========================================================

def compute_distance_axis(segment, interval_meta):
    """
    Returns the distance axis x[] for a given segment, applying
    the same trimming/scaling logic used in the figure.
    """
    if not segment:
        return np.array([0.0])

    # --- trim to programmed work time (same logic as in build_stroke_figure) ---
    t = np.array([s.get("t", np.nan) for s in segment], float)

    work_raw = None
    if interval_meta and ("time" in interval_meta):
        work_raw = float(interval_meta["time"])  # seconds

    if work_raw is not None and np.isfinite(work_raw):
        t0 = t[0] if np.isfinite(t[0]) else 0
        t_rel = t - t0
        mask = np.isfinite(t_rel) & (t_rel <= work_raw)
        if mask.any():
            last = int(np.max(np.nonzero(mask)[0]))
            segment = segment[: last + 1]
            t = t[: last + 1]

    # --- distance axis (same scaling as build_stroke_figure) ---
    d = np.array([s.get("d", np.nan) for s in segment], float)
    x = np.arange(len(segment), dtype=float)

    if np.isfinite(d).any():
        dt = np.diff(t)
        dd = np.diff(d)
        valid = (dt > 0) & np.isfinite(dt) & np.isfinite(dd)

        d_m = d.copy()
        if valid.any():
            med_v = np.median(dd[valid] / dt[valid])
            if med_v > 25:
                d_m = d / 100.0
            elif med_v > 8:
                d_m = d / 10.0
        x = d_m - d_m[0]

    # Fallback if everything blows up
    if not np.isfinite(x).any():
        x = np.arange(len(segment), dtype=float)

    return x



# =========================================================
#               BUILD STROKE FIGURE (WITH WORK-TIME TRIM)
# =========================================================

def build_stroke_figure(segment, title, interval_meta):
    """
    segment = list of strokes for this interval
    interval_meta = metadata for this interval: contains programmed work time
    """

    # ----------------------------
    # 1. TRIM STROKES TO WORK TIME
    # ----------------------------
    t = np.array([s.get("t", np.nan) for s in segment], float)

    work_raw = None
    if interval_meta and ("time" in interval_meta):
        # In your JSON, "time" is already **seconds**, not deciseconds.
        work_raw = float(interval_meta["time"])   # seconds

    if work_raw is not None and np.isfinite(work_raw):
        # shift time to start at zero
        t0 = t[0] if np.isfinite(t[0]) else 0
        t_rel = t - t0

        # keep only strokes where t_rel <= work_raw
        mask = np.isfinite(t_rel) & (t_rel <= work_raw)
        if mask.any():
            last = np.max(np.nonzero(mask)[0])
            segment = segment[: last + 1]
            t = t[: last + 1]

    # ----------------------------
    # 2. Extract arrays AFTER trimming
    # ----------------------------
    hr = np.array([s.get("hr", 0) for s in segment], float)
    spm = np.array([s.get("spm", 0) for s in segment], float)
    d = np.array([s.get("d", np.nan) for s in segment], float)

    p_raw = np.array([s.get("p", 0) for s in segment], float)
    pace_seconds = p_raw / 10.0
    pace_labels = [format_pace(v) for v in pace_seconds]

    # ----------------------------
    # 3. Distance axis
    # ----------------------------
    x = np.arange(len(segment))
    if np.isfinite(d).any():
        dt = np.diff(t)
        dd = np.diff(d)
        valid = (dt > 0) & np.isfinite(dt) & np.isfinite(dd)

        d_m = d.copy()
        if valid.any():
            med_v = np.median(dd[valid] / dt[valid])
            if med_v > 25:
                d_m = d / 100.0
            elif med_v > 8:
                d_m = d / 10.0
        x = d_m - d_m[0]

    # ----------------------------
    # 4. Build figure
    # ----------------------------
    fig = go.Figure()

    # Pace trace
    fig.add_trace(go.Scatter(
        x=x, y=pace_seconds,
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
    y_min = np.nanmin(pace_seconds)
    y_max = np.nanmax(pace_seconds)
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

    return fig



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

    # --- custom distance range slider under the chart ---
    html.Div(
        style={"width": "80%", "margin": "20px auto 0"},
        children=[
            html.Label("Filter by distance (m):"),
            dcc.RangeSlider(
                id="distance-range",
                min=0,
                max=1000,            # will be overwritten by callback
                step=50,
                value=[0, 1000],     # will be overwritten by callback
                allowCross=False,
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
            ),
        ],
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
    """
    Whenever workout/interval changes, recompute the distance axis and
    reset the slider to cover the full range for that interval.
    """
    segments, intervals_meta, w = strokes_for_workout(workout_idx)

    if not segments:
        # default small range
        return 0, 100, [0, 100]

    if interval_idx is None or interval_idx >= len(segments):
        interval_idx = 0

    seg = segments[interval_idx]
    meta = intervals_meta[interval_idx] if interval_idx < len(intervals_meta) else None

    x = compute_distance_axis(seg, meta)

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))

    # tidy up bounds
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        x_min, x_max = 0.0, 100.0

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

    if interval_idx is None or interval_idx >= len(segments):
        interval_idx = 0

    seg = segments[interval_idx]
    meta = intervals_meta[interval_idx] if interval_idx < len(intervals_meta) else None

    title = (
        f"{ROWER['name']} — {w['date']} — {w['distance']} m — "
        f"Interval {interval_idx + 1}"
    )

    fig = build_stroke_figure(seg, title, meta)

    # Apply slider filter as an x-axis range
    if distance_range and len(distance_range) == 2:
        low, high = distance_range
        fig.update_xaxes(range=[low, high])

    return fig



# =========================================================
#                     ENTRY POINT
# =========================================================

if __name__ == "__main__":
    app.run_server(debug=True)
