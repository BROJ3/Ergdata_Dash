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
                    max=1000,          # placeholder; will be updated by callback
                    step=50,
                    value=[0, 1000],   # placeholder; will be updated by callback
                    allowCross=False,
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="mouseup",   # update when mouse is released
                    marks=None

                ),
            ],
        )
    ]
)

def _strokes_from_sd(sd: dict):
    """
    Return a list of stroke dicts from stroke_data payloads shaped as:
      {"data": [ ... ]}              # old
      {"data": {"data": [ ... ]}}    # new
    """
    if not isinstance(sd, dict):
        return []
    slot = sd.get("data", [])
    if isinstance(slot, dict) and isinstance(slot.get("data"), list):
        return slot["data"]
    return slot if isinstance(slot, list) else []


def _intervals_meta_from_sd(sd: dict):
    """
    Return intervals metadata if present, regardless of wrapper.
    Accepts:
      sd["intervals_meta"]
      sd["data"]["meta"] or sd["data"]["intervals_meta"]
    """
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
    """
    Split stroke list into segments whenever the Concept2 time counter resets.
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
    """
    Build the interval dropdown options, similar to single_workout_vis.py.
    """
    options = []
    for i, seg in enumerate(segments):
        label = f"Interval {i + 1}"

        if isinstance(intervals_meta, list) and i < len(intervals_meta):
            meta = intervals_meta[i]
            if isinstance(meta, dict):
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


def format_pace(sec):
    """
    Convert seconds per 500m to 'M:SS.s' string (used for axis + hover).
    """
    if sec <= 0 or not np.isfinite(sec):
        return "-"
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:04.1f}"


def build_stroke_figure(segment, title, interval_meta):
    """
    segment = list of strokes for this interval
    interval_meta = metadata for this interval (contains programmed work time).
    This is copied from single_workout_vis.py, adapted to use in local_vis.
    """
    # 1) Trim strokes to programmed work time (if available)
    t = np.array([s.get("t", np.nan) for s in segment], float)

    work_raw = None
    if isinstance(interval_meta, dict) and ("time" in interval_meta):
        # In your JSON, "time" is already seconds.
        work_raw = float(interval_meta["time"])

    if work_raw is not None and np.isfinite(work_raw):
        t0 = t[0] if np.isfinite(t[0]) else 0.0
        t_rel = t - t0
        mask = np.isfinite(t_rel) & (t_rel <= work_raw)
        if mask.any():
            last = int(np.max(np.nonzero(mask)[0]))
            segment = segment[: last + 1]
            t = t[: last + 1]

    # 2) Extract arrays AFTER trimming
    hr = np.array([s.get("hr", 0) for s in segment], float)
    spm = np.array([s.get("spm", 0) for s in segment], float)
    d = np.array([s.get("d", np.nan) for s in segment], float)

    p_raw = np.array([s.get("p", 0) for s in segment], float)
    pace_seconds = p_raw / 10.0
    pace_labels = [format_pace(v) for v in pace_seconds]
        # --- Trim very slow strokes at the start and end if they are clear outliers ---
    valid_idx = np.where(np.isfinite(pace_seconds) & (pace_seconds > 0))[0]

    if valid_idx.size:
        valid_pace = pace_seconds[valid_idx]
        # threshold: drop strokes slower than this (e.g. warm-up / cool-down strokes)
        thr = float(np.nanpercentile(valid_pace,99))  # keep ~98% fastest strokes               #KNOB

        good = (
            np.isfinite(pace_seconds)
            & (pace_seconds > 0)
            & (pace_seconds <= thr)
        )

        if good.any():
            # first and last "good" strokes
            first_good = int(np.argmax(good))  # first True from left
            last_good = len(pace_seconds) - 1 - int(np.argmax(good[::-1]))  # from right

            # Only trim if we’re actually dropping something at the edges
            if first_good > 0 or last_good < len(pace_seconds) - 1:
                sl = slice(first_good, last_good + 1)
                t = t[sl]
                hr = hr[sl]
                spm = spm[sl]
                d = d[sl]
                pace_seconds = pace_seconds[sl]
                pace_labels = pace_labels[sl]


    # 3) Distance axis (meters, normalized to start at 0)
    # ErgData's "d" field is in decimeters (0.1 m), so convert directly.
    if np.isfinite(d).any():
        d_m = d / 10.0          # decimeters -> meters
        x = d_m - d_m[0]        # start interval at 0 m
    else:
        x = np.arange(len(segment))


    # 4) Build figure
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
    
    

    
    # ignore the first/last couple of strokes if they are odd.
    idx_valid = np.where(np.isfinite(pace_seconds) & (pace_seconds > 0))[0]

    if idx_valid.size:
        # Core indices: drop first and last 2 valid strokes if we have enough
        if idx_valid.size > 6:
            core_idx = idx_valid[0:-1]
        else:
            core_idx = idx_valid

        core_pace = pace_seconds[core_idx]

        med = float(np.nanmedian(core_pace))
        p10 = float(np.nanpercentile(core_pace, 10))
        p90 = float(np.nanpercentile(core_pace, 90))

        # Half-span around median based on spread, with a minimum width
        half_span = max((p90 - p10), 12.0)   # roughly double the previous span
        pad = 5.0

        p_lo = med - half_span - pad
        p_hi = med + half_span + pad

        # Clamp to a reasonable global range (1:20–3:00 → 80–180s)
        # Clamp to a reasonable global range (1:20–3:00 → 80–180s)
        p_lo = max(p_lo, 80.0)
        p_hi = min(p_hi, 180.0)

        span = p_hi - p_lo

        # "Nice" round splits every 5 seconds: 1:20, 1:25, 1:30, ..., 3:00
        nice = np.arange(80.0, 181.0, 1.0)
        ticks = nice[(nice >= p_lo) & (nice <= p_hi)]

        # Fallback: if for some weird reason we got < 3 ticks, use uniform grid
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


if __name__ == '__main__':
    app.run_server(debug=True)

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8050)
