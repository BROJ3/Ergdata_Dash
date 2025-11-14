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
        dcc.Graph(id='workout-stroke-graph')
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

    # Only trust explicit metadata; never infer.
    meta = _intervals_meta_from_sd(sd)
    n = len(meta) if isinstance(meta, list) else 0

    if n >= 2:
        opts = [{"label": f"Interval {i+1}", "value": i} for i in range(n)]
        return opts, 0

    # No valid metadata -> force single interval
    return [{"label": "Interval 1", "value": 0}], 0

    # Fallback: infer by time resets (legacy behavior)
    strokes = _strokes_from_sd(sd)
    t = np.array([pt.get("t", np.nan) for pt in strokes], dtype=float)
    resets = np.where(np.diff(t) < -3.0)[0] + 1
    nseg = int(len(resets) + 1) if len(t) else 0
    opts = [{"label": f"Interval {i+1}", "value": i} for i in range(nseg)]
    return opts, (0 if nseg > 0 else None)


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
)
def update_workout_graph(selected_workout, which_interval):
    if not selected_workout:
        return go.Figure(layout_title_text="No Data Available")

    sd = json.loads(selected_workout)
    strokes = _strokes_from_sd(sd)

    # series (full workout)
    p   = np.array([pt.get("p",   0) for pt in strokes], dtype=float)   # pace (deci-sec/500m)
    hr  = np.array([pt.get("hr",  0) for pt in strokes], dtype=float)   # bpm
    spm = np.array([pt.get("spm", 0) for pt in strokes], dtype=float)   # strokes/min
    t   = np.array([pt.get("t", np.nan) for pt in strokes], dtype=float)  # elapsed sec (resets by piece)

    # ----- segmentation: ONLY from explicit metadata -----
    meta = _intervals_meta_from_sd(sd)
    if isinstance(meta, list) and len(meta) >= 2:
        nseg = len(meta)

        # If your metadata includes explicit index spans {a,b}, use them.
        if isinstance(meta[0], dict) and "a" in meta[0] and "b" in meta[0]:
            if which_interval is None or which_interval < 0 or which_interval >= nseg:
                which_interval = 0
            a = int(meta[which_interval]["a"])
            b = int(meta[which_interval]["b"])

        else:
            # Otherwise: map metadata to segments using t resets, but ONLY because meta exists.
            # If count mismatch, fall back to full workout.
            resets = np.where(np.diff(t) < -3.0)[0] + 1
            boundaries = np.r_[0, resets, len(t)]
            if len(boundaries) - 1 == nseg:
                if which_interval is None or which_interval < 0 or which_interval >= nseg:
                    which_interval = 0
                a, b = boundaries[which_interval], boundaries[which_interval + 1]
            else:
                # Unknown meta format; safest is full workout
                which_interval = 0
                a, b = 0, len(t)
    else:
        # No metadata -> single interval over the whole workout
        nseg = 1
        which_interval = 0
        a, b = 0, len(t)


    t_seg = t[a:b].copy()
    p_seg = p[a:b].copy()
    hr_seg = hr[a:b].copy()
    spm_seg = spm[a:b].copy()

    # If distance 'd' exists, use it to detect movement too (optional but robust)
    d = np.array([pt.get("d", np.nan) for pt in strokes], dtype=float)
    d_seg = d[a:b] if len(d) == len(t) else None

    # ---------- trim trailing rest ----------
    SPM_ACTIVE = 12  # tweak if needed (10–16 usually works)
    active_spm = np.isfinite(spm_seg) & (spm_seg >= SPM_ACTIVE)

    if d_seg is not None and np.isfinite(d_seg).all():
        delta_d = np.r_[0, np.diff(d_seg)]
        moving = delta_d > 0
        active_mask = active_spm | moving
    else:
        active_mask = active_spm

    if active_mask.any():
        last_active = np.max(np.nonzero(active_mask)[0])
        # keep up to and including last active stroke
        t_seg  = t_seg[:last_active + 1]
        p_seg  = p_seg[:last_active + 1]
        hr_seg = hr_seg[:last_active + 1]
        spm_seg= spm_seg[:last_active + 1]
        if d_seg is not None:
            d_seg = d_seg[:last_active + 1]   

        # ---------- x axis in meters (distance) ----------
        # Pull raw distance; Concept2 often sends 'd' in decimeters (dm)
        d = np.array([pt.get("d", np.nan) for pt in strokes], dtype=float)
        d_seg = d[a:b] if len(d) == len(t) else None

        if d_seg is not None and np.isfinite(d_seg).any():
            # 1) Trim to last active stroke (already done above); d_seg now matches t_seg/p_seg slices

            # 2) Detect units by speed and/or span
            dt = np.diff(t_seg)
            dd = np.diff(d_seg)
            valid = (dt > 0) & np.isfinite(dt) & np.isfinite(dd)

            d_m = d_seg.copy()
            if valid.any():
                med_v = np.median(dd[valid] / dt[valid])  # "distance per second" in whatever units d uses
                # Typical rowing speed ≈ 3–6 m/s. If med_v is ~10× that, distance is in decimeters.
                if med_v > 25:          # very rare: centimeters → /100
                    d_m = d_seg / 100.0
                elif med_v > 8:         # common: decimeters → /10
                    d_m = d_seg / 10.0
            else:
                # Fallback by span
                span = float(d_seg[-1] - d_seg[0])
                if span > 500000:       # >500 km ⇒ centimeters
                    d_m = d_seg / 100.0
                elif span > 50000:      # >50 km  ⇒ decimeters
                    d_m = d_seg / 10.0

            # Make the interval start at 0 m
            x_dist = d_m - d_m[0]
            dist_labels = [f"{v:.0f} m" for v in x_dist]
        else:
            # No distance in stream → fall back to stroke index
            x_dist = np.arange(len(p_seg))
            dist_labels = [f"{int(v)}" for v in x_dist]

        # convert decimeters → meters
        d_m = d_seg / 10.0
        x_dist = d_m - d_m[0]
        dist_labels = [f"{v:.0f} m" for v in x_dist]

        # tick setup (keep this)
        total = x_dist[-1] if len(x_dist) else 0
        step = 250  # or whatever tick spacing you prefer
        xtickvals = list(np.arange(0, total + step, step))
        xticktext = [f"{int(v)}" for v in xtickvals]  # label in meters





    # ---------- figure ----------
    fig = go.Figure()
    pace_labels = [f"{int(v)//600}:{((int(v)%600)//10):02d}.{int(v)%10}" for v in p_seg]

    fig.add_trace(go.Scatter(
    x=x_dist, y=p_seg, name="Pace/500m", mode="lines",
    customdata=np.column_stack([dist_labels, pace_labels]),
    hovertemplate="Distance=%{customdata[0]}<br>Pace=%{customdata[1]}<extra></extra>"
    )),
    # Heart rate trace
    fig.add_trace(go.Scatter(
        x=x_dist, y=hr_seg,
        name="Heart Rate (bpm)",
        mode="lines",
        yaxis="y2",
        hovertemplate="Distance=%{x:.0f} m<br>HR=%{y:.0f} bpm<extra></extra>"
    )),
    fig.add_trace(go.Scatter(
        x=x_dist, y=spm_seg, name="SPM", mode="lines", yaxis="y3",
        customdata=dist_labels,
        hovertemplate="Distance=%{customdata}<br>SPM=%{y:.0f}<extra></extra>"
    ))

    fig.update_traces(line=dict(width=0.8))



    # Pace ticks (deci-sec → mm:ss)
    tickvals = list(range(540, 2341, 180))  # 1:30 .. 3:54 every 30s
    ticktext = [f"{v//600}:{(v%600)//10:02d}" for v in tickvals]

    fig.update_layout(
        title=f"Workout: Pace, HR, and SPM — Interval {which_interval+1}/{nseg}",
        # x-axis: linear seconds with custom mm:ss ticks
        xaxis=dict(
        title="Distance (m)",
        tickmode="array",
        tickvals=xtickvals,
        ticktext=xticktext
    ),

        # left y (pace)
        yaxis=dict(
            title="Pace (mm:ss / 500m)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            autorange="reversed", range=[max(tickvals), min(tickvals)]
        ),
        # right y (HR) – outer
        yaxis2=dict(
            title="Heart Rate (bpm)",
            overlaying="y",
            side="right",
            anchor="free",
            range=[0,220],
            position=1.0,          # right edge
            showgrid=False,
            title_standoff=12
        ),
        # right y (SPM) – slightly inside, no ticks to avoid overlap
        yaxis3=dict(
            title="SPM",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.96,
            showgrid=False,
            showticklabels=False,
            ticks="",
            title_standoff=36
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, r=160)
    )

    # headroom for HR & SPM
    if np.any(hr > 0):
        hr_min, hr_max = np.nanmin(hr), np.nanmax(hr)
        pad = max(3, 0.08 * (hr_max - hr_min if hr_max > hr_min else 10))
        fig.update_layout(yaxis2=dict(range=[hr_min - pad, hr_max + pad],
                                      overlaying="y", side="right"))
    if np.any(spm > 0):
        spm_min, spm_max = np.nanmin(spm), np.nanmax(spm)
        pad = max(1, 0.12 * (spm_max - spm_min if spm_max > spm_min else 5))
        fig.update_layout(yaxis3=dict(range=[spm_min - pad, spm_max + pad],
                                      overlaying="y", side="right",
                                      anchor="free", position=0.96))

    return fig




if __name__ == '__main__':
    app.run_server(debug=True)

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8050)
