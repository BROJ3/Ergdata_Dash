import sqlite3
import plotly.express as px
import dash
from dash import dcc, html, no_update, ctx
from dash.dependencies import Input, Output, State
from datetime import datetime
import json
import pandas as pd
import numpy as np

def apply_filters(df:pd.DataFrame, flt:dict) -> pd.DataFrame:
    sub = df

    #date range
    if flt.get("date"):
        start,end = pd.to_datetime(flt["date"][0]),pd.to_datetime(flt["date"][1])
        sub = sub[(sub["date"]>= start) & (sub["date"] <= end)]

    #rowers
    if flt.get("name"):
        sub=sub[sub["name"].isin(flt["name"])]

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

'''
distance_by_tod = (
    df[df["time_of_day"].isin(["Morning","Midday","Evening"])]
    .groupby("time_of_day", as_index=False)["distance"].sum()
    .sort_values("time_of_day")
)'''


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

'''
# total distance per week
weekly_totals = df.groupby("week_num", as_index=False)["distance"].sum()

# rower who rowed the most in each week
idx = df.groupby("week_num")["distance"].idxmax()
weekly_winner = (
    df.loc[idx, ["week_num", "name"]]
      .rename(columns={"name": "most_rowed"})
)

# combine totals + winners
weekly_leaderboard = weekly_totals.merge(weekly_winner, on="week_num")

'''



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

        html.H1("Clarkson Crew Performance Dashboard", className="dashboard-header"),
        html.H4("Data from Winter 24/25", className="dashboard-header"),


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
        )
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
                html.Div(
                    className="dropdown",
                    children=[
                        html.Label("Select Metric:"),
                        dcc.Dropdown(
                            id='metric-dropdown',
                            options=[
                                {'label': 'Strokes Per Minute', 'value': 'spm'},
                                {'label': 'Heart Rate', 'value': 'hr'},
                                {'label': 'Pace/500m', 'value': 'p'}
                            ],
                            value='spm'
                        )
                    ]
                )
            ]
        ),

        dcc.Graph(id='workout-stroke-graph')
    ]
)


#when date changes, update filters.date
@app.callback(
    Output("filters", "data", allow_duplicate=True),
    Input("date-start", "date"),
    Input("date-end", "date"),
    State("filters", "data"),
    prevent_initial_call=True
)
def set_date(start,end,flt):
    flt=dict(flt or {})

    if not start or not end:
        return no_update
    flt["date"] = [start,end]
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
    Input('rower-dropdown', 'value')
)


def update_workout_dropdown(selected_rower):
    if not selected_rower:
        return []
    sub = df[(df["name"] == selected_rower) & (df["stroke_data"].notna())].copy()
    sub["label"] = sub["date"].dt.strftime("%Y-%m-%d") + " - " + sub["distance"].astype(int).astype(str) + "m"
    
    # store the raw JSON string for the callback (dash expects JSON-serializable)
    sub["value"] = sub["stroke_data"].apply(json.dumps)
    sub = sub.sort_values("date", ascending=False)
    return sub[["label", "value"]].to_dict("records")

@app.callback(
    Output('cumulative-distance-graph', 'figure'),
    Output('time-of-day-pie-chart', 'figure'),
    Output('weekday-bar-chart', 'figure'),
    Output('leaderboard-table', 'children'),
    Input('filters', 'data')
)

def _update_top_section(flt):
    # 1) apply filters
    sub = apply_filters(df, flt or {})

    # 2) cumulative by rower
    cum = (sub.sort_values(["name","date"])
              .assign(cumulative_distance=sub.groupby("name")["distance"].cumsum())
              [["name","date","cumulative_distance"]])
    fig_cum = px.line(
        cum, x='date', y='cumulative_distance', color='name',
        title="Total Team Cumulative Distance by Rower",
        labels={'cumulative_distance': 'Cumulative Distance', 'date': 'Date', 'name': 'Rower'}
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
    fig_wday = px.bar(wday_tbl, x="weekday", y="distance",
                      title="Distance by Weekday",
                      labels={'weekday': 'Weekday', 'distance': 'Distance'})

    # 5) leaderboard (compute from filtered data)
    START = pd.Timestamp("2024-11-01")
    sub["week_num"] = ((sub["date"] - START).dt.days // 7) + 1
    weekly_totals = sub.groupby("week_num", as_index=False)["distance"].sum()
    if not sub.empty:
        idx = sub.groupby("week_num")["distance"].idxmax()
        weekly_winner = (sub.loc[idx, ["week_num","name"]]
                           .rename(columns={"name":"most_rowed"}))
        leaderboard = weekly_totals.merge(weekly_winner, on="week_num", how="left")
    else:
        leaderboard = weekly_totals.assign(most_rowed=np.nan)

    # Build table rows as HTML
    table = html.Table(
        children=[
            html.Tr([html.Th("Week"), html.Th("Team's meters rowed"), html.Th("Most Rowed")])
        ] + [
            html.Tr([html.Td(int(r.week_num)),
                     html.Td(f"{int(r.distance):,}"),
                     html.Td("" if pd.isna(r.most_rowed) else r.most_rowed)])
            for r in leaderboard.itertuples(index=False)
        ]
    )
    fig_cum.update_layout(transition_duration=400)
    fig_tod.update_layout(transition_duration=400)
    fig_wday.update_layout(transition_duration=400)


    return fig_cum, fig_tod, fig_wday, table




@app.callback(
    Output('workout-stroke-graph', 'figure'),
    [Input('workout-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_workout_graph(selected_workout, selected_metric):
    if not selected_workout:
        return px.line(title="No Data Available")

    stroke_data = json.loads(selected_workout)
    strokes = stroke_data['data']

    metric_labels = {
        'spm': 'Strokes per Minute',
        'hr': 'Heart Rate',
        'p': 'Pace'
    }

    stroke_x = list(range(len(strokes)))
    stroke_y = [point.get(selected_metric, 0) for point in strokes]

    fig = px.line(
    x=stroke_x,
    y=stroke_y,
    title=f"{metric_labels[selected_metric]} for Selected Workout",
    labels={'x': 'Stroke Number:', 'y': metric_labels[selected_metric]}
)

    
    if selected_metric == 'p':
        # Build per-point labels
        pace_labels = [
            f"{val // 600}:{((val % 600) // 10):02d}.{val % 10}"
            for val in stroke_y
        ]
        fig.update_traces(
            customdata=pace_labels,
            hovertemplate="Pace=%{customdata}<extra></extra>"
        )

        # Standardized y-axis for pace
        tickvals = list(range(540, 2341, 180))  # every 30s
        ticktext = [f"{v//600}:{(v%600)//10:02d}" for v in tickvals]
        fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                autorange="reversed",
                range=[2340, 540]
            )
        )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8050)
