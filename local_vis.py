import sqlite3
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
import json
import pandas as pd
import numpy as np


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
cumulative_df = df[["name", "date", "cumulative_distance"]]


def tod(h):
    if 5<= h < 11:
        return "Morning"
    if 11<= h < 17:
        return "Midday"
    if 17<= h < 23:
        return "Evening"
    return "Night"

df["time_of_day"] = df["hour"].apply(tod)

distance_by_tod = (
    df[df["time_of_day"].isin(["Morning","Midday","Evening"])]
    .groupby("time_of_day", as_index=False)["distance"].sum()
    .sort_values("time_of_day")
)


weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
df["weekday"] = pd.Categorical(df["weekday"], categories=weekday_order, ordered=True)

distance_by_weekday = (
    df.groupby("weekday", as_index=False, observed=False)["distance"]
      .sum()
      .sort_values("weekday")
)


START = pd.Timestamp("2024-11-01")
df["week_num"] = ((df["date"] - START).dt.days // 7) + 1

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
        html.H1("Clarkson Crew Performance Dashboard", className="dashboard-header"),
        html.H4("Data from Winter 24/25", className="dashboard-header"),

        dcc.Graph(
            id='cumulative-distance-graph',
            figure=px.line(
                cumulative_df,
                x='date',
                y='cumulative_distance',
                color='name',
                title="Total Team Cumulative Distance by Rower",
                labels={'cumulative_distance': 'Cumulative Distance', 'date': 'Date', 'name': 'Rower'}
            ),
            className="graph"
        ),

        html.Div(
            className="charts-row",
            children=[
                dcc.Graph(
                    id='time-of-day-pie-chart',
                    figure=px.pie(
                        distance_by_tod,
                        names="time_of_day",
                        values="distance",
                        title="Distance by Time of Day"
                    ),
                    className="chart"
                ),
                dcc.Graph(
                    id='weekday-bar-chart',
                    figure=px.bar(
                        distance_by_weekday,
                        x="weekday",
                        y="distance",
                        title="Distance by Weekday",
                        labels={'weekday': 'Weekday', 'distance': 'Distance'}
                    ),
                    className="chart"
                )
            ]
        ),

        html.H2("Leaderboard", className="leaderboard-header"),
        html.Div(
            className="leaderboard-container",
            children=[
                html.Table(
                    children=[
                        html.Tr([html.Th("Week"), html.Th("Team's meters rowed"), html.Th("Most Rowed")])
                    ] + [
                        html.Tr([html.Td(int(row.week_num)), html.Td(int(row.distance)), html.Td(row.most_rowed)])
                        for row in weekly_leaderboard.itertuples(index=False)
                    ]
                )
            ]
        ),
        html.Div(
            className="dropdown-row",
            children=[
                html.Div(
                    className="dropdown",
                    children=[
                        html.Label("Select Rower:"),
                        dcc.Dropdown(
                            id='rower-dropdown',
                            options=rower_options,
                            value=(rower_options[0]['value'] if rower_options else None)
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
