import sqlite3
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime, timedelta
import json

# Constants
TIME_CATEGORIES = {
    "Morning": (5, 11),
    "Midday": (11, 17),
    "Evening": (17, 23)
}

# Database connection
connection = sqlite3.connect('team_data.db')
cursor = connection.cursor()

# Fetch data
query = """
    SELECT name, distance, date, weekday, hour, time, stroke_data
    FROM crnjakt_workouts
"""
cursor.execute(query)
rows = cursor.fetchall()

# Helper function to parse rows
def parse_row(row):
    stroke_data = json.loads(row[6]) if row[6] else None
    return {
        'name': row[0],
        'distance': float(row[1]),
        'date': datetime.strptime(str(row[2]), "%Y-%m-%d"),
        'weekday': row[3],
        'hour': datetime.strptime(str(row[4]), "%H:%M:%S").hour,
        'time': int(row[5]),
        'stroke_data': stroke_data
    }

# Parse data
data = [parse_row(row) for row in rows]
data.sort(key=lambda x: (x['name'], x['date']))

# Initialize data trackers
cumulative_data = []
cumulative_tracker = {}

recent_workouts = []
recent_cumulative_data = []
recent_cumulative_tracker = {}

distance_by_time_of_day = {key: 0.0 for key in TIME_CATEGORIES}
distance_by_weekday = {}

# Define two weeks ago
two_weeks_ago = datetime.today() - timedelta(days=14)

# Populate trackers
for entry in data:
    name = entry['name']
    date = entry['date']
    distance = entry['distance']
    hour = entry['hour']
    weekday = entry['weekday']

    # Update cumulative distance
    if name not in cumulative_tracker:
        cumulative_tracker[name] = 0.0
    cumulative_tracker[name] += distance
    cumulative_data.append({'name': name, 'date': date, 'cumulative_distance': cumulative_tracker[name]})

    # Track recent workouts
    if date > two_weeks_ago:
        if name not in recent_cumulative_tracker:
            recent_cumulative_tracker[name] = 0.0
        recent_cumulative_tracker[name] += distance
        recent_cumulative_data.append({'name': name, 'date': date, 'cumulative_distance': recent_cumulative_tracker[name]})
        recent_workouts.append(entry)

    # Distance by time of day
    for category, (start, end) in TIME_CATEGORIES.items():
        if start <= hour < end:
            distance_by_time_of_day[category] += distance

    # Distance by weekday
    distance_by_weekday[weekday] = distance_by_weekday.get(weekday, 0) + distance

# Dash app setup
app = dash.Dash(__name__)
server = app.server

# Unique rower options
rower_options = [{'label': name, 'value': name} for name in sorted(set(entry['name'] for entry in data))]

# App layout
app.layout = html.Div(
    className="dashboard-container",
    children=[
        html.H1("Clarkson Crew Performance Dashboard", className="dashboard-header"),

        # Recent cumulative distance graph
        dcc.Graph(
            id='recent-cumulative-distance-graph',
            figure=px.line(
                recent_cumulative_data,
                x='date',
                y='cumulative_distance',
                color='name',
                title="Team Distance in the Past 14 Days",
                labels={'cumulative_distance': 'Cumulative Distance', 'date': 'Date', 'name': 'Rower'}
            ),
            className="graph"
        ),

        # Charts for time of day and weekday distance
        html.Div(
            className="charts-row",
            children=[
                dcc.Graph(
                    id='time-of-day-pie-chart',
                    figure=px.pie(
                        names=list(distance_by_time_of_day.keys()),
                        values=list(distance_by_time_of_day.values()),
                        title="Distance by Time of Day"
                    ),
                    className="chart"
                ),
                dcc.Graph(
                    id='weekday-bar-chart',
                    figure=px.bar(
                        x=list(distance_by_weekday.keys()),
                        y=list(distance_by_weekday.values()),
                        title="Distance by Weekday",
                        labels={'x': 'Weekday', 'y': 'Distance'}
                    ),
                    className="chart"
                )
            ]
        ),

        # Total cumulative distance graph
        dcc.Graph(
            id='cumulative-distance-graph',
            figure=px.line(
                cumulative_data,
                x='date',
                y='cumulative_distance',
                color='name',
                title="Total Team Cumulative Distance by Rower",
                labels={'cumulative_distance': 'Cumulative Distance', 'date': 'Date', 'name': 'Rower'}
            ),
            className="graph"
        ),

        # Dropdowns for workout analysis
        html.Div(
            className="dropdown-row",
            children=[
                html.Div(
                    className="dropdown",
                    children=[
                        html.Label("Select Rower:"),
                        dcc.Dropdown(id='rower-dropdown', options=rower_options, value=rower_options[0]['value']),
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

        # Workout stroke graph
        dcc.Graph(id='workout-stroke-graph')
    ]
)

# Callback to update workout dropdown
@app.callback(
    Output('workout-dropdown', 'options'),
    Input('rower-dropdown', 'value')
)
def update_workout_dropdown(selected_rower):
    options = [
        {
            'label': f"{entry['date'].strftime('%Y-%m-%d')} - {entry['distance']}m",
            'value': json.dumps(entry['stroke_data'])
        }
        for entry in data if entry['name'] == selected_rower and entry['stroke_data']
    ]
    return sorted(options, key=lambda x: x['label'], reverse=True)

# Callback to update workout stroke graph
@app.callback(
    Output('workout-stroke-graph', 'figure'),
    [Input('workout-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_workout_graph(selected_workout, selected_metric):
    if not selected_workout:
        return px.line(title="No Data Available")

    stroke_data = json.loads(selected_workout)['data']
    stroke_x = list(range(len(stroke_data)))
    stroke_y = [point.get(selected_metric, 0) for point in stroke_data]

    if selected_metric == 'p':
        tickvals = sorted(set(stroke_y))
        ticktext = [f"{val // 600}:{(val % 600) // 10:02d}" for val in tickvals]
        yaxis_config = dict(tickmode="array", tickvals=tickvals, ticktext=ticktext, autorange="reversed")
    else:
        yaxis_config = {}

    return px.line(
        x=stroke_x,
        y=stroke_y,
        title=f"{selected_metric} for Selected Workout",
        labels={'x': 'Stroke Number', 'y': selected_metric}
    ).update_layout(yaxis=yaxis_config)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
