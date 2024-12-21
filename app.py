import sqlite3
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
import json

connection = sqlite3.connect('team_data.db')

# Create a cursor object
cursor = connection.cursor()

query = """
    SELECT name, distance, date, weekday, hour, time, stroke_data
    FROM crnjakt_workouts
"""
cursor.execute(query)

rows = cursor.fetchall()

data = []

#defining data
for row in rows:
    name = row[0]
    distance = row[1]
    date = row[2]
    weekday = row[3]
    hour = row[4]
    time = row[5]
    stroke_data = row[6]

    if stroke_data:
        stroke_data = json.loads(stroke_data)
    else:
        stroke_data = None

    data.append({
        'name': name,
        'distance': float(distance),
        'date': datetime.strptime(str(date), "%Y-%m-%d"),
        'weekday': weekday,
        'hour': datetime.strptime(str(hour), "%H:%M:%S").hour,
        'time': int(time),
        'stroke_data': stroke_data
    })

data.sort(key=lambda x: (x['name'], x['date']))



# Cumulative distance data
cumulative_data = []
cumulative_tracker = {}

time_categories = {"Morning": (5, 11), "Midday": (11, 17), "Evening": (17, 23)}

distance_by_time_of_day = {
    "Morning": 0.0,
    "Midday": 0.0,
    "Evening": 0.0,
}

distance_by_weekday = {}
weekly_leaderboard = {}
rower_weekly_totals = {}

for entry in data:
    name = entry['name']
    date = entry['date']
    hour = entry['hour']
    weekday = entry['weekday']
    distance = entry['distance']

    if name not in cumulative_tracker:
        cumulative_tracker[name] = 0.0

    cumulative_tracker[name] += distance

    cumulative_data.append({
        'name': name,
        'date': date,
        'cumulative_distance': cumulative_tracker[name]
    })

    #daytime segregation of workouts
    if 5 <= hour < 11:
         distance_by_time_of_day["Morning"] += distance
    elif 11 <= hour < 17:
         distance_by_time_of_day["Midday"] += distance
    elif 17 <= hour < 23:
         distance_by_time_of_day["Evening"] += distance

    #weekday segregation of workouts
    if weekday not in distance_by_weekday:
        distance_by_weekday[weekday] = 0
    distance_by_weekday[weekday] += distance

    start_date = datetime(2024, 11, 1)
    week = ((date - start_date).days // 7) + 1

    if week not in rower_weekly_totals:
        rower_weekly_totals[week] = {}

    if name not in rower_weekly_totals[week]:
        rower_weekly_totals[week][name] = 0

    rower_weekly_totals[week][name] += distance

cumulative_data.sort(key=lambda x: (x['name'], x['date']))

for week, totals in rower_weekly_totals.items():
    if week not in weekly_leaderboard:
        weekly_leaderboard[week] = {'distance': 0, 'most_rowed': None}

    weekly_leaderboard[week]['distance'] = sum(totals.values())
    weekly_leaderboard[week]['most_rowed'] = max(totals, key=totals.get)

with open('cumulative_data.json', 'w') as f:
    json.dump(cumulative_data, f, default=str)  # Convert datetime to string

# Save leaderboard data to a JSON file
with open('weekly_leaderboard.json', 'w') as f:
    json.dump(weekly_leaderboard, f)

# Save distance by time of day to a JSON file
with open('distance_by_time_of_day.json', 'w') as f:
    json.dump(distance_by_time_of_day, f)

# Save distance by weekday to a JSON file
with open('distance_by_weekday.json', 'w') as f:
    json.dump(distance_by_weekday, f)



app = dash.Dash(__name__)

unique_rowers = set()
for entry in data:
    unique_rowers.add(entry['name'])

rower_options = []
for rower in unique_rowers:
    rower_options.append({'label': rower, 'value': rower})


app.layout = html.Div(

    className="dashboard-container",
    children=[
        html.H1("Clarkson Crew Performance Dashboard", className="dashboard-header"),

        dcc.Graph(
            id='cumulative-distance-graph',
            figure=px.line(
                cumulative_data,
                x='date',
                y='cumulative_distance',
                color='name',
                title="Cumulative Distance by Rower",
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

        html.H2("Leaderboard", className="leaderboard-header"),
        html.Div(
            className="leaderboard-container",
            children=[
                html.Table(
                    children=[
                        html.Tr([html.Th("Week"), html.Th("Total Distance"), html.Th("Most Rowed")])
                    ] + [
                        html.Tr([html.Td(week), html.Td(data['distance']), html.Td(data['most_rowed'])])
                        for week, data in sorted(weekly_leaderboard.items())
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

        dcc.Graph(id='workout-stroke-graph')
    
    ]
)

@app.callback(
    Output('workout-dropdown', 'options'),
    Input('rower-dropdown', 'value')
)
def update_workout_dropdown(selected_rower):
    options = []
    for entry in data:
        if entry['name'] == selected_rower and entry['stroke_data']:
            label = f"{entry['date'].strftime('%Y-%m-%d')} - {entry['distance']}m"
            value = json.dumps(entry['stroke_data'])
            options.append({'label': label, 'value': value})

    return options

app.clientside_callback(
    """
    function(selectedWorkout, selectedMetric) {
        if (!selectedWorkout) {
            return {
                data: [],
                layout: { title: 'No Data Available' }
            };
        }

        var strokeData = JSON.parse(selectedWorkout);  // Parse stroke data
        var strokes = strokeData.data;
        var strokeX = [...Array(strokes.length).keys()];  // Generate x-axis values
        var strokeY = strokes.map(point => point[selectedMetric] || 0);  // Extract selected metric

        // Handle metric-specific options
        var tickvals = null;
        var ticktext = null;
        if (selectedMetric === 'p') {
            tickvals = [];
            ticktext = [];
            [...new Set(strokeY)].forEach(val => {
                var minutes = Math.floor(val / 600);
                var seconds = Math.floor((val % 600) / 10);
                tickvals.push(val);
                ticktext.push(`${minutes}:${seconds.toString().padStart(2, '0')}`);
            });
        }

        return {
            data: [{
                x: strokeX,
                y: strokeY,
                type: 'scatter',
                mode: 'lines+markers',
                name: selectedMetric
            }],
            layout: {
                title: `${selectedMetric === 'spm' ? 'Strokes Per Minute' : selectedMetric === 'hr' ? 'Heart Rate' : 'Pace'} for Selected Workout`,
                yaxis: {
                    tickmode: tickvals ? 'array' : undefined,
                    tickvals: tickvals,
                    ticktext: ticktext,
                    autorange: selectedMetric === 'p' ? 'reversed' : true
                }
            }
        };
    }
    """,
    Output('workout-stroke-graph', 'figure'),
    [Input('workout-dropdown', 'value'), Input('metric-dropdown', 'value')]
)


if __name__ == '__main__':
    app.run_server(debug=True)
'''
    with open('index.html', 'w') as f:
            f.write(app.get_dist('static'))'''

#if __name__ == '__main__':
#    app.run_server(debug=True, host='0.0.0.0', port=8050) #for sharing - run this to be visible to everyone on the network
