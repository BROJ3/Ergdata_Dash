import requests
import time
from bs4 import BeautifulSoup
import sqlite3
import json
from datetime import datetime
import config

# Connect to SQLite database (or create it if it doesn't exist)
connection = sqlite3.connect('team_data.db')

# Create a cursor object
cursor = connection.cursor()

#this is for reseting the database
#cursor.execute("DROP TABLE IF EXISTS crnjakt_workouts")
#cursor.execute("DROP TABLE IF EXISTS crnjakt_rowers")

sql='''CREATE TABLE IF NOT EXISTS crnjakt_rowers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    partner_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL
);'''

cursor.execute(sql)

#adding a comment

sql='''
CREATE TABLE IF NOT EXISTS crnjakt_workouts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rower_id VARCHAR(255) NOT NULL,
    workout_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(200),
    type VARCHAR(30),
    distance INT,
    date DATE,
    weekday VARCHAR(30),
    hour TIME,
    time INT,
    timezone VARCHAR(255),
    date_utc DATETIME,
    heart_rate INT,
    calories_total INT,
    stroke_data JSON,
    FOREIGN KEY (rower_id) REFERENCES crnjakt_rowers(partner_id)
);
'''

cursor.execute(sql)


def insert_rower(conn, cursor, rower):
    sql = '''
    INSERT INTO crnjakt_rowers (partner_id, name)
    VALUES (?, ?)
    '''
    try:
        cursor.execute(sql, (
            rower['partner_id'],
            rower['name']
        ))
        conn.commit()

    except Exception as e:
        if e.__class__.__name__ == 'IntegrityError': #this happens in case of a duplicate, which will happen every time
            pass


def insert_workout(conn, cursor, workouts):
        
        rower_ids= {workout[0] for workout in workouts}
        for rower_id in rower_ids:

            #checking if the rower exists in our "rowers" database
            cursor.execute('SELECT partner_id FROM crnjakt_rowers WHERE partner_id = ?', (rower_id,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Partner ID {rower_id} not found in crnjakt_rowers")
        

            sql = '''
            INSERT INTO crnjakt_workouts (rower_id, workout_id,name, type, distance, date, weekday, hour, time, timezone, date_utc, heart_rate, calories_total,stroke_data)
            VALUES (?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?)
            ''' 
            try:
                cursor.executemany(sql, workouts)
                conn.commit()

                
            except Exception as e:
                if e.__class__.__name__ == "IntegrityError":
                    pass
                else:
                    print(f"Error in insert_workout: {e.__class__.__name__} - {e}")



#keeping track of most recent workouts, so that we scrape less data
def get_latest_workout_date(cursor, partner_id):
    sql = "SELECT MAX(date) as latest_date FROM crnjakt_workouts WHERE rower_id = ?"
    cursor.execute(sql, (partner_id,))
    result = cursor.fetchone()
    return result[0]


access_token = config.access_token 

def fetch_data(api_endpoint, headers):
    response = requests.get(api_endpoint, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    

#logging into Concept 2 website to scrape all team members data
login_url = "https://log.concept2.com/login"       
partners_url = "https://log.concept2.com/team/"+config.my_team  

session = requests.Session()

login_payload = config.login_payload

response = session.post(login_url, data=login_payload)


if str(response.status_code) == '200' in response.text:

    print("Login successful.")
    response = session.get(partners_url)
    soup = BeautifulSoup(response.text, "html.parser")
    table_body = soup.find("table", class_="table js-tablesort").find("tbody")

    rowers={}

    for row in table_body.find_all("tr"):
        name_cell = row.find("td")
        profile_link = name_cell.find("a", href=True)    
        if profile_link and "profile" in profile_link["href"]:

            partner_id = profile_link["href"].split("/")[-1]
            name = profile_link.get_text(strip=True)            
            rowers[name] = {
                'name': name,
                'partner_id': partner_id
            }
else:
    print("Login failed.")
    print(response.status_code)

start_time=time.time()

for rower in rowers.items():
    insert_rower(connection, cursor, rower[1])

    rower_id = rower[1]['partner_id']
    latest_date = get_latest_workout_date(cursor,rower_id)
    print(rower[1]['name']+" has latest record on: ", latest_date)


    try:

        if latest_date:
            api_endpoint_range = f'https://log.concept2.com/api/users/{rower_id}/results?from={latest_date}&to=2025-03-01' #+datetime.today().strftime('%Y-%m-%d')

        else:
            api_endpoint_range = f'https://log.concept2.com/api/users/{rower_id}/results?from=2024-11-01&to=2024-12-01'     #2025-03-01' #+datetime.today().strftime('%Y-%m-%d')
        
        data = fetch_data(api_endpoint_range, {'Authorization': f'Bearer {access_token}'})
        
        if not data or 'data' not in data or not data['data']:
            print(f"No workouts found for rower {rower}")
            continue

        name=rower[1]['name']
        workouts=[]

        for workout_id in data['data']:
            dates= workout_id['date'].split(' ')[0]    
            training_id = workout_id.get('id')
            distance = workout_id.get('distance', 0)
            stroke_data_flag = workout_id.get('stroke_data', False)  # True or False


            if not training_id:
                raise KeyError("Missing 'id' in workout data")
            if 'distance' not in workout_id:
                raise KeyError("Missing 'distance' in workout data")
            

            stroke_data=None

            if stroke_data_flag:
                single_workout_endpoint = f'https://log.concept2.com/api/users/{rower_id}/results/{training_id}/strokes'
                time.sleep(1.5)
                single_data = fetch_data(single_workout_endpoint, {'Authorization': f'Bearer {access_token}'})
                if single_data:
                    stroke_data = json.dumps(single_data)

            date_obj = datetime.strptime(workout_id['date'].split(' ')[0], "%Y-%m-%d")
            day_of_week = date_obj.strftime("%A")                       

            try:             
                workouts.append((
                    rower_id,
                    training_id,
                    name,
                    workout_id['type'],
                    distance,
                    workout_id.get('date', '1970-01-01').split(' ')[0],
                    day_of_week,
                    workout_id.get('date', '1970-01-01').split(' ')[1],
                    workout_id.get('time', '00:00:00'),
                    workout_id.get('timezone', 'UTC'),
                    workout_id.get('date_utc'),
                    workout_id.get('heart_rate', {}).get('average', 0),
                    workout_id.get('calories_total', 0),
                    stroke_data
                ))

                if len(workouts) ==1:
                    insert_workout(connection, cursor, workouts)
                    workouts=[]

            except Exception as e:
                print   (f"Error processing workout {workout_id}: {e}")

        insert_workout(connection, cursor, workouts)

    except Exception as e:
        print(f"Error processing rower {rower[1]['name']}: {e}")

end_time=time.time()-start_time

print("Script ran successfully! Elapsed time: ", end_time)
