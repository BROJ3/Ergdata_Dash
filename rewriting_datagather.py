import time
import json
import random
import logging
from datetime import datetime, timedelta
import sqlite3
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter                      
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed #Allows you to download multiple workouts at the same time.
from threading import Semaphore                                 #Limits how many downloads happen simultaneously.
from collections import deque
import config

access_token = config.access_token

BATCH_SIZE = 10                                                 #insert 10 workouts at a time into db
PER_PAGE = 250                                                  #better get too many than make too many requests
PAGE_SLEEP_SEC = 0.25                                           #pause between requests
JITTER_MIN, JITTER_MAX = 0.05, 0.20                             #humanizes requests
ROWER_PARALLELISM = 2                                           #ensures we don't download more than 2 workouts at a time for the same rower at a time
GLOBAL_WORKERS     = 16                                         #16 workers in threadpoolexecutor
STROKE_CONCURRENCY = 3                                          # three stroke API's at the same time
STROKE_SEM = Semaphore(STROKE_CONCURRENCY)                      #regulates stroke request traffic

DEFAULT_FROM = "2025-10-21"                            
DEFAULT_TO   = "2026-03-01"

#logging setup - prints with timestamps
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("concept2")

 #establish a reusable internet connection
def make_session():         

    session = requests.Session()                                         
    session.headers.update({'Authorization': f'Bearer {access_token}'})
    retry_settings = Retry( total=5, backoff_factor=0.5,                                          #wait twice as much as previous time before trying again
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"), respect_retry_after_header=True )
    adapter = HTTPAdapter(max_retries=retry_settings)                                             #place the retry logic into the session
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session

#DATABASE LOGIC