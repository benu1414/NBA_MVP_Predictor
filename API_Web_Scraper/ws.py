import pandas as pd
import requests
pd.set_option('display.max_columns', None) #shows all coluns in a wide DataFrame
import time
import numpy as np

test_url = "https://stats.nba.com/stats/leagueLeaders?LeagueID=00&PerMode=PerGame&Scope=S&Season=2022-23&SeasonType=Regular%20Season&StatCategory=PTS"
r = requests.get(url=test_url).json()
table_headers = r["resultSet"]["headers"]
pd.DataFrame(r["resultSet"]["headers"])
