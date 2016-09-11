import sqlite3
import pandas as pd


with sqlite3.connect('../data/soccer/database.sqlite') as con:
    countries = pd.read_sql_query("SELECT * from Country", con)
    leagues = pd.read_sql_query("SELECT * from League", con)
    matches = pd.read_sql_query("SELECT * from Match", con)
    players = pd.read_sql_query("SELECT * from Player", con)
    player_stats = pd.read_sql_query("SELECT * from Player_Stats", con)
    teams = pd.read_sql_query("SELECT * from Team", con)
