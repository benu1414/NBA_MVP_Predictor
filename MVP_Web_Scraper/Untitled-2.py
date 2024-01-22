# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
mvps = pd.read_csv("mvps.csv")
mvps

# %%
mvps = mvps[["Player", "Year", "Pts Won", "Pts Max", "Share"]]
mvps.head()

# %%
players = pd.read_csv("players.csv")
players

# %%
del players["Unnamed: 0"]
del players["Rk"]

# %%
players.head()

# %%
players["Player"].head(50)

# %%
#getting rid of the * around players (idk what * means, it might indicate they've been nominated into hall of fame)
players["Player"] = players["Player"].str.replace("*", "", regex=False)
players.head(50)

# %%
#make sure each player has only one row- there will be multiple rows if a player played more multiple teams in a season
players.groupby(["Player", "Year"]).get_group(("Ray Allen", 2003))

# %%
def single_team(df):
    if df.shape[0]==1:
        return df
    else:
        #only grab the total for that row and assign it the last team the player has played for that season
        row = df[df["Tm"]=="TOT"]
        row["Tm"] = df.iloc[-1,:]["Tm"]
        return row

players = players.groupby(["Player", "Year"]).apply(single_team)
players.head(20)
    

# %%
players.index = players.index.droplevel()
players.index = players.index.droplevel()
players

# %%
#merge the two data frames
combined = players.merge(mvps, how="outer", on=["Player", "Year"])
combined

# %%
combined[combined["Pts Won"] > 0]

# %%
# changing all the mvp pts won from NaN to 0
combined[["Pts Won", "Pts Max", "Share"]] = combined[["Pts Won", "Pts Max", "Share"]].fillna(0)
combined

# %%
teams = pd.read_csv("teams.csv")
teams

# %%
teams.head(30)

# %%
# get rid of the division rows
teams = teams[~teams["W"].str.contains("Division")]
# get rid of the asteriks - signifies if a team made playoffs
teams["Team"] = teams["Team"].str.replace("*", "", regex=False)
teams.head(5)

# %%
teams["Team"].unique()

# %%
combined["Tm"].unique()

# %%
abbreviations = {}

with open("abbreviations.csv", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines[1:]:
        abbrev,name = line.replace("\n","").split(",")
        abbreviations[abbrev] = name

# %%
combined["Team"] = combined["Tm"].map(abbreviations)
combined.head()

# %%
stats = combined.merge(teams, how="outer", on=["Team", "Year"])
stats

# %%
combined

# %%
del stats["Unnamed: 0"]
stats.dtypes

# %%
# convert data types to numerical values
stats = stats.apply(pd.to_numeric, errors="ignore")
stats.dtypes

# %%
stats["GB"].unique()

# %%
stats["GB"] = stats["GB"].str.replace("—", "0.0")
stats["GB"].unique()

# %%
stats["GB"] = pd.to_numeric(stats["GB"])
stats.dtypes

# %%
stats.to_csv("player_mvp_stats.csv")

# %%
highest_scoring = stats[stats["G"] > 70].sort_values("PTS", ascending=False).head(10)
highest_scoring.plot.bar("Player","PTS")

# %%
highest_scoring = stats.groupby("Year").apply(lambda x: x.sort_values("PTS", ascending=False).head(1))
# same thing as the lambda function
# def highest_pts(df):
#     return x.sort_values("PTS", ascending=False).head(1)
highest_scoring.plot.bar("Year", "PTS")

# %%
numeric_stats = stats.select_dtypes(include=["float64", "int64"])
correlations = numeric_stats.corr()["Share"]
correlations

# %%
correlations.plot.bar()


