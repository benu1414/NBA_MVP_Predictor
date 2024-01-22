# %%
import pandas as pd

# %%
stats = pd.read_csv("player_mvp_stats.csv")
stats

# %%
del stats["Unnamed: 0"]

# %%
pd.isnull(stats).sum()

# %%
#select all the rows in stats where 3P% is null
stats[pd.isnull(stats["3P%"])][["Player","3PA"]]

# %%
stats[pd.isnull(stats["FT%"])][["Player","FTA"]]

# %%
# replace any null value with 0 (technically not correct b/c attempting 0 3's != 0 3P%)
stats = stats.fillna(0)
pd.isnull(stats).sum()

# %%
# ML TIME- trying to predict mvp share (pts won / pts max)
stats.columns

# %%
# remove strings and share, pts won, pts max -> b/c they are what we're trying to predict
predictors = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
       '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year',
       'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS']

# %%
train = stats[stats["Year"] < 2021]
test = stats[stats["Year"] == 2021]

# %%
# Ridge is a form of linear regression that is designed to prevent overfitting
# overfitting: data is fits so well with current data, but doesn't work when we apply it to predict
from sklearn.linear_model import Ridge

reg = Ridge(alpha=0.1)

# %%
reg.fit(train[predictors], train["Share"])
Ridge(alpha=0.1)
predictions = reg.predict(test[predictors])
predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
predictions

# %%
# concat combines two data sets or data frames in panda
combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
combination

# %%
combination.sort_values("Share", ascending=False).head(10)

# %%
# error metric - let's us know if the algorithm did well or not
from sklearn.metrics import mean_squared_error
# the mean difference between the prediciton and the actual share
mean_squared_error(combination["Share"], combination["predictions"])

# %%
combination["Share"].value_counts()

# %%
combination = combination.sort_values("Share", ascending = False)
# df.shape[0] gives you the number of rows and df.shape[1] gives you the number of columns.
combination["Rk"] = list(range(1,combination.shape[0] + 1))
combination.head(10)

# %%
combination = combination.sort_values("predictions", ascending = False)
combination["Predicted_Rk"] = list(range(1, combination.shape[0] + 1))
combination.head(10)

# %%
combination.sort_values("Share", ascending = False).head(10)

# %%
# error metric that measures how many of the top 5 we were able to get correct
# this error metric sees how far down we have to go in predicted_rk until we get to the actual rk
# error metric function
def find_ap(combination):
    actual = combination.sort_values("Share", ascending = False).head(5)
    predicted =  combination.sort_values("predictions", ascending = False)
    ps = []
    found = 0
    seek = 1
    for index, row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            ps.append(found/seek)
        seek += 1
    return sum(ps) / len(ps)
find_ap(combination)

# %%
years = list(range(2003, 2022))
aps = []
all_predictions = []
# years 2003 to 2008 is the training set and 2009 is test set, and so on
for year in years[5:]:
    train = stats[stats["Year"] < year]
    test = stats[stats["Year"] == year]
    reg.fit(train[predictors], train["Share"])
    predictions = reg.predict(test[predictors])
    predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
    combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
    all_predictions.append(combination)
    aps.append(find_ap(combination))
sum(aps) / len(aps)

# %%
def add_ranks(combination):
    combination = combination.sort_values("predictions", ascending=False)
    combination["Predicted_Rk"] = list(range(1,combination.shape[0]+1))
    combination = combination.sort_values("Share", ascending=False)
    combination["Rk"] = list(range(1,combination.shape[0]+1))
    combination["Diff"] = (combination["Rk"] - combination["Predicted_Rk"])
    return combination
ranking = add_ranks(all_predictions[1])
ranking[ranking["Rk"] < 6].sort_values("Diff", ascending=False)

# %%
def backtest(stats, model, years, predictors):
    aps = []
    all_predictions = []
    for year in years:
        train = stats[stats["Year"] < year]
        test = stats[stats["Year"] == year]
        model.fit(train[predictors],train["Share"])
        predictions = model.predict(test[predictors])
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
        combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
        combination = add_ranks(combination)
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    return sum(aps) / len(aps), aps, pd.concat(all_predictions)
mean_ap, aps, all_predictions = backtest(stats, reg, years[5:], predictors)
mean_ap

# %%
all_predictions[all_predictions["Rk"] <= 5].sort_values("Diff").head(10)

# %%
pd.concat([pd.Series(reg.coef_), pd.Series(predictors)], axis=1).sort_values(0, ascending=False)

# %%
stat_ratios = stats[["PTS", "AST", "STL", "BLK", "3P"]].apply(lambda x: x/x.mean())
stat_ratios

# %%
stats[["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]] = stat_ratios[["PTS", "AST", "STL", "BLK", "3P"]]
stats.head()

# %%
predictors += ["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]
mean_ap, aps, all_predictions = backtest(stats, reg, years[5:], predictors)
mean_ap

# %%
stats["NPos"] = stats["Pos"].astype("category").cat.codes
stats["NTm"] = stats["Tm"].astype("category").cat.codes


# %%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=1, min_samples_split=5)
mean_ap, aps, all_predictions = backtest(stats, rf, years[16:], predictors)
mean_ap

# %%
mean_ap, aps, all_predictions = backtest(stats, reg, years[16:], predictors)
mean_ap

# %%



