import pandas as pd
import numpy as np
#scrape basketball reference and clean datasets, use 2018-19 as train data to predict current all-stars.
df = pd.read_html("https://www.basketball-reference.com/leagues/NBA_2020_totals.html#totals_stats::fg")
df_past1 = pd.read_html("https://www.basketball-reference.com/leagues/NBA_2019_totals.html")
df_past2 = pd.read_html("https://www.basketball-reference.com/leagues/NBA_2018_totals.html")
print(df[0].shape)
current_year = df[0]
all_names_current = current_year['Player']
last_year = df_past1[0]
all_names_last = last_year['Player']
two_years_ago = df_past2[0]
all_names_two = two_years_ago['Player']

#train_data = pd.concat((last_year, two_years_ago))
train_labels_two = pd.read_html("http://bkref.com/tiny/Q2VZD")
train_labels_two = train_labels_two[0]
train_labels_last = pd.read_html("http://bkref.com/tiny/QOdIc")
train_labels_last = train_labels_last[0]
test_labels_now = pd.read_html("http://bkref.com/tiny/7yydr")
test_labels_now = test_labels_now[0]
names_now = test_labels_now.iloc[:,1]
all_star_index = np.zeros((two_years_ago.shape[0]))
test_index = np.zeros((current_year.shape[0]))
names_last = train_labels_last.iloc[:,1]
names_two = train_labels_two.iloc[:,1]
names_two = names_two[names_two != 'Player']
names_two.index = range(len(names_two))
for i in range(0, two_years_ago.shape[0]):
    for j in range(0, len(names_two)):
        if all_names_two[i] == names_two[j]:
            all_star_index[i] +=1
for k in range(0, current_year.shape[0]):
    for l in range(0, len(names_now)):
        if all_names_current[k] == names_now[l]:
            test_index[k] +=1
two_years_ago['Result'] = all_star_index
current_year['Result'] = test_index
#two_years_ago.to_csv("2017-18_NBA_Season_all_Star.csv", index = False)
current_year.to_csv("Current_NBA_Season_all_Star.csv", index = False)



