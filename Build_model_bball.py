import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#figure out what model to build, get test data as well
#remove all rows with nonnumeric data, have to adjust the target array as well
#first build a linear model
df = pd.read_csv("2017-18_NBA_Season_all_Star.csv")
df1 = pd.read_csv("Current_NBA_Season_all_Star.csv")
#isinstance(df, (pd.DataFrame))
df = df.fillna(value = 0.0)
df1 = df1.fillna(value = 0.0)
test_targets_1 = df1.iloc[:, np.r_[5:len(df1.columns)-1]]
test_current_labels = df1['Result']
print(df1.columns)
targets = df.iloc[:, np.r_[5:len(df.columns)-1]]
#train_targets = train_targets._get_numeric_data()
labels = df['Result']
print(labels.shape, targets.shape)
#train_labels = train_labels.to_numpy()
gnb = GaussianNB()
log = LogisticRegression(penalty='l2')
log_reg_model = log.fit(targets, labels)
Naive_bayes_model = gnb.fit(targets,labels)
pred_log = log.predict(test_targets_1)
preds = gnb.predict(test_targets_1)
print(pred_log.shape)
print(test_current_labels.shape)
print(test_current_labels)
accuracy = accuracy_score(preds, test_current_labels)
accuracy_log = accuracy_score(pred_log, test_current_labels)
print("The accuracy for Naive Bayes is", accuracy, "and the accuracy for Logistic Regression is:", accuracy_log)
#pred_2 = gnb.predict(test_targets_1)
#pred_2_log = log_reg_model.predict(test_targets_1)
joblib.dump(log_reg_model, "log_reg_model.pkl")
output = pd.DataFrame()
output['Player'] = df1['Player']
output['Allstar'] = pred_log
#output.to_csv("NBA_All_Star_Predictions.csv", index = False)
print(output.shape)
#print("Predictions for 2019-20 by logistic regression have accuracy of {}, while Naive Bayes has accuracy of {}".format(accuracy_score(pred_2_log, test_current_labels), accuracy_score(pred_2, test_current_labels)))
