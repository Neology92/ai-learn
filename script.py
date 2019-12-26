import numpy as np
import pandas as pd


# Load data
print("Loading data...")
df = pd.read_csv('./weatherAUS.csv')

# Remove wasted data
df = df.drop(columns=
            ['Cloud9am', 'Cloud3pm',
            'Evaporation', 'Sunshine',
            'Date', 'RISK_MM', 'Location', 
            'WindDir9am', 'WindDir3pm',
            'WindGustDir'], axis=1)

df = df.dropna(how='any')

# Remove stand-off data
print("Formating data...")
from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
df = df[(z<3).all(axis=1)]

# Change string data to numerical values
df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

# Normalize 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
df_scaled = scaler.fit_transform(df.values)
df = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)

# End of pre-processing data #

# EDA
# Check which features are the most important
# ----------------
# from sklearn.feature_selection import SelectKBest, chi2
# X = df.loc[:,df.columns!='RainTomorrow']
# Y = df[['RainTomorrow']]
# selector = SelectKBest(chi2, k=3)
# X_new = selector.fit_transform(X, Y)


df = df[['Humidity3pm', 'Rainfall', 'RainToday', 'RainTomorrow']]

# Get only one feature because of learning proposes
X = df.loc[:,df.columns!='RainTomorrow']
# X = df[['Humidity3pm']]
Y = np.ravel(df[['RainTomorrow']])


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
clf_logreg = LogisticRegression(random_state=0)
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
clf_dt = DecisionTreeClassifier(random_state=0)
clf_svm = svm.SVC(kernel='linear')

print("Training logreg...")
t0_logreg = time.time()
clf_logreg.fit(X_train, Y_train)
t1_logreg = time.time() - t0_logreg

print("Testing logreg...")
Y_pred = clf_logreg.predict(X_test)
score = accuracy_score(Y_test, Y_pred)
print("")
print('Accuracy using LogReg: ', score)
print('Time taken using LogReg: ', t1_logreg)

print("\n====================================")

print("Training rf...")
t0_rf = time.time()
clf_rf.fit(X_train, Y_train)
t1_rf = time.time() - t0_rf

print("Testing rf...")
Y_pred = clf_rf.predict(X_test)
score = accuracy_score(Y_test, Y_pred)
print("")
print('Accuracy using rf: ', score)
print('Time taken using rf: ', t1_rf)

print("\n====================================")

print("Training dt...")
t0_dt = time.time()
clf_dt.fit(X_train, Y_train)
t1_dt = time.time() - t0_dt

print("Testing dt...")
Y_pred = clf_dt.predict(X_test)
score = accuracy_score(Y_test, Y_pred)
print("")
print('Accuracy using dt: ', score)
print('Time taken using dt: ', t1_dt)

print("\n====================================")

print("Training svm...")
t0_svm = time.time()
clf_svm.fit(X_train, Y_train)
t1_svm = time.time() - t0_svm

print("Testing svm...")
Y_pred = clf_svm.predict(X_test)
score = accuracy_score(Y_test, Y_pred)
print("")
print('Accuracy using svm: ', score)
print('Time taken using svm: ', t1_svm)




