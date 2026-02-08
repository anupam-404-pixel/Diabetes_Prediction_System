from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import os
from django.conf import settings

CSV_PATH = os.path.join(
    settings.BASE_DIR,
    'static',
    'DiabetesPrediction',
    'csv file',
    'diabetes.csv'
)
df = pd.read_csv(CSV_PATH)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
xgboostmodel = GradientBoostingClassifier()
xgboostmodel.fit(X_train, y_train)

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    # df = pd.read_csv(CSV_PATH)
    # # df
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1]
    #
    # sc = StandardScaler()
    # X = sc.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # xgboostmodel = GradientBoostingClassifier()
    # xgboostmodel.fit(X_train, y_train)

    val1 = float(request.POST['n1'])
    val2 = float(request.POST['n2'])
    val3 = float(request.POST['n3'])
    val4 = float(request.POST['n4'])
    val5 = float(request.POST['n5'])
    val6 = float(request.POST['n6'])
    val7 = float(request.POST['n7'])
    val8 = float(request.POST['n8'])

    user_data = sc.transform([[val1, val2, val3, val4, val5, val6, val7, val8]])
    pred = xgboostmodel.predict(user_data)
    result1 = ""
    if pred == [1]:
        result1 = " Diabetes Detected"
    else:
        result1 = " No Diabetes Detected"


    return render(request,'predict.html' , {'result2':result1})
