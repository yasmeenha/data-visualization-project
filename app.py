from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request 
import pandas as pd
import numpy as np
import joblib
#importing utility libraries
import math
import warnings
import string
#importing classifier
from sklearn.ensemble import RandomForestClassifier
#importing model
from model import Chart
from obesity import Obesity

# Define a Flask app
app = Flask(__name__)

global gl_fig, low, high
# sklearn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


#ignoring all warinings

warnings.filterwarnings("ignore")  

df = pd.read_csv("obesity_data.csv",delimiter=",",header= 0)
df.head()

df_cleaned = df 

df_scaled=df_cleaned.copy()

columns_to_scale = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']

scaler = StandardScaler()
df_scaled[columns_to_scale] = scaler.fit_transform(df_cleaned[columns_to_scale])

df_scaled.head()

#we perform some Standardization using minmaxscaler
df_scaled_mm=df_cleaned.copy()

columns_to_scale_mm = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']

mmscaler = MinMaxScaler()
df_scaled_mm[columns_to_scale_mm] = mmscaler.fit_transform(df_cleaned[columns_to_scale_mm])

df_scaled_mm.head()


X = df_cleaned.drop(['NObeyesdad'], axis=1) #features 
y = df_cleaned['NObeyesdad']  #target feature

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)


#Train-test-split for scaled data
X_scaled = df_scaled.drop(['NObeyesdad'], axis=1) #features 
y_scaled = df_scaled['NObeyesdad']  #target feature

X_scaled_mm = df_scaled_mm.drop(['NObeyesdad'], axis=1) #features 
y_scaled_mm = df_scaled_mm['NObeyesdad']  #target feature

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle = True)
X_train_scaled_mm, X_test_scaled_mm, y_train_scaled_mm, y_test_scaled_mm = train_test_split(X_scaled_mm, y_scaled_mm, test_size=0.2, random_state=42, shuffle = True)

rndForestClassifier = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [500], 'max_features':['sqrt'], 'max_depth':[20], 'max_leaf_nodes':[2,5,10,50,100,200,300,400,500,750,1000]}, cv=5, scoring=['accuracy','recall'], refit='accuracy').fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=500, max_depth=20, max_features='sqrt', max_leaf_nodes=750)
rf.fit(X_train,y_train)

y_predict = rf.predict(X_test)
y_predicted = np.array(y_predict > 0.5, dtype=float)

rndForest_acc = accuracy_score(y_test, y_predicted)
print(rndForest_acc)
cm = confusion_matrix(y_test, y_predicted)
print(cm)
rndForest_tpr = cm[1][1] /(cm[1][0] + cm[1][1])
rndForest_report = classification_report(y_test, y_predicted)
print(rndForest_report)
print(X_test.columns)
print(X_test.head(3))





# Define a route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
#reading the obesity dataset
    d = {'Gender':[], 'Age':[], 'Height':[], 'Weight':[], 'family_history_with_overweight':[], 'FAVC':[], 'FCVC':[],
       'NCP':[], 'CAEC':[], 'SMOKE':[], 'CH2O':[], 'SCC':[], 'FAF':[], 'TUE':[], 'CALC':[], 'MTRANS':[]}
    output = request.form.to_dict()
#appending all the features
    d["Gender"].append(int(output["Gender"]))
    d["Age"].append(int(output["Age"]))
    d["Height"].append(int(output["Height"]))
    d["Weight"].append(int(output["Weight"]))
    d["family_history_with_overweight"].append(int(output["family_history_with_overweight"]))
    d["FAVC"].append(int(output["FAVC"]))
    d["FCVC"].append(int(output["FCVC"]))
    d["NCP"].append(int(output["NCP"]))
    d["CAEC"].append(int(output["CAEC"]))
    d["SMOKE"].append( int(output["SMOKE"]))
    d["CH2O"].append(int(output["CH2O"]))
    d["SCC"].append(int(output["SCC"]))
    d["FAF"].append(int(output["FAF"]))
    d["TUE"].append(int(output["TUE"]))
    d["CALC"].append( int(output["CALC"]))
    d["MTRANS"].append(int(output["MTRANS"]))

    print(d)
    df1 = pd.DataFrame(d)
    pred = rf.predict(df1)

    if pred[0]==1:
      note = "Normal weight"
    elif pred[0]==2:
      note = "Overweight I"
    elif pred[0]==3:
      note = "Overweight II"
    elif pred[0]==4:
      note = "Obesity I"
    elif pred[0]==5:
      note = "Obesity II"
    elif pred[0]==6:
      note = "Obesity III"
    elif pred[0]==7:
      note = "Insufficient Weight"
    else:
      note = "No result"
    return render_template('index.html', prediction= note)
#linking to chart and analyze.html
@app.route('/analyze', methods=['GET'])
def analyze():
    global gl_fig
    chart = Chart()
    fig = chart.main_world() 
    gl_fig = fig.to_html(full_html=False)
    return render_template('analyze.html', fig=gl_fig)
#finding highest obesity and linking to chart and analyze.html
@app.route('/highest')
def highest():
    global gl_fig,low,high
    chart = Chart()
    high = chart.highest()
    high = high.to_html(full_html=False)
    return render_template('analyze.html', fig=gl_fig, high = high)
#finding lowest obesity and linking to chart and analyze.html
@app.route('/lowest')
def lowest():
    global gl_fig,low,high
    chart = Chart()
    low = chart.lowest()
    low = low.to_html(full_html=False)
    return render_template('analyze.html', fig=gl_fig, low = low, high=high)
#finding region and linking to chart and analyze.html
@app.route('/region')
def region():
    global gl_fig,low,high
    chart = Chart()
    reg = chart.regions()
    region = reg.to_html(full_html=False)
    return render_template('analyze.html', fig=gl_fig, high = region, low = low)

#analysing for obesity prediction and linking to obesity.html
@app.route('/obesity', methods=['GET', 'POST'])
def obesity():
    obesity = Obesity()
    if request.method == 'POST':
      option = request.form.get('option')
      if option == 'obesity':
        fig = obesity.obesity()
      elif option == 'Age':
        fig = obesity.age()
      elif option == 'Height':
        fig = obesity.height()
      elif option == 'Weight':
        fig = obesity.weight()
      elif option == 'FHO':
        fig = obesity.fho()
      elif option == 'FAVC':
        fig = obesity.favc()
      elif option == 'FCVC':
        fig = obesity.fcvc()
      elif option == 'NCP':
        fig = obesity.ncp()
      elif option == 'CAEC':
        fig = obesity.caec()
      elif option == 'SMOKE':
        fig = obesity.smoke()
      elif option == 'CH2O':
        fig = obesity.water()
      elif option == 'SCC':
        fig = obesity.calories()
      elif option == 'FAF':
        fig = obesity.physical()
      elif option == 'TUE':
        fig = obesity.tue()
      elif option == 'CALC':
        fig = obesity.alcohol()
      elif option == 'MTRANS':
        fig = obesity.mtrans()
    else:
      fig = obesity.obesity()
    
    figure = fig.to_html(full_html=False)
    return render_template('obesity.html', fig=figure)
#finding male and female obesity and linking to charts.html
@app.route('/chart', methods=['GET', 'POST'])
def chart():
    chart = Chart()
    if request.method == 'POST':
      gender = request.form.get('gender')
      if gender == 'Male':
        fig = chart.male()
      elif gender == 'Female':
        fig = chart.female()
      else:
        fig = chart.world()   
    else:
      fig = chart.world()
    figure = fig.to_html(full_html=False)
    return render_template('charts.html', fig=figure)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
