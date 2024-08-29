from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
def load_model():
    data = pd.read_csv('titanic.csv')
    
    
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
    
    data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
    
    X = data[['Pclass', 'Sex', 'Age', 'Fare']]
    y = data['Survived']
    
    model = RandomForestClassifier()
    model.fit(X, y)
    
    return model

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    fare = float(request.form['fare'])
    
    features = [[pclass, sex, age, fare]]
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        result = "Survived"
    else:
        result = "Did Not Survive"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
