# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""





#**Import Libraries**

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# **Load Data**

dataset=pd.read_csv("D:\SP20-FYP\Final Year Project\ML and AI\Data sets\Disease_symptom_and_patient_profile_dataset.csv")
dataset

#**Len of Disease uniques classes**

len(dataset['Disease'].value_counts())

# **Count values of the unique Disease**

dataset['Disease'].value_counts()

#**Filter classes with samples >=7**


min_samples =7
filtered_classes = dataset['Disease'].value_counts()[dataset['Disease'].value_counts() >= min_samples].index
filtered_data = dataset[dataset['Disease'].isin(filtered_classes)]
filtered_data

len(filtered_data['Disease'].value_counts())

filtered_data['Disease'].value_counts()

# **Balanced the Disease classes using Randomoversampling**

Randomover = RandomOverSampler(random_state=42)
X = filtered_data.drop(['Disease'], axis=1)
y = filtered_data['Disease']
X_resampled, y_resampled = Randomover.fit_resample(X, y)
data= pd.concat([X_resampled, y_resampled], axis=1)
data

#**Encoding categorical variables**


label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

#**Splitting the dataset into features (X) and target variables (y)**


X = data.drop(['Disease', 'Outcome Variable'], axis=1)
y = data[['Disease', 'Outcome Variable']]

#**Histograms for features**

X.hist(bins=15, figsize=(15, 10))
plt.suptitle('Feature Distributions')
plt.show()

#**Heatmap for correlations**


plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()

#**Splitting into training and testing sets**


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#**Standardizing the features**


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#**Creating a MultiOutputClassifier**

multi_label_model = MultiOutputClassifier(RandomForestClassifier(random_state=42))

#**Training the model**


multi_label_model.fit(X_train_scaled, y_train)

#**Making predictions**


multi_label_pred = multi_label_model.predict(X_test_scaled)

#**Evaluating the model**


#**Accuracy For Disease**


accuracy_disease = accuracy_score(y_test['Disease'], multi_label_pred[:, 0])
print("Accuracy for Disease Prediction:", accuracy_disease)

# **Classification report and confusion matrix**

print("Confusion Matrix:\n", confusion_matrix(y_test['Disease'], multi_label_pred[:, 0]))
print("Classification Report:\n", classification_report(y_test['Disease'], multi_label_pred[:, 0]))

#**Plotting F1-score, Recall, and Precision for Disease**


report_disease = classification_report(y_test['Disease'], multi_label_pred[:, 0], output_dict=True)
df_report_disease = pd.DataFrame(report_disease).transpose()

df_report_disease[:-3].plot(y=['f1-score', 'precision', 'recall'], kind='bar', figsize=(12, 6))
plt.title('F1-score, Precision, and Recall for Disease Prediction')
plt.ylabel('Score')
plt.show()

#**Confusion Matrix Visualization**


cm = confusion_matrix(y_test['Disease'], multi_label_pred[:, 0])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g')
plt.title('Confusion Matrix for Disease Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#**For Outcome Variable**


accuracy_outcome = accuracy_score(y_test['Outcome Variable'], multi_label_pred[:, 1])
print("Accuracy for Outcome Variable Prediction:", accuracy_outcome)

#**Classification report and confusion matrix for outcome varible**

print("Confusion Matrix:\n", confusion_matrix(y_test['Outcome Variable'], multi_label_pred[:, 1]))
print("Classification Report:\n", classification_report(y_test['Outcome Variable'], multi_label_pred[:, 1]))

#**confusion Matrix Visualization**


cm = confusion_matrix(y_test['Outcome Variable'], multi_label_pred[:, 1])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g')
plt.title('Confusion Matrix for Outcome Variable Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#**Plotting F1-score, Recall, and Precision for Outcome Variable**


report_outcome = classification_report(y_test['Outcome Variable'], multi_label_pred[:, 1], output_dict=True)
df_report_outcome = pd.DataFrame(report_outcome).transpose()

df_report_outcome[:-3].plot(y=['f1-score', 'precision', 'recall'], kind='bar', figsize=(12, 6))
plt.title('F1-score, Precision, and Recall for Outcome Variable Prediction')
plt.ylabel('Score')
plt.show()

#**Prediction Function**


def predict_disease_and_outcome(input_features):
    scaled_features = scaler.transform([input_features])

    prediction = multi_label_model.predict(scaled_features)

    predicted_disease = label_encoders['Disease'].inverse_transform(prediction[:, 0])
    predicted_outcome = label_encoders['Outcome Variable'].inverse_transform(prediction[:, 1])
    return predicted_disease[0], predicted_outcome[0]

#**Example Usage**


user_symptoms = {
    'Fever': 0,
    'Cough': 1,
    'Fatigue': 1,
    'DifficultyBreathing': 0,
    'Age': 25,
    'Gender': 0,
    'BloodPressure': 2,
    'CholesterolLevel':2,
}

# **Predict the Disease_name and Disease_Happened**




input_features = [user_symptoms[feature] for feature in X.columns]
prediction = predict_disease_and_outcome(input_features)
print("The Disease:",prediction[0],"is", prediction[1])

 
 

from flask import Flask, request, jsonify
 
import numpy as np

app = Flask(__name__)
 
 
  # Start ngrok when the app is run

# Load the trained model and other necessary components
# You need to have the 'multi_label_model', 'scaler', 'label_encoders', and 'X.columns' available from the previous cells.

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the request
        user_symptoms = request.get_json()

        # Preprocess input features
        input_features = [user_symptoms[feature] for feature in X.columns]

        # Scale the input features
        scaled_features = scaler.transform([input_features])

        # Make predictions using the trained model
        prediction = multi_label_model.predict(scaled_features)

        # Inverse transform predictions to get disease and outcome names
        predicted_disease = label_encoders['Disease'].inverse_transform(prediction[:, 0])
        predicted_outcome = label_encoders['Outcome Variable'].inverse_transform(prediction[:, 1])

        # Prepare the response
        response = [
             predicted_disease[0],
             predicted_outcome[0]

        ]


        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0')

 

