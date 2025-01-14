import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

dataset_path = 'Financial_inclusion_dataset.csv'
data = pd.read_csv(dataset_path)

label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    if col != 'uniqueid':  # Exclude the unique ID column
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

X = data.drop(['bank_account', 'uniqueid'], axis=1)
y = data['bank_account']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

model_path = "RFmodel.pkl"
joblib.dump(clf, model_path)


clf = joblib.load(model_path)

original_label_mappings = {col: dict(zip(le.classes_, range(len(le.classes_)))) for col, le in label_encoders.items()}

st.title("Bank Account Prediction App")
st.write("Enter the following details to predict whether the person has a bank account or not:")


# User input
country = st.selectbox("Country", options=original_label_mappings['country'].keys())
year = st.number_input("Year", min_value=int(data['year'].min()), max_value=int(data['year'].max()), step=1)
location_type = st.selectbox("Location Type", options=original_label_mappings['location_type'].keys())
cellphone_access = st.selectbox("Cellphone Access", options=original_label_mappings['cellphone_access'].keys())
household_size = st.number_input("Household Size", min_value=int(data['household_size'].min()), max_value=int(data['household_size'].max()), step=1)
age_of_respondent = st.number_input("Age of Respondent", min_value=int(data['age_of_respondent'].min()), max_value=int(data['age_of_respondent'].max()), step=1)
gender_of_respondent = st.selectbox("Gender", options=original_label_mappings['gender_of_respondent'].keys())
relationship_with_head = st.selectbox("Relationship with Head", options=original_label_mappings['relationship_with_head'].keys())
marital_status = st.selectbox("Marital Status", options=original_label_mappings['marital_status'].keys())
education_level = st.selectbox("Education Level", options=original_label_mappings['education_level'].keys())
job_type = st.selectbox("Job Type", options=original_label_mappings['job_type'].keys())


# Prepare user input
user_input = pd.DataFrame({
    'country': [original_label_mappings['country'][country]],
    'year': [year],
    'location_type': [original_label_mappings['location_type'][location_type]],
    'cellphone_access': [original_label_mappings['cellphone_access'][cellphone_access]],
    'household_size': [household_size],
    'age_of_respondent': [age_of_respondent],
    'gender_of_respondent': [original_label_mappings['gender_of_respondent'][gender_of_respondent]],
    'relationship_with_head': [original_label_mappings['relationship_with_head'][relationship_with_head]],
    'marital_status': [original_label_mappings['marital_status'][marital_status]],
    'education_level': [original_label_mappings['education_level'][education_level]],
    'job_type': [original_label_mappings['job_type'][job_type]],
})

# Predict and display result
if st.button("Predict"):
    prediction = clf.predict(user_input)
    result = "Bank Account" if prediction[0] == 1 else "No Bank Account"
    st.write(f"Predicted Bank Account: {result}")