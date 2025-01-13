import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
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

#saving the model
model = "RFmodel.pkl"
joblib.dump(clf,model)

#load the model
clf = joblib.load(model)

country_options = data['country'].unique().tolist()
location_type_options = data['location_type'].unique().tolist()
cellphone_access_options = data['cellphone_access'].unique().tolist()
gender_options = data['gender_of_respondent'].unique().tolist()
relationship_options = data['relationship_with_head'].unique().tolist()
marital_status_options = data['marital_status'].unique().tolist()
education_options = data['education_level'].unique().tolist()
job_type_options = data['job_type'].unique().tolist()


# Streamlit UI
st.title("Bank Account Prediction App")

country = st.selectbox("Country", options=country_options)
year = st.number_input("Year", min_value=data['year'].min(), max_value=data['year'].max(), step=1)
location_type = st.selectbox("Location Type", options=location_type_options)
cellphone_access = st.selectbox("Cellphone Access", options=cellphone_access_options)
household_size = st.number_input("Household Size", min_value=data['household_size'].min(), max_value=data['household_size'].max(), step=1)
age_of_respondent = st.number_input("Age of Respondent", min_value=data['age_of_respondent'].min(), max_value=data['age_of_respondent'].max(), step=1)
gender_of_respondent = st.selectbox("Gender", options=gender_options)
relationship_with_head = st.selectbox("Relationship with Head", options=relationship_options)
marital_status = st.selectbox("Marital Status", options=marital_status_options)
education_level = st.selectbox("Education Level", options=education_options)
job_type = st.selectbox("Job Type", options=job_type_options)

user_input = pd.DataFrame({
    'country': [country],
    'year': [year],
    'location_type': [location_type],
    'cellphone_access': [cellphone_access],
    'household_size': [household_size],
    'age_of_respondent': [age_of_respondent],
    'gender_of_respondent': [gender_of_respondent],
    'relationship_with_head': [relationship_with_head],
    'marital_status': [marital_status],
    'education_level': [education_level],
    'job_type': [job_type],
})


for col, le in label_encoders.items(): 
    if col in user_input.columns:
        user_input[col] = le.transform(user_input[col])

prediction = clf.predict(user_input)

if st.button("Predict"):
    prediction = clf.predict(user_input)
    result = "Bank Account" if prediction[0] == 1 else "No Bank Account"
    st.write(f"Predicted Bank Account: {result}")