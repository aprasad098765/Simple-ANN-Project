import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open ('ohe.pkl','rb') as file:
    ohe = pickle.load(file)

with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open ('scalar.pkl','rb') as file:
    scalar = pickle.load(file)

st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
estimated_salary = st.number_input('Estimated Salary')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is active member',[0,1])
balance = st.number_input('Balance')

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=ohe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis = 1)
input_data_scaled = scalar.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(prediction_prob)

if prediction_prob > 0.5:
    st.write('Customer likely to churn')
else:
    st.write('Customer Not likely to churn')