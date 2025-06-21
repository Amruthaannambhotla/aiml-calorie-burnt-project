import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
rfr = pickle.load(open('rfr.pkl', 'rb'))
x_train = pd.read_csv('X_train.csv')

def pred(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp):
    features = np.array([[Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp]])
    prediction = rfr.predict(features).reshape(1, -1)
    return prediction[0]

# Web app
# Title
st.title("Predicting calories burnt")
st.image("calorie-burned-walking.png", caption="Calories Burn Prediction", width=500)

# Descriptive Text
st.write("""
## Enter the details below to predict the calories burned.
""")

# Input Fields
col1, col2 = st.columns([1, 5])
with col1:
    st.image("gender.png", width=100)
with col2:
    Gender = st.selectbox('Gender', x_train['Gender'].unique())
col1, col2 = st.columns([5, 1])
with col2:
    st.image("age.png", width=100)
with col1:
    Age = st.slider('Age', min_value=int(x_train['Age'].min()), max_value=int(x_train['Age'].max()), value=int(x_train['Age'].mean()))
col1, col2 = st.columns([1, 5])
with col1:
    st.image("height.jpeg", width=100)
with col2:
    Height = st.slider('Height (cm)', min_value=int(x_train['Height'].min()), max_value=int(x_train['Height'].max()), value=int(x_train['Height'].mean()))
col1, col2 = st.columns([5, 1])
with col2:
    st.image("weight.jpg", width=100)
with col1:
    Weight = st.slider('Weight (kg)', min_value=int(x_train['Weight'].min()), max_value=int(x_train['Weight'].max()), value=int(x_train['Weight'].mean()))
col1, col2 = st.columns([1, 5])
with col1:
    st.image("duration.png", width=100)
with col2:
    Duration = st.slider('Duration (minutes)', min_value=int(x_train['Duration'].min()), max_value=int(x_train['Duration'].max()), value=int(x_train['Duration'].mean()))
col1, col2 = st.columns([5, 1])
with col2:
    st.image("heartrate.jpg", width=100)
with col1:
    Heart_rate = st.slider('Heart Rate (bpm)', min_value=int(x_train['Heart_Rate'].min()), max_value=int(x_train['Heart_Rate'].max()), value=int(x_train['Heart_Rate'].mean()))
col1, col2 = st.columns([1, 5])
with col1:
    st.image("body temp.jpeg", width=100)
with col2:
    Body_temp = st.slider('Body Temperature (°C)', min_value=float(x_train['Body_Temp'].min()), max_value=float(x_train['Body_Temp'].max()), value=float(x_train['Body_Temp'].mean()))

# Prediction
if st.button('Predict'):
    result = pred(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp)
    st.success(f"You have consumed approximately {result[0]:.2f} calories.")
st.title("Interactive Machine Learning App")

# Add a dropdown (selectbox)
options = ["Gender Distribution", "Age distribution", "Height distribution by gender", "Weight distribution by gender",
           "Calories burned vs Duration", "Heart rate vs body temperature", "Calories burned by gender"]

# Additional Visualizations
st.write("""
## Data Distribution
""")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(options)

with tab1:
    st.write("### Gender Distribution")
    fig, ax = plt.subplots(figsize=(5,3))
    x_train['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightpink'], ax=ax,)
    ax.set_ylabel('')
    st.pyplot(fig)

with tab2:
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(x_train['Age'], kde=True, ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

with tab3:
    st.write("### Height Distribution by Gender")
    fig, ax = plt.subplots()
    sns.boxplot(x='Gender', y='Height', data=x_train, ax=ax)
    st.pyplot(fig)

with tab4:
    st.write("### Weight Distribution by Gender")
    fig, ax = plt.subplots()
    sns.boxplot(x='Gender', y='Weight', data=x_train, ax=ax)
    st.pyplot(fig)

with tab5:
    st.write("### Calories Burned vs. Duration")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Duration', y='Calories', hue='Gender', data=x_train, ax=ax)
    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Calories Burned')
    st.pyplot(fig)

with tab6:
    st.write("### Heart Rate vs. Body Temperature")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Body_Temp', y='Heart_Rate', hue='Gender', data=x_train, ax=ax)
    ax.set_xlabel('Body Temperature (°C)')
    ax.set_ylabel('Heart Rate (bpm)')
    st.pyplot(fig)

with tab7:
    st.write("### Calories Burned by Gender")
    fig, ax = plt.subplots()
    sns.boxplot(x='Gender', y='Calories', data=x_train, ax=ax)
    st.pyplot(fig)
