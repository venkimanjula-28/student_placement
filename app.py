import streamlit as st 
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model=joblib.load(r"c:\student_placement\models\models.pkl")  # Updated to absolute path; adjust if the file is in a different location

# Load data for consistent encoding
data = pd.read_csv(r"c:\student_placement\data\Sample.csv")

# Fit encoders on training data
le_gender = LabelEncoder().fit(data["Gender"])
le_10th_board = LabelEncoder().fit(data["10th board"])
le_12th_board = LabelEncoder().fit(data["12th board"])
le_stream = LabelEncoder().fit(data["Stream"])

st.set_page_config(page_title=" Student placement prediction system",
                    page_icon="🎓",
                    layout="wide")
#title
st.markdown("<h1 style='text-align: center; background-color:lightpink;color:blue;'>🎓Student Placement Prediction System 🎓</h1>", 
            unsafe_allow_html=True)
st.markdown("======")
st.sidebar.header("Student Details")
gender=st.selectbox("Gender",["Male","Female"])
tenth_board=st.selectbox("10th board", sorted(data["10th board"].unique()))
tenth_marks=st.number_input("10th marks")
twelth_board=st.selectbox("12th board", sorted(data["12th board"].unique()))
twelth_marks=st.number_input("12th marks")
stream=st.selectbox("Stream", sorted(data["Stream"].unique()))
cgpa=st.number_input("Cgpa")
internship=st.selectbox("Internships(Y/N)",["Yes","No"])

training=st.selectbox("Training(Y/N)",["Yes","No"])
backlog=st.selectbox("Backlog in 5th sem ",["Yes","No"])
project=st.selectbox("Innovative Project(Y/N)",["Yes","No"])
Community_course=st.slider("Communitication level",1,15)
course=st.selectbox(" Technical Course(Y/N)",["Yes","No"])
st.markdown("🔤Student input summary✅")
col1,col2,col3=st.columns(3)
col1.metric("10th marks", tenth_marks)
col2.metric("12th marks", twelth_marks)
col3.metric("Cgpa", cgpa)
st.markdown("======================================================")
if st.button("Predict Placement"):
    gender_enc = le_gender.transform([gender])[0]
    tenth_board_enc = le_10th_board.transform([tenth_board])[0]
    twelth_board_enc = le_12th_board.transform([twelth_board])[0]
    stream_enc = le_stream.transform([stream])[0]
    internship_enc=1 if internship=="Yes" else 0
    training_enc=1 if training=="Yes" else 0
    backlog_enc=1 if backlog=="Yes" else 0
    project_enc=1 if project=="Yes" else 0
    course_enc=1 if course== "Yes" else 0
   
    input_data=np.array([[gender_enc, tenth_board_enc, tenth_marks, twelth_board_enc, twelth_marks, stream_enc, cgpa, internship_enc, training_enc, backlog_enc, project_enc, Community_course, course_enc]])
    prediction=model.predict(input_data)
    if prediction[0]==1:
        st.success("🎉🎉==student will be placed==🎊")
        st.ballon()
        
    else:
        st.error("❌==🥺 student not likely to be placed🥹==")
        st.snow()