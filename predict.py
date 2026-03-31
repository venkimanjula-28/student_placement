import joblib
import numpy as np

#load trained model
model=joblib.load("../models/models.pkl")

# example Student data
student= np.array([[1,85,82,4,5,6,7,0,1,0,1,3,0]])
prediction=model.predict(student)
if prediction[0]==1:
 print("student will be placed ")
else:
 print("student will not placed")