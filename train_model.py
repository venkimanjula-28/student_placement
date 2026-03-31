import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("../data/Sample.csv")
data=data.drop(["Email","Name"],axis=1)
yn_col=["Internships(Y/N)",
        "Training(Y/N)",
        "Backlog in 5th sem",
        "Innovative Project(Y/N)",
        "Technical Course(Y/N)",
        
]

for cols in yn_col:
    data[cols]=data[cols].map({"Yes":1, "No":0})
data["Placement(Y/N)?"]=data["Placement(Y/N)?"].map({"Yes":1,"No":0,"Placed":1,
"Not Placed":0
})
le=LabelEncoder()
data["Gender"]=le.fit_transform(data["Gender"])
data["10th board"]=le.fit_transform(data["10th board"])
data["12th board"]=le.fit_transform(data["12th board"])
data["Stream"]=le.fit_transform(data["Stream"])

#Features and Target
x=data.drop("Placement(Y/N)?",axis=1)
y=data["Placement(Y/N)?"]

# train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#model training
model=RandomForestClassifier()

model.fit(x_train,y_train)

# prediction
pred=model.predict(x_test)

# accuracy
accuracy=accuracy_score(y_test,pred)
print("Model accuracy")

# save model in pickel
joblib.dump(model,"../models/models.pkl")
print("model saved successfully!")