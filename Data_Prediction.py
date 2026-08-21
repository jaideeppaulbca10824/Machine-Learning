import pandas as pd                             #For load the DataSet
#For Preprocessing
from sklearn.compose import ColumnTransformer   
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
#For train/test split
from sklearn.model_selection import train_test_split
#For LogisticRegression Model
from sklearn.linear_model import LogisticRegression
#For Evaluation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# 1. Load Dataset
file=pd.read_csv("StudentPerformanceFactors.csv")
print("Dataset Shape: ",file.shape)
print(file.head())
print(file.info(),end="\n")

# 2. Create PASS/FAIL target            #All the mistakes from this part
pass_mark=40
print("\nExam Score Statistics:")
file["Result"]=(file["Exam_Score"]>=pass_mark).astype(int)
print("Pass/Fail distribution:\n")
print(file["Result"].value_counts())
print("\n")

# 3. Remove Exam_Score and Result from features
x=file.drop(columns=["Exam_Score","Result"])
y=file["Result"]

# 4. Identify numerical and categorical columns
numerical_features=x.select_dtypes(include=["int64","float64"]).columns
categorical_features=x.select_dtypes(include=["object"]).columns

# 5. Preprocessing
numerical_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])
categorical_transformer=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])
preprocessor=ColumnTransformer(
    transformers=[
    ("num",numerical_transformer,numerical_features),
    ("cat",categorical_transformer,categorical_features)
])

# 6. Split into training and unseen testing data
x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print(f"Training Samples: {len(x_train)}")
print(f"Testing Samples: {len(x_test)}")
print(end="\n")

# 7. Logistic Regression Model
logistic_model=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("classifier",LogisticRegression(max_iter=1000))
])

# 8. Train Model
logistic_model.fit(x_train,y_train)

# 9. Predict the unseen test data
y_pred=logistic_model.predict(x_test)

# 10. Evaluate model
accuracy=accuracy_score(y_test,y_pred)
print("\nLogistic Regression Accuracy: ",round(accuracy*100,2),"%")
print("\nConfusion Matrix: ",confusion_matrix(y_test,y_pred))
print("\nClassification Matrix: ",classification_report(
    y_test,
    y_pred,
    target_names=["Fail","Pass"]
))

