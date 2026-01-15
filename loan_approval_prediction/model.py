import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("loan_data.csv")
df.drop(columns=["Loan_ID"], inplace=True)
df["ApplicantIncome"] = pd.to_numeric(df["ApplicantIncome"], errors="coerce")
df["CoapplicantIncome"] = pd.to_numeric(
    df["CoapplicantIncome"], errors="coerce")
df["LoanAmount"] = pd.to_numeric(df["LoanAmount"], errors="coerce")
df["LoanAmount"] = pd.to_numeric(df["LoanAmount"], errors="coerce")
df["Loan_Amount_Term"] = pd.to_numeric(df["Loan_Amount_Term"], errors="coerce")
df["Credit_History"] = pd.to_numeric(df["Credit_History"], errors="coerce")
df["Dependents"] = pd.to_numeric(df["Dependents"].astype(
    str).replace("3+", "3"), errors="coerce")
df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (df["Total_Income"] + 1)
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].median())
df["ApplicantIncome"] = df["ApplicantIncome"].fillna(
    df["ApplicantIncome"].median())
df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(
    df["CoapplicantIncome"].median())
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].median())
df["Credit_History"] = df["Credit_History"].fillna(
    df["Credit_History"].median())
df["Total_Income"] = df["Total_Income"].fillna(df["Total_Income"].median())
df["Loan_to_Income_Ratio"] = df["Loan_to_Income_Ratio"].fillna(
    df["Loan_to_Income_Ratio"].median())

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

df = df.dropna()

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Dependents",
    "Total_Income",
    "Loan_to_Income_Ratio"
]

categorical_features = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area"
]

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

numeric_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))


def predict_loan(applicant_data: dict, model):
    df_applicant = pd.DataFrame([applicant_data])
    df_applicant["Dependents"] = pd.to_numeric(
        df_applicant["Dependents"].astype(str).replace("3+", "3"),
        errors="coerce"
    )
    df_applicant["Dependents"] = df_applicant["Dependents"].fillna(0)
    df_applicant["Total_Income"] = (
        df_applicant["ApplicantIncome"] +
        df_applicant["CoapplicantIncome"]
    )
    df_applicant["Loan_to_Income_Ratio"] = (
        df_applicant["LoanAmount"] /
        (df_applicant["Total_Income"] + 1)
    )
    probability = model.predict_proba(df_applicant)[0][1]
    prediction = model.predict(df_applicant)[0]
    return prediction, probability


test_applicant = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 3000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 100,
    "Loan_Amount_Term": 10,
    "Credit_History": 1,
    "Property_Area": "Urban"
}

prediction, probability = predict_loan(test_applicant, model)
print("Loan Status:", "Approved" if prediction else "Rejected")
print("Approval Probability:", round(probability * 100, 4), "%")
