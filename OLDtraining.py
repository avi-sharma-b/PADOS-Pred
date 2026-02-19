import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv("Clean_Cs_GSVE_Mh_zip_plus_admin.csv")


y = df["PADO"].astype(int)
X = df.drop(columns=["PADO"]).copy()

for c in ["zip_code", "zipcode_gsv", "cod_postal"]:
    if c in X.columns:
        X[c] = X[c].astype(str)

"""
split 80/20
"""


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Total rows:", len(df))
print("Train rows:", len(X_train), "Test rows:", len(X_test))
print("Overall label counts:\n", y.value_counts())
print("Test label counts:\n", y_test.value_counts())


cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]


preprocess_sparse = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),  # sparse output
        ]), cat_cols),
    ]
)



ohe_dense = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocess_dense = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe_dense),
        ]), cat_cols),
    ]
)

hgb = Pipeline([
    ("prep", preprocess_dense),
    ("clf", HistGradientBoostingClassifier(random_state=42))
])

def evaluate(model, name):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.2).astype(int)
    print(f"\n=== {name} ===")
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print("PR-AUC :", average_precision_score(y_test, proba))
    print(classification_report(y_test, pred, digits=3))

evaluate(hgb, "HistGradientBoosting")
