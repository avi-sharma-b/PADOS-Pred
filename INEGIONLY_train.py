import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from sklearn.ensemble import HistGradientBoostingClassifier



CSV_PATH = "readyToTrainV1.csv"
LABEL_COL = "pado"
RANDOM_STATE = 42

# If you want to optimize something other than F1 on the VAL set:
# - "f1" (default)
# - "precision_at_recall" (set MIN_RECALL)
# - "recall_at_precision" (set MIN_PRECISION)
THRESHOLD_MODE = "f1"
MIN_RECALL = 0.90
MIN_PRECISION = 0.5

RATE_COLS = [
    "insured_rate",
    "disability_rate",
    "electricity_access_rate",
    "piped_water_access_rate",
    "drainage_access_rate",
    "floor_material_rate",
    "occupants_per_room",
    "average_schooling",
]



df = pd.read_csv(CSV_PATH)

y = df[LABEL_COL].astype(int)
X = df.drop(columns=[LABEL_COL]).copy()

for c in RATE_COLS:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")


X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval
)

print("Total rows:", len(df))
print("Train rows:", len(X_train), "Val rows:", len(X_val), "Test rows:", len(X_test))
print("Overall label counts:\n", y.value_counts())
print("Train label counts:\n", y_train.value_counts())
print("Val label counts:\n", y_val.value_counts())
print("Test label counts:\n", y_test.value_counts())


cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ],
    remainder="drop"
)

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))
])



param_grid = {
    "clf__learning_rate": [0.03, 0.06, 0.1],
    "clf__max_depth": [3, 5, None],
    "clf__max_leaf_nodes": [15, 31, 63],
    "clf__min_samples_leaf": [10, 20, 50],
    "clf__l2_regularization": [0.0, 0.1, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="average_precision",   # PR-AUC
    cv=cv,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print("\nBest params:", search.best_params_)


val_proba = best_model.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_proba)

def pick_threshold(mode: str):
    if len(thr) == 0:
        return 0.5

    if mode == "f1":
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        idx = np.nanargmax(f1)
        
        if idx == 0:
            return thr[0]
        if idx - 1 >= len(thr):
            return thr[-1]
        return thr[idx - 1]

    if mode == "precision_at_recall":
        
        ok = np.where(rec[:-1] >= MIN_RECALL)[0]  
        if len(ok) == 0:
            return thr[np.argmax(rec[:-1])]
        best = ok[np.argmax(prec[ok])]
        return thr[best]

    if mode == "recall_at_precision":
        ok = np.where(prec[:-1] >= MIN_PRECISION)[0]
        if len(ok) == 0:
            return thr[np.argmax(prec[:-1])]
        best = ok[np.argmax(rec[ok])]
        return thr[best]

    raise ValueError("Unknown THRESHOLD_MODE")

threshold = pick_threshold(THRESHOLD_MODE)
print(f"Chosen threshold (on VAL, mode={THRESHOLD_MODE}): {threshold:.6f}")


test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= threshold).astype(int)

print("\n=== TEST METRICS ===")
print("ROC-AUC:", roc_auc_score(y_test, test_proba))
print("PR-AUC :", average_precision_score(y_test, test_proba))
print(classification_report(y_test, test_pred, digits=3))
