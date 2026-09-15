import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv(
    "data/processed/tourism_final.csv"
)

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------

features = [
    "VisitYear",
    "VisitMonth",
    "Rating",
    "UserAverageRating",
    "AttractionAverageRating"
]

X = df[features]

y = df["VisitMode"]

# ---------------------------------------------------
# ENCODE TARGET
# ---------------------------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nClassification Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/visit_mode_model.pkl"
)

joblib.dump(
    encoder,
    "models/visit_mode_encoder.pkl"
)

joblib.dump(
    features,
    "models/classification_features.pkl"
)

print("\nClassification model saved!")