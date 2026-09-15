import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# LOAD
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
    "UserAverageRating",
    "AttractionAverageRating"
]

X = df[features]

y = df["Rating"]

# ---------------------------------------------------
# SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

model = RandomForestRegressor(
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

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nRegression Results")

print("MAE:", round(mae, 3))

print("MSE:", round(mse, 3))

print("R2 Score:", round(r2, 3))

# ---------------------------------------------------
# SAVE
# ---------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/rating_model.pkl"
)

joblib.dump(
    features,
    "models/regression_features.pkl"
)

print("\nRegression model saved!")