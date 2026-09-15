import pandas as pd
import os

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

users = pd.read_csv("data/raw/users.csv")
attractions = pd.read_csv("data/raw/attractions.csv")
transactions = pd.read_csv("data/raw/transactions.csv")

print("Users:", users.shape)
print("Attractions:", attractions.shape)
print("Transactions:", transactions.shape)

# ---------------------------------------------------
# MERGE DATA
# ---------------------------------------------------

df = transactions.merge(
    users,
    on="UserId",
    how="left"
)

df = df.merge(
    attractions,
    on="AttractionId",
    how="left"
)

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

df.drop_duplicates(inplace=True)

df["Rating"] = df["Rating"].clip(1, 5)

df["VisitYear"] = df["VisitYear"].astype(int)

df["VisitMonth"] = df["VisitMonth"].astype(int)

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

df["VisitSeason"] = df["VisitMonth"].apply(
    lambda x:
        "Winter" if x in [12, 1, 2]
        else "Spring" if x in [3, 4, 5]
        else "Summer" if x in [6, 7, 8]
        else "Autumn"
)

# User average rating
user_avg_rating = (
    df.groupby("UserId")["Rating"]
    .mean()
    .reset_index()
)

user_avg_rating.rename(
    columns={"Rating": "UserAverageRating"},
    inplace=True
)

df = df.merge(
    user_avg_rating,
    on="UserId",
    how="left"
)

# Attraction average rating
attraction_avg_rating = (
    df.groupby("AttractionId")["Rating"]
    .mean()
    .reset_index()
)

attraction_avg_rating.rename(
    columns={"Rating": "AttractionAverageRating"},
    inplace=True
)

df = df.merge(
    attraction_avg_rating,
    on="AttractionId",
    how="left"
)

# ---------------------------------------------------
# SAVE
# ---------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/tourism_final.csv",
    index=False
)

print("\nPreprocessing completed!")

print(
    "Final dataset:",
    df.shape
)

print(
    "\nSaved as:",
    "data/processed/tourism_final.csv"
)

print("\nColumns:")
print(df.columns.tolist())