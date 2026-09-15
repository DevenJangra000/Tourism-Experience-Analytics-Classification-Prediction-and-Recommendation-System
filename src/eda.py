import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(
    "data/processed/tourism_final.csv"
)

os.makedirs("data/processed/charts", exist_ok=True)

# ---------------------------------------------------
# VISIT MODE
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="VisitMode"
)

plt.title("Tourists by Visit Mode")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "data/processed/charts/visit_mode.png"
)

plt.close()

# ---------------------------------------------------
# RATINGS
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Rating"],
    bins=10,
    kde=True
)

plt.title("Rating Distribution")
plt.tight_layout()

plt.savefig(
    "data/processed/charts/rating_distribution.png"
)

plt.close()

# ---------------------------------------------------
# ATTRACTION TYPE
# ---------------------------------------------------

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="AttractionType"
)

plt.title("Attraction Type Popularity")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "data/processed/charts/attraction_types.png"
)

plt.close()

# ---------------------------------------------------
# CONTINENT
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Continent"
)

plt.title("Visitors by Continent")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "data/processed/charts/continent.png"
)

plt.close()

# ---------------------------------------------------
# AVERAGE RATING BY ATTRACTION TYPE
# ---------------------------------------------------

rating_data = (
    df.groupby("AttractionType")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))

rating_data.plot(
    kind="bar"
)

plt.title(
    "Average Rating by Attraction Type"
)

plt.ylabel("Average Rating")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "data/processed/charts/average_rating.png"
)

plt.close()

print("EDA completed successfully!")

print("\nAverage Rating:")
print(rating_data)