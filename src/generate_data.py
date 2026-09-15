import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ---------------------------------------------------
# CREATE FOLDERS
# ---------------------------------------------------

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ---------------------------------------------------
# BASIC DATA
# ---------------------------------------------------

continents = [
    "Asia",
    "Europe",
    "North America",
    "South America",
    "Africa",
    "Australia"
]

countries = {
    "Asia": ["India", "Japan", "Thailand", "Singapore"],
    "Europe": ["France", "Italy", "Spain", "Germany"],
    "North America": ["USA", "Canada", "Mexico"],
    "South America": ["Brazil", "Argentina", "Chile"],
    "Africa": ["Egypt", "Kenya", "South Africa"],
    "Australia": ["Australia", "New Zealand"]
}

cities = {
    "India": ["Delhi", "Mumbai", "Jaipur", "Goa"],
    "Japan": ["Tokyo", "Kyoto", "Osaka"],
    "Thailand": ["Bangkok", "Phuket", "Pattaya"],
    "Singapore": ["Singapore"],
    "France": ["Paris", "Nice"],
    "Italy": ["Rome", "Venice", "Milan"],
    "Spain": ["Madrid", "Barcelona"],
    "Germany": ["Berlin", "Munich"],
    "USA": ["New York", "Los Angeles", "Las Vegas"],
    "Canada": ["Toronto", "Vancouver"],
    "Mexico": ["Cancun", "Mexico City"],
    "Brazil": ["Rio de Janeiro", "Sao Paulo"],
    "Argentina": ["Buenos Aires"],
    "Chile": ["Santiago"],
    "Egypt": ["Cairo", "Luxor"],
    "Kenya": ["Nairobi", "Mombasa"],
    "South Africa": ["Cape Town", "Johannesburg"],
    "Australia": ["Sydney", "Melbourne"],
    "New Zealand": ["Auckland", "Queenstown"]
}

attraction_types = [
    "Beach",
    "Museum",
    "Park",
    "Historical Site",
    "Temple",
    "Adventure",
    "Wildlife",
    "Shopping"
]

visit_modes = [
    "Business",
    "Couples",
    "Family",
    "Friends",
    "Solo"
]

# ---------------------------------------------------
# GENERATE USERS
# ---------------------------------------------------

num_users = 1000

users = []

for user_id in range(1, num_users + 1):

    continent = np.random.choice(continents)

    country = np.random.choice(countries[continent])

    city = np.random.choice(cities[country])

    users.append([
        user_id,
        continent,
        country,
        city
    ])

users_df = pd.DataFrame(
    users,
    columns=[
        "UserId",
        "Continent",
        "Country",
        "City"
    ]
)

# ---------------------------------------------------
# GENERATE ATTRACTIONS
# ---------------------------------------------------

attractions = []

num_attractions = 100

for attraction_id in range(1, num_attractions + 1):

    continent = np.random.choice(continents)

    country = np.random.choice(countries[continent])

    city = np.random.choice(cities[country])

    attraction_type = np.random.choice(attraction_types)

    attraction_name = f"{city} {attraction_type} {attraction_id}"

    attractions.append([
        attraction_id,
        city,
        country,
        attraction_type,
        attraction_name,
        f"Main Road, {city}"
    ])

attractions_df = pd.DataFrame(
    attractions,
    columns=[
        "AttractionId",
        "City",
        "Country",
        "AttractionType",
        "Attraction",
        "AttractionAddress"
    ]
)

# ---------------------------------------------------
# GENERATE TRANSACTIONS
# ---------------------------------------------------

num_transactions = 5000

transactions = []

for transaction_id in range(1, num_transactions + 1):

    user_id = np.random.randint(1, num_users + 1)

    attraction_id = np.random.randint(1, num_attractions + 1)

    visit_year = np.random.choice(
        [2022, 2023, 2024, 2025]
    )

    visit_month = np.random.randint(1, 13)

    visit_mode = np.random.choice(
        visit_modes,
        p=[0.15, 0.25, 0.25, 0.20, 0.15]
    )

    rating = np.clip(
        np.random.normal(3.8, 0.7),
        1,
        5
    )

    transactions.append([
        transaction_id,
        user_id,
        visit_year,
        visit_month,
        visit_mode,
        attraction_id,
        round(rating, 1)
    ])

transactions_df = pd.DataFrame(
    transactions,
    columns=[
        "TransactionId",
        "UserId",
        "VisitYear",
        "VisitMonth",
        "VisitMode",
        "AttractionId",
        "Rating"
    ]
)

# ---------------------------------------------------
# SAVE DATA
# ---------------------------------------------------

users_df.to_csv(
    "data/raw/users.csv",
    index=False
)

attractions_df.to_csv(
    "data/raw/attractions.csv",
    index=False
)

transactions_df.to_csv(
    "data/raw/transactions.csv",
    index=False
)

print("===================================")
print("Tourism Dataset Created Successfully")
print("===================================")

print("\nUsers:", len(users_df))
print("Attractions:", len(attractions_df))
print("Transactions:", len(transactions_df))

print("\nFiles created:")

print("data/raw/users.csv")
print("data/raw/attractions.csv")
print("data/raw/transactions.csv")