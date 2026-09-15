import streamlit as st
import pandas as pd
import joblib
import sys
import os

# ---------------------------------------------------
# PATH
# ---------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.recommendation import recommend_attractions

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title(
    "🌍 Tourism Experience Analytics"
)

st.write(
    "Classification, Rating Prediction & "
    "Personalized Attraction Recommendation System"
)

st.divider()

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

classification_model = joblib.load(
    "models/visit_mode_model.pkl"
)

classification_encoder = joblib.load(
    "models/visit_mode_encoder.pkl"
)

regression_model = joblib.load(
    "models/rating_model.pkl"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

data = pd.read_csv(
    "data/processed/tourism_final.csv"
)

attractions = pd.read_csv(
    "data/raw/attractions.csv"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header(
    "Tourist Information"
)

visit_year = st.sidebar.selectbox(
    "Visit Year",
    [2022, 2023, 2024, 2025]
)

visit_month = st.sidebar.slider(
    "Visit Month",
    1,
    12,
    6
)

rating = st.sidebar.slider(
    "Expected Rating",
    1.0,
    5.0,
    4.0,
    0.1
)

user_average_rating = st.sidebar.slider(
    "Your Average Rating",
    1.0,
    5.0,
    4.0,
    0.1
)

attraction_average_rating = st.sidebar.slider(
    "Attraction Average Rating",
    1.0,
    5.0,
    4.0,
    0.1
)

# ---------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------

st.header(
    "🤖 Visit Mode Prediction"
)

input_classification = pd.DataFrame({
    "VisitYear": [visit_year],
    "VisitMonth": [visit_month],
    "Rating": [rating],
    "UserAverageRating": [user_average_rating],
    "AttractionAverageRating": [
        attraction_average_rating
    ]
})

if st.button(
    "Predict Visit Mode"
):

    prediction = classification_model.predict(
        input_classification
    )

    visit_mode = classification_encoder.inverse_transform(
        prediction
    )[0]

    st.success(
        f"Predicted Visit Mode: **{visit_mode}**"
    )

# ---------------------------------------------------
# REGRESSION
# ---------------------------------------------------

st.header(
    "⭐ Attraction Rating Prediction"
)

input_regression = pd.DataFrame({
    "VisitYear": [visit_year],
    "VisitMonth": [visit_month],
    "UserAverageRating": [
        user_average_rating
    ],
    "AttractionAverageRating": [
        attraction_average_rating
    ]
})

if st.button(
    "Predict Attraction Rating"
):

    predicted_rating = regression_model.predict(
        input_regression
    )[0]

    predicted_rating = max(
        1,
        min(5, predicted_rating)
    )

    st.success(
        f"Predicted Rating: ⭐ "
        f"{predicted_rating:.2f} / 5"
    )

# ---------------------------------------------------
# RECOMMENDATION
# ---------------------------------------------------

st.header(
    "🎯 Personalized Attraction Recommendations"
)

col1, col2 = st.columns(2)

with col1:

    attraction_type = st.selectbox(
        "Preferred Attraction Type",
        sorted(
            attractions["AttractionType"]
            .unique()
        )
    )

with col2:

    city = st.selectbox(
        "Preferred City",
        sorted(
            attractions["City"]
            .unique()
        )
    )

if st.button(
    "Get Recommendations"
):

    recommendations = recommend_attractions(
        attraction_type=attraction_type,
        city=city,
        top_n=5
    )

    if len(recommendations) == 0:

        st.warning(
            "No exact match found. "
            "Showing attractions from the selected type."
        )

        recommendations = recommend_attractions(
            attraction_type=attraction_type,
            top_n=5
        )

    st.dataframe(
        recommendations,
        use_container_width=True
    )

# ---------------------------------------------------
# ANALYTICS
# ---------------------------------------------------

st.divider()

st.header(
    "📊 Tourism Analytics"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Users",
        data["UserId"].nunique()
    )

with col2:

    st.metric(
        "Total Attractions",
        attractions["AttractionId"].nunique()
    )

with col3:

    st.metric(
        "Total Visits",
        len(data)
    )

# ---------------------------------------------------
# POPULAR VISIT MODES
# ---------------------------------------------------

st.subheader(
    "Visit Mode Distribution"
)

visit_mode_count = (
    data["VisitMode"]
    .value_counts()
)

st.bar_chart(
    visit_mode_count
)

# ---------------------------------------------------
# POPULAR ATTRACTION TYPES
# ---------------------------------------------------

st.subheader(
    "Popular Attraction Types"
)

attraction_count = (
    data["AttractionType"]
    .value_counts()
)

st.bar_chart(
    attraction_count
)

# ---------------------------------------------------
# RATING ANALYSIS
# ---------------------------------------------------

st.subheader(
    "Average Rating by Attraction Type"
)

rating_analysis = (
    data.groupby(
        "AttractionType"
    )["Rating"]
    .mean()
    .sort_values(
        ascending=False
    )
)

st.bar_chart(
    rating_analysis
)

st.divider()

st.caption(
    "Tourism Experience Analytics | "
    "Machine Learning + Recommendation System"
)