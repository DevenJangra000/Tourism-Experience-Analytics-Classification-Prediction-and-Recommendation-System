import pandas as pd


def recommend_attractions(
    attraction_type=None,
    city=None,
    top_n=5
):

    df = pd.read_csv(
        "data/raw/attractions.csv"
    )

    result = df.copy()

    # Filter by attraction type
    if attraction_type:
        result = result[
            result["AttractionType"] == attraction_type
        ]

    # Filter by city
    if city:
        city_result = result[
            result["City"] == city
        ]

        if len(city_result) > 0:
            result = city_result

    # Select recommendations
    recommendations = result.head(top_n)

    return recommendations[
        [
            "Attraction",
            "City",
            "Country",
            "AttractionType",
            "AttractionAddress"
        ]
    ]


if __name__ == "__main__":

    print(
        recommend_attractions(
            attraction_type="Beach",
            top_n=5
        )
    )