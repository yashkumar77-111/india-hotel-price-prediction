"""
Streamlit app for the India Hotel Price Predictor.

Loads the trained pipeline from models/model.pkl. If it's missing OR fails
to load (e.g. a scikit-learn version mismatch between the machine that
saved it and the machine running this app), the model is trained fresh
right here instead - this guarantees the model always matches whichever
scikit-learn version is actually installed in this environment, which a
pre-saved file can't guarantee across different machines/cloud platforms.
"""
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from landmarks import LANDMARK_COORDS, STATE_LANDMARKS
from src.predict import load_model, predict_price
from src.train import AMENITY_FEATURES, DATA_PATH, MODEL_PATH, clean_data, load_data, train_and_save

st.set_page_config(page_title="India Hotel Price Predictor", page_icon="🏨")


@st.cache_resource
def get_model():
    try:
        return load_model(MODEL_PATH)
    except Exception:
        # Missing file, or a version-mismatch error unpickling an existing
        # one - either way, train a fresh model in this exact environment
        # rather than surfacing a confusing error to the user.
        with st.spinner("Setting up the model for the first time - this takes a minute..."):
            train_and_save(data_path=DATA_PATH, model_path=MODEL_PATH)
        return load_model(MODEL_PATH)


@st.cache_data
def get_reference_data():
    """Training data, used only to populate dropdown options and numeric defaults."""
    return clean_data(load_data(DATA_PATH))


st.title("🏨 India Hotel Price Predictor")
st.write("Enter listing details to estimate the nightly price (in ₹).")
st.caption(
    "Trained on ~1,700 real India hotel listings combined from multiple "
    "sources, so treat predictions as a rough estimate rather than a "
    "precise one."
)

model = get_model()

df = get_reference_data()

st.subheader("Location")
states = sorted(df["state"].dropna().unique())
state = st.selectbox("State", states)

# --- Landmark + distance section ---------------------------------------
# Only 6 states have real landmark data (from the MakeMyTrip source, which
# only covered those 6 cities). For any other state this section is
# skipped entirely rather than showing an empty or fake dropdown.
#
# The landmark and distance chosen here feed directly into the prediction
# as distance_to_landmark_km - this is a real model input, not a separate
# search/browse feature, so there is exactly one predicted price, not a
# predicted price sitting next to a list of real ones.
DISTANCE_TIERS = {
    "Within 1 km": 0.5,
    "Within 2 km": 1.5,
    "Within 5 km": 3.5,
    "Within 10 km": 7.5,
    "Any distance": None,
}

landmark = None
distance_to_landmark_km = None

if state in STATE_LANDMARKS:
    landmark_options = ["None"] + STATE_LANDMARKS[state]
    landmark = st.selectbox(
        "Nearby landmark (optional)",
        landmark_options,
        help="Only available for the 6 cities with real landmark-distance data.",
    )

    if landmark != "None":
        distance_choice = st.selectbox(
            "Distance from landmark",
            list(DISTANCE_TIERS.keys()),
            help="Used as an input to the price estimate.",
        )
        distance_to_landmark_km = DISTANCE_TIERS[distance_choice]

        landmark_lat, landmark_lon = LANDMARK_COORDS[landmark]
        st.caption(
            f"Map reference for {landmark.strip()}. Distance is a "
            "straight-line (\"as the crow flies\") estimate, not driving distance."
        )

        # Map: landmark pin + a circle sized to the selected distance tier,
        # purely a visual reference for what was picked above.
        m = folium.Map(location=[landmark_lat, landmark_lon], zoom_start=13)
        folium.Marker(
            [landmark_lat, landmark_lon],
            tooltip=landmark.strip(),
            icon=folium.Icon(color="red", icon="star"),
        ).add_to(m)
        circle_radius_m = (distance_to_landmark_km or 8) * 1000
        folium.Circle(
            [landmark_lat, landmark_lon],
            radius=circle_radius_m,
            color="red",
            fill=True,
            fill_opacity=0.08,
        ).add_to(m)
        st_folium(m, height=350, width=700, key="landmark_map")

# Property type isn'''t offered as a choice: 99.6% of the training data is
# hotels (1,664 of 1,671 listings), so the model has almost no real signal
# for Apartment/Villa/House/Bungalow. Rather than show a dropdown that
# implies those are equally reliable options, this app is scoped to hotels
# only.
property_type = "Hotel"

st.subheader("Ratings")
col3, col4 = st.columns(2)

with col3:
    star_rating = st.selectbox(
        "Star rating",
        options=[1, 2, 3, 4, 5],
        index=2,
        help="Official hotel star classification.",
    )

with col4:
    review_rating = st.slider(
        "Guest review rating",
        min_value=1.9,
        max_value=5.0,
        value=4.1,
        step=0.1,
        help="Average guest satisfaction score, separate from star rating. "
             "Based on real data: half of listings fall between 3.9 and 4.4.",
    )

REVIEW_COUNT_TIERS = {
    "Unknown / skip": None,
    "50+": 150,
    "250+": 600,
    "500+": 1200,
    "1,000+": 2000,
    "2,500+": 4000,
    "5,000+": 7000,
}
review_count_choice = st.selectbox(
    "Minimum number of reviews",
    list(REVIEW_COUNT_TIERS.keys()),
    help="A rough popularity tier, not an exact count.",
)
review_count = REVIEW_COUNT_TIERS[review_count_choice]

st.subheader("Amenities")
st.caption("Select everything this listing offers.")

AMENITY_LABELS = {
    "has_wifi": "Free Wi-Fi",
    "has_parking": "Parking",
    "has_pool": "Pool",
    "has_ac": "Air conditioning",
    "has_breakfast": "Breakfast included",
    "has_fitness_center": "Fitness center",
    "has_spa": "Spa",
    "has_restaurant": "Restaurant",
    "has_bar": "Bar",
    "pet_friendly": "Pet-friendly",
    "kid_friendly": "Kid-friendly",
    "wheelchair_accessible": "Wheelchair accessible",
    "has_hot_tub": "Hot tub",
    "has_airport_shuttle": "Airport shuttle",
    "has_room_service": "Room service",
}

amenity_values = {}
amenity_cols = st.columns(3)
for i, feature in enumerate(AMENITY_FEATURES):
    with amenity_cols[i % 3]:
        amenity_values[feature] = st.checkbox(AMENITY_LABELS[feature])

if st.button("Predict Price", type="primary"):
    input_data = {
        "state": state,
        "property_type": property_type,
        "star_rating": star_rating,
        "review_rating": review_rating,
        "review_count": review_count,
        "distance_to_landmark_km": distance_to_landmark_km,
        **{feature: int(value) for feature, value in amenity_values.items()},
    }
    try:
        price = predict_price(input_data, model=model)
        st.subheader("Estimated Price")
        st.metric(label="Predicted price", value=f"₹{price:,.0f} / night")
    except Exception as exc:  # keep the app alive and show a clean message
        st.error(f"Couldn't generate a prediction: {exc}")
