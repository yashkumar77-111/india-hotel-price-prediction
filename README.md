# Airbnb India Price Predictor

A machine learning project that predicts the nightly price of an Airbnb
listing in India, based on its state, property type, location, guest
capacity, and superhost status. Includes a Streamlit web app for
interactive predictions.

## ⚠️ A note on dataset size

This project is trained on **500 real India Airbnb listings** (492 after
cleaning). That is a small dataset for machine learning — for comparison,
a typical Airbnb pricing project (e.g. the classic NYC listings dataset)
uses 40,000+ rows. With this little data, the model can pick up on real,
broad patterns (e.g. "villas cost more than homestays," "Goa prices
differ from smaller towns"), but predictions carry a wide margin of
error and won't be precise for less common combinations. The results
below are reported honestly, not adjusted to look better than they are.

## Project structure

```
airbnb-price-prediction/
├── data/
│   └── listings.csv          # Raw India Airbnb listings
├── src/
│   ├── train.py               # Data cleaning + model training
│   └── predict.py             # Prediction helper
├── models/
│   └── model.pkl               # Trained pipeline (created by running main.py)
├── main.py                     # Entry point: trains and saves the model
├── app.py                      # Streamlit web app
├── requirements.txt
└── README.md
```

## Setup

```bash
cd airbnb-price-prediction
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

## Train the model

```bash
python main.py
```

This loads `data/listings.csv`, cleans it, trains and compares three
models, and saves the best one to `models/model.pkl`. Takes a few
seconds given the small dataset.

## Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Pick a state, property type, number of
guests, and superhost status, then click **Predict Price** for an
estimated nightly rate in ₹.

## Data cleaning

The raw data needed real cleanup before it was usable:

- **State, not city.** The raw `address` column has 206 different cities
  across only 500 rows — most appear once or twice, far too sparse for a
  model to learn from. The state was extracted instead (e.g. "Manali,
  Himachal Pradesh, India" → "Himachal Pradesh"), giving each category
  enough listings to be meaningful. Abbreviations, misspellings, and
  non-English entries were standardized to one canonical name per state;
  rare states became an "Other" bucket.
- **Property type, not raw room type.** The raw `roomType` column had 40
  inconsistent values mixing property types (Villa, Apartment) with room
  configurations (Shared room), many with only 1–2 examples. These were
  grouped into 5 consistent property-type categories — Villa, Homestay,
  Hotel/Resort, Apartment, Other — matching how travel sites like
  MakeMyTrip categorize stays.
- **Dropped `stars`.** Missing for 61% of listings (304 of 500) — too
  unreliable to keep.
- **Removed price outliers.** 8 listings (1.6%) priced above ₹5,000/night
  were excluded as extreme luxury outliers that would otherwise distort
  predictions for typical listings.

## Modeling approach

Three regression models were trained and compared on a held-out 20% test
set: Linear Regression, Random Forest, and Gradient Boosting — each
wrapped in a pipeline that handles missing values, scales numeric
features, and one-hot encodes categorical ones. The best model by MAE
was then lightly tuned. Tree depths and estimator counts were kept
smaller than a large-dataset project would use, since a complex model
would simply memorize 492 rows instead of learning general patterns.

### Results (held-out 20% test set)

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---|---|---|
| Linear Regression | 465.53 | 695.47 | 0.244 |
| Random Forest | 408.71 | 617.91 | 0.404 |
| **Gradient Boosting (saved)** | **397.57** | **669.30** | **0.300** |

**Gradient Boosting** was selected (lowest MAE). On average, its price
predictions are off by about ₹398. The R² of 0.30 means the model
explains roughly 30% of the variation in price — a real but modest
result, consistent with a 500-row dataset covering many different
states and property types.

## Feature list

- `state` — categorical, cleaned from the address
- `propertyType` — categorical, grouped from the raw room type
- `latitude`, `longitude` — numeric
- `numberOfGuests` — numeric
- `isHostedBySuperhost` — numeric (0/1)
