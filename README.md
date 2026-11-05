# Warsaw Apartment Valuation Model (AVM)

A machine-learning model that estimates the market price of an apartment in Warsaw from its size, building
and location, served through an interactive Dash app with a map of the city and the metro lines.

Built as a high-school coursework project (November 2025). It works well on typical apartments and is not
meant for unusual ones.

## Results

| | |
| --- | --- |
| Model | Random Forest regressor (scikit-learn), 180 trees, max depth 18 |
| Data | 5,471 Warsaw apartment listings (August 2023 snapshot of a public Polish listings dataset) |
| Test set | 20% random hold-out, not used in training |
| Out-of-sample R² | 0.85 |
| Mean absolute error | about 110,000 PLN (median price about 800,000 PLN) |

Metrics are those stored with the trained model in `models/warsaw_valuation_model.pkl`.

## What the model uses

- **Apartment:** area, rooms, floor and floor ratio, building height, year built, parking, balcony, lift,
  security, storage room.
- **Location:** distance to the city centre (Palace of Culture), distance to the nearest metro station on
  lines M1 and M2 (haversine distance to every station), distances to schools, kindergartens, clinics,
  pharmacies, post offices, restaurants and colleges, and an average of the service distances.
- **In the app:** when a user values a single address, missing point-of-interest distances are filled by a
  distance-weighted k-nearest-neighbours model (k = 5) fitted on the listings.

## Limitations

- **Typical apartments only.** Very large, luxury or otherwise unusual apartments are rare in the data, so
  their valuations are much less reliable.
- **Asking prices, not transaction prices,** from a single month (August 2023). The model does not know
  about later price changes.
- **Random train/test split.** Nearby listings can sit on both sides of the split, so the R² is likely
  somewhat optimistic compared with a spatial or time-based split.
- Missing numeric values are filled with the median, and the metro station list covers lines M1 and M2
  as hard-coded coordinates.

## Project structure

```
app.py                     Dash app: valuation form, map with metro lines, R² inspector by feature set
src/data_loader.py         Loads the listings and keeps Warsaw only
src/feature_engineering.py Distances, binary flags, floor ratio, average service distance
src/metro_helper.py        Metro M1/M2 station coordinates and nearest-station distance
src/model_engine.py        Trains the model, reports R²/MAE/RMSE on the hold-out set, saves it
data/apartments.csv        Listings (all Polish cities; Warsaw is filtered in code)
models/                    Trained model with its features and metrics
AVM.pdf                    Short technical write-up
```

## Running it

```bash
pip install -r requirements.txt
python -m src.model_engine   # retrain the model and print R² and MAE (optional)
python app.py                # open http://127.0.0.1:8065
```
