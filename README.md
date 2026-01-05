# NBA Game Predictor

This project predicts NBA game outcomes using historical game data and rolling team statistics. The prediction pipeline combines **RidgeClassifier** (for winner prediction) and **Logistic Regression** (for home win probability) in a weighted ensemble.

---

## Project Structure

```
.
├── scrape/
│   ├── fetch_nba_seasons.py      # Downloads season schedule HTML files
│   ├── read_nba_seasons.py       # Downloads individual box score HTML files
│   ├── parse_nba_data.py         # Parses box score HTML files into structured CSV
│   ├── preprocess_nba_data.py    # Loads and cleans parsed CSV data
│   └── data/
│       └── nba_games.csv         # Parsed game data CSV
│
├── ml/
│   ├── data/
│   │   └── rolling_df.csv        # Rolling features for model input
│   ├── features/
│   │   └── rolling_features.py   # Computes rolling averages for each team
│   ├── models/                   # Trained model files
│   │   ├── logistic_model_final.pkl
│   │   ├── ridge_classifier_final.pkl
│   │   ├── selected_predictors_logistic.pkl
│   │   └── selected_predictors_ridge.pkl
│   ├── predictor/
│   │   └── ensemble_predictor.py # Weighted ensemble prediction function
│   ├── train/
│   │   ├── train_nba_model_log_reg.py  # Logistic regression training
│   │   └── train_nba_model_ridge.py    # Ridge classifier training
│   ├── test/
│   │   ├── test_log_reg.py       # Test script for Logistic Regression
│   │   └── test_ridge.py         # Test script for Ridge Classifier
│   └── predict_game.py           # User-facing script to predict a single game
```

---

## Data Scraping (Important Notes)

The repository includes scripts to scrape NBA game data from **Basketball-Reference**, but please note:

* **`fetch_nba_seasons.py`**
  Downloads all season schedules and standings HTML files.

  * **Time:** Can take **a couple of hours** depending on the number of seasons.
  * Extracts HTML for **all games from the 2020 season to the most recent season (2026)**.

* **`read_nba_seasons.py`**
  Downloads **individual box score pages** for every game in the saved HTML standings files.

  * **Time:** Can take **almost a full day** because it processes all games month by month and season by season.

* **`parse_nba_data.py`**
  Converts all saved box score HTML files into a structured CSV (`nba_games.csv`) for model training.

> **Important:** These scraping scripts are **not meant to be run** in this repository.
> All models have already been trained, and the HTML files are **not included** due to size constraints.

**Data Range Used for Model Training:**

* All models are trained on games from the **start of the 2020 NBA season up to 01/02/2026**.
* To update the dataset beyond this, you would need to run the scraping pipeline (`fetch_nba_seasons.py` → `read_nba_seasons.py` → `parse_nba_data.py`), which takes several hours to a full day.
* The included `rolling_df.csv` and trained models reflect **only the data from the beginning of the 2020 season up to 01/02/2026**.

> The scraping scripts are included to show methodology and reproducibility, but **running them is optional and not required to use the prediction models**.

---

## Machine Learning Pipeline

1. **Data Preparation**

   * Load the parsed `nba_games.csv`.
   * Compute rolling statistics for each team using `rolling_features.py`.
   * Align rolling stats to the next game for matchup modeling.

2. **Model Training**

   * **Ridge Classifier:** Predicts winner (home vs away).
   * **Logistic Regression:** Predicts home win probability.
   * Sequential Feature Selection is used to select top predictive features.
   * Trained models are saved in `ml/models/`.

3. **Prediction**

   * Use `predict_game.py` for user-friendly predictions.
   * Ensemble combines Ridge (winner) + Logistic (probability) predictions.

---

## Usage

### Predict a Single Game

Run the `predict_game.py` script with home and away team abbreviations:

```bash
python ml/predict_game.py LAL BOS
```

Example output:

```
Home team: LAL
Away team: BOS
Winner: BOS
Loser: LAL
Home win probability: 45.23%
Away win probability: 54.77%
```

### Using Models in Python

You can also use the ensemble predictor directly:

```python
import pandas as pd
import joblib
from ml.predictor.ensemble_predictor import predict_game_ensemble_weighted

# Load rolling data
rolling_df = pd.read_csv("ml/data/rolling_df.csv")

# Load models and predictor columns
ridge_model = joblib.load("ml/models/ridge_classifier_final.pkl")
ridge_predictors = joblib.load("ml/models/selected_predictors_ridge.pkl")
logistic_model = joblib.load("ml/models/logistic_model_final.pkl")
logistic_predictors = joblib.load("ml/models/selected_predictors_logistic.pkl")

# Predict a game
result = predict_game_ensemble_weighted(
    rolling_df,
    ridge_model,
    ridge_predictors,
    logistic_model,
    logistic_predictors,
    home_team="LAL",
    away_team="BOS"
)

print(result)
```

---

## Notes

* Rolling features are computed over a **10-game window by default**.
* Models were trained on **historical data from 2020–2026 only**. Predictions outside this range may be unreliable.
* Scraping scripts demonstrate the methodology, but **running them is not necessary** for predictions.
