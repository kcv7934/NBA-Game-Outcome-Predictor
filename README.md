# NBA Game Predictor

This project predicts NBA game outcomes using historical game data and rolling team statistics. The prediction pipeline uses machine learning models trained on six NBA seasons (2020–2026), with model performance evaluated through walk-forward season backtesting. The best-performing model achieved **62.85% backtest accuracy** on historical NBA game outcomes.

---

## Project Structure

```text
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
│   │   └── ensemble_predictor.py # Prediction utilities
│   ├── train/
│   │   ├── train_nba_model_log_reg.py
│   │   └── train_nba_model_ridge.py
│   ├── test/
│   │   ├── test_log_reg.py
│   │   ├── test_ridge.py
│   │   └── test_ensemble.py
│   └── predict_game.py
```

---

## Data Scraping (Important Notes)

The repository includes scripts to scrape NBA game data from Basketball Reference, but please note:

### fetch_nba_seasons.py

Downloads all season schedules and standings HTML files.

* Can take several hours depending on the number of seasons.
* Extracts HTML for all games from the 2020 season through the most recent season.

### read_nba_seasons.py

Downloads individual box score pages for every game found in the saved schedule files.

* Can take close to a full day because it processes every game across multiple seasons.

### parse_nba_data.py

Converts downloaded box score HTML files into a structured CSV dataset used for model training.

> Important: These scraping scripts are included for reproducibility and methodology purposes. The raw HTML files are not included in the repository due to size constraints.

### Data Range Used

* Models were trained using NBA games from the start of the 2020 season through January 2, 2026.
* Updating the dataset requires rerunning the scraping pipeline.
* Included trained models and rolling features reflect only data available through January 2, 2026.

---

## Machine Learning Pipeline

### 1. Data Preparation

* Load and clean parsed NBA game data.
* Compute rolling team statistics using a 10-game window.
* Align rolling statistics with future matchups to prevent information leakage.
* Generate matchup-level features for model training.

### 2. Model Training

#### Logistic Regression

* Predicts home-team win probability.
* Uses MinMax scaling and Sequential Feature Selection.
* Achieved **62.85% walk-forward backtest accuracy** on historical NBA game outcomes.

#### Ridge Classifier

* Predicts game winners directly.
* Uses the same rolling feature pipeline and feature selection process.
* Achieved **61.91% walk-forward backtest accuracy** on historical NBA game outcomes.

### 3. Model Evaluation

* Walk-forward season backtesting is used to simulate real-world forecasting.
* Models are trained on prior seasons and evaluated on future seasons.
* This approach provides a more realistic measure of predictive performance than standard train/test splits.

### 4. Prediction

* `predict_game.py` provides user-facing predictions.
* Logistic Regression generates win probabilities.
* Ridge Classification provides winner predictions.
* Ensemble experimentation is included for research purposes.

---

## Usage

### Predict a Single Game

```bash
python ml/predict_game.py LAL BOS
```

Example output:

```text
Home team: LAL
Away team: BOS
Winner: BOS
Loser: LAL
Home win probability: 45.23%
Away win probability: 54.77%
```

---

## Using Models in Python

```python
import pandas as pd
import joblib
from ml.predictor.ensemble_predictor import predict_game_ensemble_weighted

rolling_df = pd.read_csv("ml/data/rolling_df.csv")

ridge_model = joblib.load("ml/models/ridge_classifier_final.pkl")
ridge_predictors = joblib.load("ml/models/selected_predictors_ridge.pkl")

logistic_model = joblib.load("ml/models/logistic_model_final.pkl")
logistic_predictors = joblib.load("ml/models/selected_predictors_logistic.pkl")

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

## Results

| Model               | Walk-Forward Backtest Accuracy |
| ------------------- | ------------------------------ |
| Logistic Regression | 62.85%                         |
| Ridge Classifier    | 61.91%                         |

The Logistic Regression model produced the strongest historical performance and serves as the primary benchmark for prediction quality.

---

## Notes

* Rolling features are computed using a 10-game window by default.
* Models are trained only on historical data from 2020–2026.
* Predictions outside the training period may be less reliable.
* Scraping scripts are included to document methodology and reproducibility but are not required to use the trained models.
