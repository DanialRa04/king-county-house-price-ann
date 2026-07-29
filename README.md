# King County House Price ANN

This project predicts house sale prices from the King County housing dataset with a cleaned neural-network regression workflow. It keeps the original idea from `housepre.ipynb`, but fixes the project structure, data path handling, and train/test preprocessing flow so the results are easy to rerun.

## What It Does

- loads `data/kc_house_data.csv`
- engineers sale-date and renovation features
- keeps `zipcode` as a categorical location feature
- compares a median baseline, linear regression, and an ANN regressor
- evaluates on a held-out test split with saved notebook outputs

## Dataset Notes

- Dataset file: `data/kc_house_data.csv`
- Rows: `21,597`
- Target: `price`
- Scope: residential home sales from King County

The dataset is included in this repo because it is small enough for GitHub and required to rerun the notebook.

## Model And Approach

The main model is a fully connected ANN regressor built with scikit-learn's `MLPRegressor`. I keep the workflow leakage-safe by splitting the data before fitting any scaler or encoder.

Key preprocessing choices:

- drop `id`
- parse `date` into `sale_year` and `sale_month`
- add `house_age_at_sale`, `renovated`, and `years_since_renovation`
- standardize numeric features
- one-hot encode `zipcode`
- train the ANN on `log1p(price)` after scaling the transformed target

## How To Run

1. Install the dependencies from `requirements.txt`.
2. Open `king-county-house-price-ann.ipynb` from the repo root.
3. Run the notebook from top to bottom.

The notebook expects the CSV at `data/kc_house_data.csv`.

## Results

Saved test-set metrics from the executed notebook:

- Median baseline: RMSE `370,824.42`, MAE `218,689.96`, R2 `-0.0560`
- Linear regression: RMSE `153,343.23`, MAE `74,395.25`, R2 `0.8194`
- ANN regressor: RMSE `117,529.84`, MAE `66,968.70`, R2 `0.8939`

The ANN was the strongest of the three models in the saved run.

## Limitations

- This is still a tabular model on one county and one time window.
- Sale month and sale year only capture a small part of market drift.
- The model is predictive, not causal, so it should not be treated like an appraisal tool.
