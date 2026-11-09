# SSU Data Dash 2026 — Toronto Home-Price Prediction

Team: The Outliers
Team members: Maggie Huang, Tianyi Zhao, Wenjun Bo, Yibo Zhang

This project predicts home sale prices for two competition tracks:

- **Off-market:** predict sale price without access to list price.
- **List-aware:** predict sale price using the known list price.

Official dataset:
https://huggingface.co/datasets/Baliyang/SSU_data_dash

Presentation Video Link:
https://www.youtube.com/watch?v=d42YsT9QbvA

## Final models

### Off-market

**12-month Box–Cox multiple linear regression with lot features**

Features:
- Bedrooms, washrooms, parking total, and house-area midpoint.
- Property type, house style, municipality, and community.
- Lot frontage, lot depth, and their missing-value indicators.

The four original numerical predictors and the target sale price use Box–Cox transformations. Missing-value imputation, transformation parameters, scaling, and categorical encoding are fitted using training data only.

Lot dimensions are median-imputed and standardized. Categorical features are one-hot encoded. The regression is ordinary least squares; predictions are converted back to dollars using the inverse Box–Cox transformation.

The final specification does not include lot-by-property-type interactions or a listing-time trend.

### List-aware

**12-month Ridge regression with community features and an area spline**

Features:
- List price, bedrooms, washrooms, and parking total.
- House-area midpoint represented using spline features.
- Property type, house style, and community location.

Community location combines municipality and community to distinguish neighbourhoods with the same name.

Fixed settings:
- Ridge alpha: 100.
- Cubic area spline: degree 3 and four quantile knots.
- Linear spline extrapolation.
- Training-only imputation, scaling, and one-hot encoding.

The final specification does not add separate municipality indicators or lot features.

## Validation

We use monthly walk-forward validation from **December 2025 through
May 2026**.

For each validation month:
1. Select sales from the preceding 12 months.
2. Fit preprocessing and the model using those sales only.
3. Predict homes sold during the validation month.
4. Round predictions to dollars before calculating percentage errors.

Each track is evaluated on **10,069 validation homes**.

Results:
- **Off-market:** pooled MdAPE **11.746%**, pooled MAPE **14.945%**.
- **List-aware:** pooled MdAPE **4.453%**, pooled MAPE **5.656%**.

These validation months were used during model selection. The results are not independent hidden-test performance estimates.

Neither model uses sold price, sale date, days on market, or external AVM estimates as prediction features. Sale date is used only to define training windows and validation folds. The off-market model also excludes list price.

## Final test prediction protocol

The official training data contains sale prices through May 2026. Both final models are fitted on sales from **June 1, 2025 through May 31, 2026**.

The same fitted model is then used for all homes in the corresponding June–September 2026 hidden-test dataset. Models are not updated using unavailable test sale prices.

Unlike monthly walk-forward validation, this test protocol keeps the model fixed across the four-month prediction period. Market changes may therefore affect later test months differently.

## Repository files

- `final_off_market.py`: standalone final off-market model.
- `final_list_aware.py`: standalone final list-aware model.
- `models.py`: selects the final model for each track.
- `data.py`: official data loading and removal of unavailable answer fields.
- `features.py`: starter feature utilities retained as a code dependency.
- `run.py`: validation, reporting, prediction, and submission checking.
- `check_submission.py`: official submission-format and test-ID checker.
- `verify_final_models.py`: additional validation reproduction checks.
- `requirements.txt`: Python dependencies.
- `submissions/`: final predictions for both test datasets.

The starter baseline classes remain in `models.py` for reference.
They are not selected for final predictions.

## Setup

Local reproduction was tested with **Python 3.13**.

Clone the repository and enter its directory:

```bash
git clone https://github.com/1mmagg1e/DATA-DSAH-2026.git
cd DATA-DSAH-2026
```

Create an isolated environment on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If the repository is private, GitHub access is required to clone it.

## Data loading

By default, `data.py` downloads the official dataset from Hugging Face and caches it locally.

Alternatively, set `DATA_DIR` to a folder containing:

- `listings.csv`
- `test_off_market.csv`
- `test_list_aware.csv`

The loader also supports the corresponding Parquet files.

Example for macOS/Linux:

```bash
export DATA_DIR="/absolute/path/to/data"
```

If you previously set `DATA_DIR` but want automatic downloading instead:

```bash
unset DATA_DIR
```

## Reproduce validation

Run these commands from the repository's main directory:

```bash
python run.py validate
python run.py report
python verify_final_models.py
```

Generated files include:

- `outputs/validation_predictions.csv`
- `reports/validation_report.md`
- `reports/validation_report.json`
- `reports/results_table.tex`
- `reports/final_model_verification.json`
- `figures/error_by_price_range.png`

The verification script checks validation sample IDs, actual prices, expected metrics, the off-market training window, area reconstruction, and exclusion of unavailable prediction inputs.

The report includes bootstrap intervals. These treat homes as independent; shared market conditions may introduce additional uncertainty.

## Generate and check final submissions

```bash
python run.py predict
python run.py check
```

Expected submission files:

- `submissions/submission_off_market.csv`: 3,974 predictions.
- `submissions/submission_list_aware.csv`: 4,064 predictions.

Both files have exactly two columns:

```csv
ml_num,predicted_price
```

The official checker verifies matching test IDs, duplicate or missing homes, column format, and positive finite prediction values.

Both final files passed the official checker in our local run.
Passing this check confirms submission validity, not predictive accuracy.

## Reproducibility status

Reproducibility was verified on macOS using Python 3.13 in a fresh virtual
environment, with code downloaded from GitHub and dependencies installed
from requirements.txt.

Using local official CSV files, both models reproduced the reported
validation results and passed the additional verification checks.

With DATA_DIR unset, the code successfully downloaded the official
training and test datasets from Hugging Face, generated predictions
for both tracks, and passed the official submission checks.

These checks confirm that the code runs reproducibly and produces valid
submission files. Hidden-test accuracy is evaluated by the organizers
using the withheld sale prices.

## Limitations

- The data is synthetic, and results should not be interpreted as validated
  performance on real Toronto properties.
- Model selection reused the validation period.
- The final model remains fixed throughout the hidden-test period.
- Performance may vary by property type, neighbourhood, and price range.
- No outside datasets are used.

## Acknowledgements

This project builds on the official SSU Data Dash starter repository:
https://github.com/erwinzhang7/SSU-Data-Dash
