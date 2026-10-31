# SSU Data Dash 2026

Housing sale-price prediction for SSU Data Dash 2026.

## Project status

This repository is being assembled for the team submission. The list-aware model below has been selected and validated locally. Model code, reproducible execution instructions, team details, the off-market model, and final submission files will be added during integration.

## Selected list-aware model

**Ridge regression with community features and a cubic spline for house area, trained on the most recent 12 months.**

- Ridge alpha: **100**.
- Area spline: cubic (degree 3), **4 quantile knots**, linear extrapolation.
- Numeric inputs: list price, bedrooms, washrooms, parking total, and house-area midpoint.
- Categorical inputs: property type, house style, and community location (municipality + community).
- Numeric missing values are imputed using training medians. Numeric features and spline features are standardized.
- Categorical inputs use most-frequent imputation and one-hot encoding, with unseen categories ignored.
- All learned preprocessing is fitted only on the selected training window.

## Validation

Walk-forward validation covers **December 2025–May 2026**. Each month is predicted using sales from the preceding 12 months; the target month is excluded from training. Validation homes are grouped by sale month. Predictions are rounded to dollars before scoring.

On **10,069 validation homes**:

- **MdAPE: 4.453%**
- **MAPE: 5.656%**
- **Median signed percentage error: +0.791%**

Signed percentage error is `100 * (predicted - actual) / actual`; positive values indicate overprediction. A local run using the official data loader and list-aware validation/report functions reproduced these results.

These months were used for model selection. The scores are validation results, not independent hidden-test performance.

## Main findings

Restricting training to recent observations produced the largest observed improvement. The 12-month window outperformed the tested 24-month and all-history windows. Adding the area spline provided a modest further gain: pooled MdAPE decreased from 4.495% to 4.453%, and MAPE from 5.670% to 5.656%. Monthly MdAPE improved in five of six months and MAPE in all six. No statistical-significance claim is made for this small gain.

## Data

Competition dataset: [Baliyang/SSU_data_dash](https://huggingface.co/datasets/Baliyang/SSU_data_dash).

The selected list-aware model uses the supplied competition fields. Post-sale fields and external AVM estimates are not prediction inputs.

## Final prediction protocol

With training labels available through May 2026, fit the final model on **June 2025–May 2026** and use that fitted model for the June–September test batch. Do not update it using unavailable future sale prices.

## Remaining integration

- Add team name and full member names.
- Add the selected off-market model and validation results.
- Upload the model code, dependencies, and tested reproduction commands.
- Generate and check both final submission CSV files.
- Add the final report and video link.
