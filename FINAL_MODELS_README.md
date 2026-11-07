# Final regression models

- Off-market: `final_off_market.py`, `FinalOffMarketModel`: 12-month Box-Cox MLR + lot dimensions and their missing indicators. No interactions or listing-time trend.
- List-aware: `final_list_aware.py`, `FinalListAwareModel`: 12-month Ridge (alpha 100) + community + cubic area spline (four quantile knots).
- `models.py` selects these models for the official `run.py` pipeline.

Keep both model files, `models.py`, `features.py`, `data.py`, `run.py`, and `requirements.txt` together. The starter baseline classes retained in `models.py` import `features.py`, though those baselines are not selected.

## Reproduce validation

From this directory, with Python 3.13:

```bash
python3 -m pip install -r requirements.txt
python3 run.py validate
python3 run.py report
python3 verify_final_models.py
```

The official loader downloads data if `DATA_DIR` is unset. To use local CSVs instead, set `DATA_DIR` to the directory containing `listings.csv` (and test CSVs for prediction). For the current local folder layout on macOS/Linux:

```bash
export DATA_DIR="$(cd .. && pwd)"
```

Validation covers December 2025 through May 2026, with each model fitting only the preceding 12 months. The runner prints the number of earlier sales passed into `fit`; each final model applies its own 12-month filter internally. Predictions are rounded to dollars before scoring. Both tracks have 10,069 validation homes.

Off-market pooled MdAPE/MAPE reproduce 11.746%/14.945%; list-aware reproduce 4.453%/5.656%. The off-market monthly metrics also match the supplied notebook screenshots to three decimals. Notebook per-home predictions were not exported, so this is metric reproduction, not a claim of exact per-home identity with Colab.

`verify_final_models.py` checks sample IDs, actual prices, valid predictions, expected pooled/monthly scores, training window, area reconstruction and independence from excluded prediction inputs. Its checks on training homes are interface checks, not an additional performance estimate.

## Final test predictions (next step)

```bash
python3 run.py predict
python3 run.py check
```

For the final batch, both models use June 2025--May 2026 sales and remain fixed for the June--September test homes. With the complete official training file, `fit()` infers the cutoff as June 1, 2026. It can also be passed explicitly when calling the model directly.

The two files should be `submissions/submission_off_market.csv` and `submissions/submission_list_aware.csv`, with columns `ml_num,predicted_price`.

**Pending:** the official `check_submission.py` must be restored from the starter repository before `run.py check` works. The test CSVs are not present in this local DATA_DIR. This extraction/validation task has not generated new submission files or uploaded anything to GitHub. Existing CSVs named with `(1)` have not been verified against these final models. The newly listed download dependencies have not been tested in a fresh environment.
