"""Check saved official validation outputs and the standalone off-market model.

Run after `python run.py validate`. DATA_DIR may point to local listings.csv.
Expected off-market scores below come from the team's notebook screenshots;
matching rounded metrics does not assert prediction-by-prediction identity.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
import models
from final_off_market import parse_area


def main():
    root = Path(__file__).resolve().parent
    predictions = pd.read_csv(root / "outputs" / "validation_predictions.csv")
    listings = data.load("listings")
    dates = listings["date_sold"].dt.to_period("M")
    expected_homes = listings.loc[dates.between(pd.Period("2025-12"), pd.Period("2026-05"))]
    expected = expected_homes[["ml_num", "sold_price"]].copy()
    expected["month"] = expected_homes["date_sold"].dt.to_period("M").astype(str)
    expected_scores = {"off_market": [11.746, 14.945], "list_aware": [4.453, 5.656]}
    summary = {}
    assert set(predictions["track"]) == set(expected_scores)
    for track, scores in expected_scores.items():
        rows = predictions.loc[predictions["track"].eq(track)]
        paired = expected.merge(rows, on=["ml_num", "month"], how="outer",
                                validate="one_to_one", indicator=True, suffixes=("_expected", ""))
        assert paired["_merge"].eq("both").all()
        np.testing.assert_array_equal(paired.sold_price_expected, paired.sold_price)
        p = paired.predicted_price.to_numpy()
        assert np.isfinite(p).all() and (p > 0).all()
        signed = 100 * (p - paired.sold_price.to_numpy()) / paired.sold_price.to_numpy()
        score = [float(np.median(abs(signed))), float(np.mean(abs(signed)))]
        np.testing.assert_array_equal(np.round(score, 3), scores)
        summary[track] = {"N": len(rows), "MdAPE_pct": score[0], "MAPE_pct": score[1],
                          "Median_signed_error_pct": float(np.median(signed))}
        print(f"PASS {track}: {len(rows):,} homes; MdAPE {score[0]:.3f}%; MAPE {score[1]:.3f}%")

    off = predictions.loc[predictions.track.eq("off_market")].copy()
    off["APE"] = 100 * abs(off.predicted_price - off.sold_price) / off.sold_price
    monthly = off.groupby("month").APE.agg(["median", "mean"])
    np.testing.assert_array_equal(monthly.round(3).to_numpy(), [
        [13.751, 16.676], [12.901, 15.854], [11.036, 14.577],
        [10.935, 14.377], [11.520, 14.454], [11.705, 14.748],
    ])

    # Fit the final cutoff and test the actual test-like prediction interface.
    model = models.make_model("off_market").fit(listings, cutoff="2026-06-01")
    selected = listings.loc[listings.date_sold.ge("2025-06-01") & listings.date_sold.lt("2026-06-01")]
    assert model.training_n_ == len(selected)
    homes = data.hide_answers(listings.tail(100).copy(), "off_market")
    before = homes.copy(deep=True)
    raw_prediction = model.predict(homes)
    pd.testing.assert_frame_equal(homes, before)
    prepared = homes.copy()
    prepared["house_area_mid"] = prepared.house_area.map(parse_area)
    np.testing.assert_array_equal(raw_prediction, model.predict(prepared))
    poisoned = homes.copy()
    for column in data.AFTER_SALE_COLUMNS + ["list_price"]:
        poisoned[column] = "FORBIDDEN_INPUT"
    np.testing.assert_array_equal(raw_prediction, model.predict(poisoned))
    print("PASS monthly scores, final training window, raw area reconstruction, and excluded inputs.")
    summary["checks"] = {"status": "PASS", "final_off_market_training_n": model.training_n_,
                          "notebook_comparison": "pooled and monthly scores match to 3 decimals",
                          "test_predictions_generated": False}
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "final_model_verification.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
