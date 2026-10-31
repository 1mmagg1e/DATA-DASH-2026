"""Standalone list-aware model: 12-month Ridge + community + area spline.

Dependencies: numpy, pandas, scikit-learn (OneHotEncoder.sparse_output).
No other project modules or notebook variables are required.

Usage:
    from final_list_aware import FinalListAwareModel
    model = FinalListAwareModel().fit(train, cutoff="2026-06-01")
    predictions = model.predict(homes)

For this cutoff, training sales are restricted to June 1, 2025 through
May 31, 2026. Dates are filtered with start <= date_sold < cutoff.
Use the same fitted model for the June--September test batch when future
sale labels are unavailable. predict() does not require date_sold or sold_price.

Raw prediction columns:
    list_price, bedrooms, washrooms, parking_total, house_area,
    house_type_name, house_style, municipality, community.
Prepared house_area_mid/community_location may replace their source fields.
Training additionally requires date_sold and sold_price.

Fixed settings: alpha=100; cubic area spline with four quantile knots,
include_bias=False and linear extrapolation. All learned preprocessing is
fitted on the selected training window only. Predictions are unrounded dollar
amounts; round once when scoring or writing official submissions.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def make_area_spline_ridge():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    area = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=4, degree=3, knots="quantile",
                                     include_bias=False, extrapolation="linear")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop=None,
                                  sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, ["list_price", "bedrooms", "washrooms", "parking_total"]),
        ("area_spline", area, ["house_area_mid"]),
        ("categorical", categorical,
         ["house_type_name", "house_style", "community_location"]),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("ridge", Ridge(alpha=100, solver="cholesky")),
    ])


class FinalListAwareModel:
    @staticmethod
    def prepare_features(homes):
        X = homes.copy()
        numeric = ["list_price", "bedrooms", "washrooms", "parking_total"]
        for column in numeric:
            X[column] = pd.to_numeric(X[column], errors="coerce")
        if "house_area_mid" not in X:
            parts = X["house_area"].astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
            low = pd.to_numeric(parts[0], errors="coerce")
            high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
            X["house_area_mid"] = (low + high) / 2
        X["house_area_mid"] = pd.to_numeric(X["house_area_mid"], errors="coerce")
        if "community_location" not in X:
            location = {
                column: X[column].astype("string").str.strip().replace("", pd.NA).fillna("Unknown")
                for column in ["municipality", "community"]
            }
            X["community_location"] = location["municipality"] + " | " + location["community"]
        return X[numeric + ["house_area_mid", "house_type_name", "house_style", "community_location"]]

    def fit(self, train, cutoff=None):
        dates = pd.to_datetime(train["date_sold"], errors="raise")
        if train.empty or dates.isna().any():
            raise ValueError("Training data is empty or contains missing sale dates.")
        # Official loader supplies complete months; no test dates or labels are used.
        if cutoff is None:
            cutoff = (dates.max().to_period("M") + 1).start_time
        self.cutoff_ = pd.Timestamp(cutoff)
        self.window_start_ = self.cutoff_ - pd.DateOffset(months=12)
        selected = train.loc[(dates >= self.window_start_) & (dates < self.cutoff_)].copy()
        y = pd.to_numeric(selected["sold_price"], errors="raise").to_numpy(dtype=float)
        if selected.empty or not (np.isfinite(y).all() and (y > 0).all()):
            raise ValueError("Empty training window or invalid prices.")
        self.model_ = make_area_spline_ridge()
        self.model_.fit(self.prepare_features(selected), y)
        self.training_n_ = len(selected)
        return self

    def predict(self, homes):
        predicted = np.asarray(self.model_.predict(self.prepare_features(homes)), dtype=float).reshape(-1)
        if len(predicted) != len(homes) or not np.isfinite(predicted).all():
            raise ValueError("Invalid predictions.")
        return predicted
