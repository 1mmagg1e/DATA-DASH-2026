"""Final off-market model: 12-month Box-Cox MLR + lot features.

fit(train, cutoff=None) selects start <= date_sold < cutoff, where start is
12 months before cutoff. With complete monthly training data, the default cutoff
is the first day after the last training month. For the final test batch, use
cutoff='2026-06-01' (June 2025 through May 2026 training sales) and keep the
same fitted model for all June--September homes.

Predictors: bedrooms, washrooms, parking_total, house_area (parsed midpoint),
house_type_name, house_style, municipality, community, lot_front, lot_depth.
No list price, sale date, days on market, or AVM estimate enters predict().

The first four numeric features and sold_price use training-only Box-Cox
parameters. Lot dimensions use median imputation and standardization, with two
missing indicators. Categorical features use drop-first one-hot encoding.
Regression is ordinary least squares, not Ridge. Target inversion is direct,
without smearing correction. Predictions are unrounded dollars.
"""
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ["bedrooms", "washrooms", "parking_total", "house_area_mid"]
CATEGORICAL = ["house_type_name", "house_style", "municipality", "community"]
LOT = ["lot_front", "lot_depth"]
SHIFTS = dict(zip(NUMERIC, [1.0, 1.0, 1.0, 0.0]))


def parse_area(value):
    """Match the off-market notebook's numeric/range midpoint conversion."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("–", "-").replace("—", "-").replace(",", "")
    try:
        return float(text)
    except ValueError:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*", text)
        return (float(match[1]) + float(match[2])) / 2 if match else np.nan


class FinalOffMarketModel:
    @staticmethod
    def prepare_features(homes):
        X = homes.copy()
        if "house_area_mid" not in X:
            X["house_area_mid"] = X["house_area"].map(parse_area)
        X = X[NUMERIC + CATEGORICAL + LOT].copy()
        for column in NUMERIC + LOT:
            X[column] = pd.to_numeric(X[column], errors="raise").astype(float)
            observed = X[column].dropna().to_numpy()
            if not np.isfinite(observed).all() or (observed < 0).any():
                raise ValueError(f"{column}: nonfinite or negative values.")
        return X

    def _transform_inputs(self, X):
        X = X.copy()
        for column in NUMERIC:
            values = X[column].fillna(self.medians_[column]).to_numpy(dtype=float)
            shifted = values + SHIFTS[column]
            if not np.isfinite(shifted).all() or (shifted <= 0).any():
                raise ValueError(f"{column}: invalid Box-Cox domain.")
            lam = self.lambdas_[column]
            X[column] = values if lam is None else stats.boxcox(shifted, lmbda=lam)
        for column in LOT:
            X[column + "_missing"] = X[column].isna().astype(float)
        return X

    def fit(self, train, cutoff=None):
        dates = pd.to_datetime(train["date_sold"], errors="raise")
        if train.empty or dates.isna().any():
            raise ValueError("Training data is empty or contains missing sale dates.")
        if cutoff is None:
            cutoff = (dates.max().to_period("M") + 1).start_time
        self.cutoff_ = pd.Timestamp(cutoff)
        self.window_start_ = self.cutoff_ - pd.DateOffset(months=12)
        selected = train.loc[(dates >= self.window_start_) & (dates < self.cutoff_)]
        y = pd.to_numeric(selected["sold_price"], errors="raise").to_numpy(dtype=float)
        if selected.empty or not np.isfinite(y).all() or (y <= 0).any():
            raise ValueError("Empty training window or invalid sold prices.")
        if np.ptp(y) == 0:
            raise ValueError("Box-Cox target requires nonconstant training prices.")

        X = self.prepare_features(selected)
        self.medians_, self.lambdas_ = {}, {}
        for column in NUMERIC:
            median = X[column].median()
            if not np.isfinite(median):
                raise ValueError(f"{column}: no observed training values.")
            self.medians_[column] = float(median)
            values = X[column].fillna(median).to_numpy(dtype=float)
            shifted = values + SHIFTS[column]
            if (shifted <= 0).any():
                raise ValueError(f"{column}: invalid Box-Cox domain.")
            self.lambdas_[column] = (
                None if np.ptp(values) == 0 else float(stats.boxcox(shifted)[1])
            )
        for column in LOT:
            if X[column].notna().sum() == 0:
                raise ValueError(f"{column}: no observed training values.")
        target, self.y_lambda_ = stats.boxcox(y)

        preprocessing = ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore",
                                          sparse_output=False)),
            ]), CATEGORICAL),
            ("lot", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), LOT),
            ("lot_missing", "passthrough", [column + "_missing" for column in LOT]),
        ])
        self.pipeline_ = Pipeline([
            ("preprocessing", preprocessing),
            ("regression", LinearRegression()),
        ])
        self.pipeline_.fit(self._transform_inputs(X), target)
        self.training_n_ = len(selected)
        return self

    def predict(self, homes):
        transformed = self.pipeline_.predict(
            self._transform_inputs(self.prepare_features(homes))
        )
        domain_ok = np.isfinite(transformed)
        if self.y_lambda_ != 0:
            domain_ok &= (1 + self.y_lambda_ * transformed) > 0
        if not domain_ok.all():
            raise ValueError("Prediction is outside the inverse Box-Cox domain.")
        with np.errstate(over="ignore", invalid="ignore"):
            predicted = np.asarray(inv_boxcox(transformed, self.y_lambda_), dtype=float)
        if predicted.shape != (len(homes),) or not np.isfinite(predicted).all() or (predicted <= 0).any():
            raise ValueError("Invalid or nonpositive dollar predictions.")
        return predicted
