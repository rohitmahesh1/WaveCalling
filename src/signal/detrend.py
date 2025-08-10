import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def _mad(a: np.ndarray) -> float:
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def _make_ransac(estimator, **kwargs) -> RANSACRegressor:
    """
    Create a RANSACRegressor compatible with both old (base_estimator=)
    and new (estimator=) scikit-learn versions.
    """
    try:
        # Newer scikit-learn (>=1.1)
        return RANSACRegressor(estimator=estimator, **kwargs)
    except TypeError:
        # Older scikit-learn (<1.1)
        return RANSACRegressor(base_estimator=estimator, **kwargs)


def _fit_poly_baseline(x: np.ndarray, y: np.ndarray, degree: int) -> Pipeline:
    """
    Plain polynomial least-squares baseline (no RANSAC), used as fallback.
    """
    model = Pipeline([
        ("poly", PolynomialFeatures(degree, include_bias=False)),
        ("lin", LinearRegression()),
    ])
    model.fit(x.reshape(-1, 1), y)
    return model


def fit_baseline_ransac(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    min_samples: float | int = 0.5,
    residual_threshold: float | None = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Fit a robust polynomial baseline with RANSAC.
    Returns a Pipeline(poly -> ransac).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Default residual threshold = MAD(y); fallback to 0.5*std if MAD==0
    if residual_threshold is None:
        rt = _mad(y)
        if not np.isfinite(rt) or rt == 0.0:
            s = float(np.std(y))
            rt = 0.5 * s if s > 0 else 1.0
        residual_threshold = float(rt)

    ransac = _make_ransac(
        LinearRegression(),
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        random_state=random_state,
    )

    model = Pipeline([
        ("poly", PolynomialFeatures(degree, include_bias=False)),
        ("ransac", ransac),
    ])

    # If RANSAC fails (too few inliers, etc.), fall back to plain poly LSQ
    try:
        model.fit(x.reshape(-1, 1), y)
        return model
    except Exception:
        return _fit_poly_baseline(x, y, degree)


def detrend_residual(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    **ransac_kwargs
) -> np.ndarray:
    """
    Remove baseline trend from y by fitting and subtracting a robust baseline.
    Falls back to LSQ poly if RANSAC fails.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    try:
        model = fit_baseline_ransac(x, y, degree=degree, **ransac_kwargs)
    except Exception:
        model = _fit_poly_baseline(x, y, degree)

    y_baseline = model.predict(x.reshape(-1, 1))
    return y - y_baseline


def detrend_dataframe(
    df: pd.DataFrame,
    x_col: str = 'frame',
    y_col: str = 'position',
    degree: int = 1,
    **ransac_kwargs
) -> pd.DataFrame:
    """
    Add a 'residual' column to df by subtracting a robust polynomial baseline.
    """
    df_out = df.copy()
    x = df_out[x_col].to_numpy()
    y = df_out[y_col].to_numpy()
    df_out['residual'] = detrend_residual(x, y, degree=degree, **ransac_kwargs)
    return df_out
