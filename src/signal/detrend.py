import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd


def fit_baseline_ransac(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    min_samples: float = 0.5,
    residual_threshold: float = None,
    random_state: int = 42
) -> object:
    """
    Fit a robust baseline to the data using RANSAC.

    Args:
        x: 1D array of independent variable (e.g., frame indices or time).
        y: 1D array of dependent variable (e.g., track coordinate).
        degree: polynomial degree for baseline fit (1 = linear).
        min_samples: proportion (0<min_samples<=1) or absolute count for RANSAC.
        residual_threshold: maximum residual for inlier classification. If None,
            defaults to median absolute deviation of y.
        random_state: seed for reproducibility.

    Returns:
        ransac: fitted RANSAC model (scikit-learn pipeline).
    """
    # reshape inputs
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    # default residual threshold
    if residual_threshold is None:
        residual_threshold = np.median(np.abs(y - np.median(y)))

    # create polynomial + RANSAC pipeline
    model = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        RANSACRegressor(
            base_estimator=LinearRegression(),
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            random_state=random_state
        )
    )
    model.fit(x, y)
    return model


def detrend_residual(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    **ransac_kwargs
) -> np.ndarray:
    """
    Remove baseline trend from y by fitting and subtracting a RANSAC baseline.

    Args:
        x: 1D array of independent variable.
        y: 1D array of dependent variable.
        degree: polynomial degree for baseline.
        **ransac_kwargs: extra parameters passed to fit_baseline_ransac.

    Returns:
        residual: detrended signal y - y_baseline.
    """
    # fit baseline
    model = fit_baseline_ransac(x, y, degree=degree, **ransac_kwargs)

    # predict baseline
    x_reshaped = np.array(x).reshape(-1, 1)
    y_baseline = model.predict(x_reshaped)

    # compute residual
    residual = y - y_baseline
    return residual


def detrend_dataframe(
    df: pd.DataFrame,
    x_col: str = 'frame',
    y_col: str = 'position',
    degree: int = 1,
    **ransac_kwargs
) -> pd.DataFrame:
    """
    Given a DataFrame with x/y columns, fit and subtract baseline to add a residual column.

    Args:
        df: DataFrame containing at least x_col and y_col.
        x_col: name of independent variable column.
        y_col: name of dependent variable column.
        degree: polynomial degree for baseline.
        **ransac_kwargs: passed to fit_baseline_ransac.

    Returns:
        df_out: copy of df with an added 'residual' column.
    """
    df_out = df.copy()
    x = df_out[x_col].values
    y = df_out[y_col].values
    residual = detrend_residual(x, y, degree=degree, **ransac_kwargs)
    df_out['residual'] = residual
    return df_out
