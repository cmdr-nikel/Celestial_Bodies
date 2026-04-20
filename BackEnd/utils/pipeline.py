import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Feature definitions
# Extend these lists when adding new datasets (DR18, DR16, etc.)
# ──────────────────────────────────────────────

PHOTOMETRIC_FEATURES = ["u", "g", "r", "i", "z"]
COORD_FEATURES = ["alpha", "delta"]
SPECTRAL_FEATURES = ["redshift"]   # The key feature — can be excluded for the experiment

TARGET_COLUMN = "class"
TARGET_MAPPING = {"STAR": 0, "GALAXY": 1, "QSO": 2}


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the SDSS CSV dataset and returns a clean DataFrame.
    Compatible with DR17 and DR18 (shared column names).
    """
    df = pd.read_csv(filepath)
    return df


def get_feature_list(include_redshift: bool = True) -> list:
    """
    Returns the list of input features to use.
    Set include_redshift=False for the photometric-only experiment.
    """
    features = PHOTOMETRIC_FEATURES + COORD_FEATURES
    if include_redshift:
        features += SPECTRAL_FEATURES
    return features


def encode_target(df: pd.DataFrame) -> pd.Series:
    """
    Encodes the target column: STAR=0, GALAXY=1, QUASAR=2.
    Returns encoded Series.
    """
    return df[TARGET_COLUMN].map(TARGET_MAPPING)


def build_pipeline(include_redshift: bool = True) -> Pipeline:
    """
    Builds a scikit-learn preprocessing pipeline.

    - Applies StandardScaler to all numeric features
    - include_redshift=True  → full feature set (~99% accuracy expected)
    - include_redshift=False → photometric-only (~92-96% accuracy expected)

    Usage:
        from utils.pipeline import build_pipeline, get_feature_list, load_data, encode_target

        df = load_data("data/star_classification.csv")
        features = get_feature_list(include_redshift=True)
        X = df[features]
        y = encode_target(df)
        pipeline = build_pipeline(include_redshift=True)
    """
    features = get_feature_list(include_redshift)

    preprocessor = ColumnTransformer(transformers=[
        ("scaler", StandardScaler(), features)
    ], remainder="drop")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])

    return pipeline


def prepare_train_test(
    df: pd.DataFrame,
    include_redshift: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Full preparation: feature selection, encoding, train/test split.
    Returns X_train, X_test, y_train, y_test (all as numpy arrays, ready for sklearn).

    Usage:
        X_train, X_test, y_train, y_test = prepare_train_test(df, include_redshift=True)
    """
    features = get_feature_list(include_redshift)
    X = df[features]
    y = encode_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y       # preserves class balance in both splits
    )

    pipeline = build_pipeline(include_redshift)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    return X_train_processed, X_test_processed, y_train.values, y_test.values, pipeline