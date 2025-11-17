import os

# Reproducibility
SEED = 42
TEST_SIZE = 0.2

# File locations
RAW_CSV = os.path.join("data", "raw", "heart_dataset.csv")
PROCESSED_DIR = os.path.join("data", "processed")
REPORTS_DIR = "reports"
FIGS_DIR = os.path.join(REPORTS_DIR, "figs")

# Preprocessing variants
VariantA = {
    "name": "VariantA_mean_minmax",
    "numeric_imputation": "mean",       # options: "mean", "median", "knn", etc.
    "categorical_imputation": "mode",   # options: "mode", "constant", "center_mode"
    "scaler": "minmax",                 # options: "minmax", "standard"
    "notes": "Numeric imputation with mean + Min-Max scaling. May compress distances into [0,1], amplifying relative differences for features with originally small ranges."
}

VariantB = {
    "name": "VariantB_median_standard",
    "numeric_imputation": "median",
    "categorical_imputation": "mode",
    "scaler": "standard",               # z-score standardization
    "notes": "Numeric imputation with median + Z-score scaling. Median is robust to outliers; standard scaling centers features which affects Euclidean distance differently than Min-Max."
}

PREPROCESSING_VARIANTS = [VariantA, VariantB]

# Model grid and evaluation settings
K_GRID = [3, 5, 7, 9, 11, 13, 15]     # odd k values to try; extendable to 31 if desired
CV_FOLDS = 5                          # number of cross-validation folds
PRIMARY_METRIC = "f1"                 # primary evaluation metric: "f1" (use "f1_macro" or "f1_weighted" if classes imbalanced)

# Other experiment defaults
RESULTS_DIR = os.path.join(REPORTS_DIR, "results")
MODEL_ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "artifacts")
