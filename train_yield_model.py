from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from database import count_yield_records, import_sample_data, load_yield_records


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sample_data.csv"
MODEL_PATH = BASE_DIR / "crop_yield_model.pkl"

FEATURE_COLUMNS = [
    "temperature_c",
    "rainfall_mm",
    "irrigation_mm",
    "soil_n",
    "soil_p",
    "soil_k",
    "fertilizer_kg",
]
TARGET_COLUMN = "yield_t_per_ha"


def main():
    if count_yield_records() == 0:
        import_sample_data(replace=True)

    data = load_yield_records()
    if data.empty:
        data = pd.read_csv(DATA_PATH)

    print("Dataset columns:", data.columns.tolist())

    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)

    score = pipeline.score(x_test, y_test)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Validation R2 score: {score:.3f}")
    print(f"Model trained and saved as {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
