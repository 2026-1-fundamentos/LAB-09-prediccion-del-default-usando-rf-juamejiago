"""Train the credit default model and export the graded artifacts."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	balanced_accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "files" / "input"
MODEL_PATH = BASE_DIR / "files" / "models" / "model.pkl.gz"
METRICS_PATH = BASE_DIR / "files" / "output" / "metrics.json"
TARGET_COLUMN = "default payment next month"
FEATURE_COLUMNS = [
	"LIMIT_BAL",
	"SEX",
	"EDUCATION",
	"MARRIAGE",
	"AGE",
	"PAY_0",
	"PAY_2",
	"PAY_3",
	"PAY_4",
	"PAY_5",
	"PAY_6",
	"BILL_AMT1",
	"BILL_AMT2",
	"BILL_AMT3",
	"BILL_AMT4",
	"BILL_AMT5",
	"BILL_AMT6",
	"PAY_AMT1",
	"PAY_AMT2",
	"PAY_AMT3",
	"PAY_AMT4",
	"PAY_AMT5",
	"PAY_AMT6",
]
CATEGORICAL_COLUMNS = [
	"SEX",
	"EDUCATION",
	"MARRIAGE",
	"PAY_0",
	"PAY_2",
	"PAY_3",
	"PAY_4",
	"PAY_5",
	"PAY_6",
]


def load_and_clean_data(path: Path) -> pd.DataFrame:
	"""Load a dataset and apply the required cleaning steps."""

	data = pd.read_csv(path)
	data = data.rename(columns={TARGET_COLUMN: "default"})
	data = data.drop(columns=["ID"])
	data = data.dropna().copy()
	data["EDUCATION"] = data["EDUCATION"].where(data["EDUCATION"] <= 4, 4)
	return data


def build_pipeline() -> Pipeline:
	"""Create the preprocessing and modeling pipeline."""

	preprocessor = ColumnTransformer(
		transformers=[
			(
				"onehot",
				OneHotEncoder(handle_unknown="ignore"),
				CATEGORICAL_COLUMNS,
			)
		],
		remainder="passthrough",
	)

	return Pipeline(
		steps=[
			("preprocessor", preprocessor),
			(
				"random_forest",
				RandomForestClassifier(random_state=42, n_jobs=1),
			),
		]
	)


def compute_metrics(
	model: GridSearchCV,
	features: pd.DataFrame,
	target: pd.Series,
	dataset: str,
) -> list[dict[str, object]]:
	"""Compute the requested metrics and confusion matrix payloads."""

	predictions = model.predict(features)
	metrics = {
		"dataset": dataset,
		"precision": precision_score(target, predictions),
		"balanced_accuracy": balanced_accuracy_score(target, predictions),
		"recall": recall_score(target, predictions),
		"f1_score": f1_score(target, predictions),
	}

	tn, fp, fn, tp = confusion_matrix(target, predictions).ravel()
	matrix = {
		"type": "cm_matrix",
		"dataset": dataset,
		"true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
		"true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
	}
	return [metrics, matrix]


def main() -> None:
	"""Train the model and write the grading artifacts."""

	train_data = load_and_clean_data(INPUT_DIR / "train_data.csv.zip")
	test_data = load_and_clean_data(INPUT_DIR / "test_data.csv.zip")

	x_train = train_data[FEATURE_COLUMNS]
	y_train = train_data["default"]
	x_test = test_data[FEATURE_COLUMNS]
	y_test = test_data["default"]

	search = GridSearchCV(
		estimator=build_pipeline(),
		param_grid={
			"random_forest__n_estimators": [20],
			"random_forest__max_depth": [20],
			"random_forest__min_samples_split": [2],
		},
		cv=10,
		scoring="balanced_accuracy",
		n_jobs=1,
	)
	search.fit(x_train, y_train)

	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	with gzip.open(MODEL_PATH, "wb") as file:
		pickle.dump(search, file)

	metrics_payload: list[dict[str, object]] = []
	metrics_payload.extend(compute_metrics(search, x_train, y_train, "train"))
	metrics_payload.extend(compute_metrics(search, x_test, y_test, "test"))

	METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
	with METRICS_PATH.open("w", encoding="utf-8") as file:
		for row in metrics_payload:
			file.write(json.dumps(row))
			file.write("\n")


if __name__ == "__main__":
	main()
#
