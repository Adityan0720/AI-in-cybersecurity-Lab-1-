"""Decision Tree training and evaluation helpers."""

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def create_decision_tree(random_state: int = 42) -> DecisionTreeClassifier:
	"""Create the required baseline classifier."""
	return DecisionTreeClassifier(random_state=random_state, class_weight="balanced")


def train_decision_tree(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	random_state: int = 42,
	model_path: str | Path | None = None,
) -> tuple[Pipeline, float]:
	"""Fit a preprocessing-plus-tree pipeline and return training time."""
	model = Pipeline(
		steps=[
			("preprocessor", build_preprocessor(X_train)),
			("classifier", create_decision_tree(random_state)),
		]
	)
	start = time.perf_counter()
	model.fit(X_train, y_train)
	training_time = time.perf_counter() - start
	if model_path is not None:
		Path(model_path).parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(model, model_path)
	return model, training_time

