"""Simple MLP training and evaluation helpers."""

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def create_mlp(random_state: int = 42) -> MLPClassifier:
	"""Create a small beginner-friendly neural-network classifier."""
	return MLPClassifier(
		hidden_layer_sizes=(32,),
		max_iter=100,
		batch_size=256,
		early_stopping=True,
		n_iter_no_change=5,
		random_state=random_state,
	)


def train_mlp(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	random_state: int = 42,
	model_path: str | Path | None = None,
) -> tuple[Pipeline, float]:
	"""Fit a preprocessing-plus-MLP pipeline and return training time."""
	model = Pipeline(
		steps=[
			("preprocessor", build_preprocessor(X_train)),
			("classifier", create_mlp(random_state)),
		]
	)
	start = time.perf_counter()
	model.fit(X_train, y_train)
	training_time = time.perf_counter() - start
	if model_path is not None:
		Path(model_path).parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(model, model_path)
	return model, training_time