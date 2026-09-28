"""Dataset preparation helpers used before model training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def require_target(data: pd.DataFrame, target_column: str) -> None:
	"""Validate that the inspected dataset contains the requested target."""
	if target_column not in data.columns:
		raise KeyError(f"Target column not found: {target_column}")


def add_binary_target(
	data: pd.DataFrame,
	label_column: str = "label",
	target_column: str = "behavior",
) -> pd.DataFrame:
	"""Add a Normal/Attack target while preserving the original labels."""
	require_target(data, label_column)
	result = data.copy()
	result[target_column] = result[label_column].eq("normal").map(
		{True: "Normal", False: "Attack"}
	)
	return result


@dataclass
class PreparedData:
	"""Processed train/test data and the fitted preprocessing pipeline."""

	preprocessor: ColumnTransformer
	X_train: pd.DataFrame
	X_test: pd.DataFrame
	y_train: pd.Series
	y_test: pd.Series


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
	"""Build imputing and encoding steps from the dataset's actual columns."""
	from sklearn.compose import ColumnTransformer
	from sklearn.impute import SimpleImputer
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import OneHotEncoder, StandardScaler

	numeric_columns = features.select_dtypes(include="number").columns.tolist()
	categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

	numeric_pipeline = Pipeline(
		steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
		]
	)
	return ColumnTransformer(
		transformers=[
			("numeric", numeric_pipeline, numeric_columns),
			("categorical", categorical_pipeline, categorical_columns),
		]
	)


def split_dataset(
	data: pd.DataFrame,
	target_column: str = "behavior",
	label_columns: tuple[str, ...] = ("label", "difficulty_level"),
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Separate features and target with a reproducible stratified split."""
	from sklearn.model_selection import train_test_split

	require_target(data, target_column)
	drop_columns = [target_column, *[column for column in label_columns if column in data.columns]]
	features = data.drop(columns=drop_columns)
	target = data[target_column]
	return train_test_split(
		features,
		target,
		test_size=0.2,
		random_state=random_state,
		stratify=target,
	)