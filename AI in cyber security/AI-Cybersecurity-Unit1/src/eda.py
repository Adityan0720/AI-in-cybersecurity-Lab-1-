"""Reusable, dataset-agnostic exploratory analysis helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def inspect_dataset(data: pd.DataFrame) -> dict[str, object]:
	"""Return the main inspection outputs for display in a notebook."""
	return {
		"shape": data.shape,
		"columns": data.columns.tolist(),
		"dtypes": data.dtypes,
		"missing_values": data.isna().sum(),
		"duplicate_rows": int(data.duplicated().sum()),
		"statistics": data.describe(include="all").transpose(),
	}


def save_class_distribution(data: pd.DataFrame, target_column: str, output_path: str | Path) -> None:
	"""Save a simple class-frequency plot using the inspected target column."""
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(8, 5))
	sns.countplot(data=data, x=target_column)
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()
	plt.savefig(output, dpi=150)
	plt.close()


def save_attack_type_distribution(data: pd.DataFrame, output_path: str | Path) -> None:
	"""Save frequencies of the original NSL-KDD attack labels."""
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	counts = data["label"].value_counts().sort_values(ascending=True)
	plt.figure(figsize=(9, 8))
	counts.plot(kind="barh", color="#2563eb")
	plt.title("NSL-KDD Event Label Frequency")
	plt.xlabel("Number of records")
	plt.ylabel("Original label")
	plt.tight_layout()
	plt.savefig(output, dpi=150)
	plt.close()


def save_feature_distributions(
	data: pd.DataFrame,
	feature_names: list[str],
	output_path: str | Path,
) -> None:
	"""Save histograms for selected numerical cybersecurity features."""
	available = [feature for feature in feature_names if feature in data.columns]
	if not available:
		raise ValueError("None of the requested features exist in the dataset")
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(len(available), 1, figsize=(9, 3 * len(available)))
	if len(available) == 1:
		axes = [axes]
	for axis, feature in zip(axes, available):
		sns.histplot(data=data, x=feature, bins=40, ax=axis, color="#0f766e")
		axis.set_title(f"Distribution of {feature}")
	fig.tight_layout()
	fig.savefig(output, dpi=150)
	plt.close(fig)


def save_correlation_heatmap(
	data: pd.DataFrame,
	feature_names: list[str],
	output_path: str | Path,
) -> None:
	"""Save a correlation heatmap for selected numerical features."""
	available = [feature for feature in feature_names if feature in data.columns]
	if len(available) < 2:
		raise ValueError("At least two numerical features are required")
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(10, 8))
	sns.heatmap(data[available].corr(), cmap="vlag", center=0, annot=False)
	plt.title("Correlation Between Selected Numerical Features")
	plt.tight_layout()
	plt.savefig(output, dpi=150)
	plt.close()