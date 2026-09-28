"""Evaluation and comparison helpers for the two classifiers."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def comparison_table(records: list[dict[str, object]]) -> pd.DataFrame:
	"""Create a comparison table from metrics produced by real model runs."""
	return pd.DataFrame(records, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "Training Time"])


def evaluate_model(
	model: object,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	model_name: str,
	training_time: float,
	metrics_dir: str | Path,
	plots_dir: str | Path,
) -> dict[str, object]:
	"""Calculate actual test metrics and save a confusion matrix."""
	predictions = model.predict(X_test)
	record = {
		"Model": model_name,
		"Accuracy": accuracy_score(y_test, predictions),
		"Precision": precision_score(y_test, predictions, pos_label="Attack"),
		"Recall": recall_score(y_test, predictions, pos_label="Attack"),
		"F1-score": f1_score(y_test, predictions, pos_label="Attack"),
		"Training Time": training_time,
	}
	metrics_path = Path(metrics_dir)
	plots_path = Path(plots_dir)
	metrics_path.mkdir(parents=True, exist_ok=True)
	plots_path.mkdir(parents=True, exist_ok=True)
	confusion = confusion_matrix(y_test, predictions, labels=["Normal", "Attack"])
	plt.figure(figsize=(5, 4))
	sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
	plt.title(f"{model_name} Confusion Matrix")
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.tight_layout()
	plt.savefig(plots_path / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150)
	plt.close()
	return record


def save_comparison(records: list[dict[str, object]], metrics_dir: str | Path, plots_dir: str | Path) -> pd.DataFrame:
	"""Save the actual comparison table and a metric chart."""
	table = comparison_table(records)
	metrics_path = Path(metrics_dir)
	plots_path = Path(plots_dir)
	metrics_path.mkdir(parents=True, exist_ok=True)
	plots_path.mkdir(parents=True, exist_ok=True)
	table.to_csv(metrics_path / "model_comparison.csv", index=False)
	long_table = table.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"], var_name="Metric", value_name="Value")
	plt.figure(figsize=(9, 5))
	sns.barplot(data=long_table, x="Metric", y="Value", hue="Model")
	plt.ylim(0, 1)
	plt.title("Decision Tree vs MLP Performance")
	plt.tight_layout()
	plt.savefig(plots_path / "model_comparison.png", dpi=150)
	plt.close()
	return table