"""Helpers for loading and inspecting the NSL-KDD tabular dataset."""

from pathlib import Path

import pandas as pd

NSL_KDD_COLUMNS = [
	"duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
	"land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
	"logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
	"num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
	"is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
	"srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
	"diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
	"dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
	"dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
	"dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level",
]


def load_csv(file_path: str | Path, column_names: list[str] | None = None) -> pd.DataFrame:
	"""Load a CSV file and raise clear errors for common input problems."""
	path = Path(file_path)
	if not path.is_file():
		raise FileNotFoundError(f"Dataset not found: {path}")

	try:
		header = None if column_names else "infer"
		data = pd.read_csv(path, header=header, names=column_names)
	except pd.errors.ParserError as error:
		raise ValueError(f"Invalid CSV file: {path}") from error

	if data.empty:
		raise ValueError(f"Dataset is empty: {path}")
	return data


def load_nsl_kdd(file_path: str | Path) -> pd.DataFrame:
	"""Load a headerless NSL-KDD file using its documented 43-column schema."""
	return load_csv(file_path, column_names=NSL_KDD_COLUMNS)


def inspect_nsl_kdd(data: pd.DataFrame) -> dict[str, object]:
	"""Collect factual inspection values without changing the dataset."""
	if list(data.columns) != NSL_KDD_COLUMNS:
		raise ValueError("DataFrame does not use the standard NSL-KDD columns")

	return {
		"shape": data.shape,
		"columns": data.columns.tolist(),
		"dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
		"missing_values": {column: int(value) for column, value in data.isna().sum().items()},
		"duplicate_rows": int(data.duplicated().sum()),
		"target_column": "label",
		"target_unique_values": sorted(data["label"].unique().tolist()),
		"target_distribution": data["label"].value_counts().to_dict(),
		"numeric_features": data.select_dtypes(include="number").columns.tolist(),
		"categorical_features": data.select_dtypes(exclude="number").columns.tolist(),
	}