from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_nsl_kdd
from src.preprocessing import add_binary_target, split_dataset
from src.ml_model import train_decision_tree
from src.dl_model import train_mlp
from src.compare_models import evaluate_model, save_comparison


def main() -> None:
    data_path = ROOT / 'data' / 'KDDTrain+.txt'
    data = load_nsl_kdd(data_path)
    processed = add_binary_target(data)
    X_train, X_test, y_train, y_test = split_dataset(
        processed,
        target_column='behavior',
        label_columns=('label', 'difficulty_level'),
    )

    print(f'DATASET_SHAPE={data.shape}')
    print(f'CLASS_COUNTS={processed["behavior"].value_counts().to_dict()}')

    dt_model, dt_time = train_decision_tree(
        X_train,
        y_train,
        model_path=ROOT / 'results' / 'models' / 'decision_tree_model.pkl',
    )
    dt_record = evaluate_model(
        dt_model,
        X_test,
        y_test,
        'Decision Tree',
        dt_time,
        ROOT / 'results' / 'metrics',
        ROOT / 'results' / 'plots',
    )
    print(f'DECISION_TREE={dt_record}')

    mlp_model, mlp_time = train_mlp(
        X_train,
        y_train,
        model_path=ROOT / 'results' / 'models' / 'mlp_model.pkl',
    )
    mlp_record = evaluate_model(
        mlp_model,
        X_test,
        y_test,
        'MLP',
        mlp_time,
        ROOT / 'results' / 'metrics',
        ROOT / 'results' / 'plots',
    )
    print(f'MLP={mlp_record}')

    table = save_comparison(
        [dt_record, mlp_record],
        ROOT / 'results' / 'metrics',
        ROOT / 'results' / 'plots',
    )
    print(table.to_dict(orient='records'))
    print('MODELS=', sorted(p.name for p in (ROOT / 'results' / 'models').iterdir()))


if __name__ == '__main__':
    main()
