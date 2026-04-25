import mlflow
import pandas as pd
from pathlib import Path

# Указываем путь к mlruns
MLFLOW_PATH = Path(__file__).parent.parent / 'src' / 'mlruns'
mlflow.set_tracking_uri(f'file://{MLFLOW_PATH.absolute()}')

exp = mlflow.get_experiment_by_name('Random_Forest_Classifier')

runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

all_results = []

for _, run in runs.iterrows():
    result = {
        'run_id': run['run_id'],
        'run_name': run.get('tags.mlflow.runName', ''),
    }

    param_cols = [col for col in run.index if col.startswith('params.')]

    for col in param_cols:
        param_name = col.replace('params.', '')
        result[param_name] = run[col]

    metrics = [
        'cohen_kappa',
        'f1_score',
        'matthews_corrcoef',
        'precision',
        'recall',
        'roc_auc_score'
    ]

    for metric in metrics:
        metric_col = f'metrics.{metric}'
        result[metric] = run[metric_col] if metric_col in run else None

    all_results.append(result)

df = pd.DataFrame(all_results)
df.to_csv('hyperparameters_n_results.csv', index=False)
