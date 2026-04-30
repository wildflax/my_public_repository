import json
import pandas as pd

from function import TimeSeriesExperimentRunner, TabularExperimentRunner


def build_kwargs():
    return {
        "model_name": "model_1",
        "pr_auc": 90.0,
        "f1_at_fpr_1": 80.0,
        "recall": 70.0,
        "precision": 60.0,
        "precision_at_100": 50.0,
        "recall_at_100": 40.0,
        "event_recall": 30.0,
        "event_precision": 20.0,
        "f1_event": 10.0,
        "ttd_mean": 5.0,
        "ttd_median": 4.0,
        "alerts": 3.0,
        "n_true_events": 2,
        "n_pred_events": 1,
        "train_time_sec": 12.345,
        "inference_time_sec": 0.987,
        "model_params": {"a": 1},
        "threshold": "0.95, +, 1",
        "f1_anom": 55.0,
    }


def test_timeseries_add_results():
    runner = TimeSeriesExperimentRunner()
    df = pd.DataFrame()

    result = runner.add_results(df, **build_kwargs())

    assert "model_1" in result.index
    assert result.loc["model_1", "PR-AUC"] == 90.0
    assert result.loc["model_1", "train time sec"] == 12.35

    params = result.loc["model_1", "model params"]
    assert isinstance(params, str)
    assert json.loads(params) == {"a": 1}


def test_tabular_add_results():
    runner = TabularExperimentRunner()
    df = pd.DataFrame()

    result = runner.add_results(df, **build_kwargs())

    assert "model_1" in result.index
    assert result.loc["model_1", "PR-AUC"] == 90.0
    assert result.loc["model_1", "model params"] == {"a": 1}