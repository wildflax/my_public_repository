import pandas as pd

from function import ExperimentBase


class DummyRunner(ExperimentBase):
    def prepare_scaled_data(self, *args, **kwargs):
        raise NotImplementedError

    def fit_model(self, *args, **kwargs):
        raise NotImplementedError

    def score_model(self, *args, **kwargs):
        raise NotImplementedError

    def add_results(self, df, **kwargs):
        return df


def test_compute_metrics_basic():
    runner = DummyRunner()

    idx = pd.date_range("2024-01-01", periods=6, freq="1h")
    y_true = pd.Series([0, 1, 1, 0, 0, 1], index=idx)

    test_scores = pd.Series([0.1, 0.9, 0.8, 0.2, 0.1, 0.95], index=idx)
    val_scores = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=pd.RangeIndex(5))

    metrics = runner.compute_metrics(
        y_true=y_true,
        test_scores_series=test_scores,
        val_scores_series=val_scores,
        q=0.8,
        alpha=1.0,
        gap_fill=60,
        min_event=0,
        event_interval=180,
    )

    assert "threshold_value" in metrics
    assert "pr_auc_val" in metrics
    assert "f1_val" in metrics
    assert "pred_events" in metrics
    assert "true_events" in metrics
    assert len(metrics["y_pred"]) == len(y_true)
    assert metrics["cm"].shape == (2, 2)


def test_compute_metrics_with_calibration():
    runner = DummyRunner()

    idx = pd.date_range("2024-01-01", periods=5, freq="1h")
    y_true = pd.Series([0, 1, 0, 1, 0], index=idx)
    test_scores = pd.Series([10, 20, 30, 40, 50], index=idx)
    val_scores = pd.Series([5, 15, 25, 35, 45], index=pd.RangeIndex(5))

    metrics = runner.compute_metrics(
        y_true=y_true,
        test_scores_series=test_scores,
        val_scores_series=val_scores,
        q=0.8,
        alpha=1.0,
        gap_fill=60,
        min_event=0,
        event_interval=180,
        use_calibrate=True,
    )

    assert metrics["val_scores_series"].between(0, 1).all()
    assert metrics["test_scores_series"].between(0, 1).all()