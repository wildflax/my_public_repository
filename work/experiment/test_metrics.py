import numpy as np
import pandas as pd
import pytest

from function import MetricsBase


def test_event_lengths_empty():
    result = MetricsBase.event_lengths([])
    assert result == {"min": 0.0, "median": 0.0}


def test_event_lengths_non_empty():
    events = [
        ("2024-01-01 00:00:00", "2024-01-01 00:10:00"),
        ("2024-01-01 01:00:00", "2024-01-01 01:20:00"),
        ("2024-01-01 02:00:00", "2024-01-01 02:30:00"),
    ]
    result = MetricsBase.event_lengths(events)

    assert result["min"] == 10.0
    assert result["median"] == 20.0


def test_event_params_clipping():
    gap_fill, min_event, event_interval, window_size = MetricsBase.event_params(1000, 1000)

    assert gap_fill == 15
    assert min_event == 120
    assert event_interval == 300
    assert window_size == 120


def test_binary_to_events_basic():
    idx = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 00:05:00",
        "2024-01-01 00:10:00",
        "2024-01-01 01:00:00",
        "2024-01-01 01:05:00",
    ])
    s = pd.Series([1, 1, 1, 1, 1], index=idx)

    events = MetricsBase.binary_to_events(s, gap_minutes=10, min_len_minutes=5)

    assert len(events) == 2
    assert events[0][0] == pd.Timestamp("2024-01-01 00:00:00")
    assert events[0][1] == pd.Timestamp("2024-01-01 00:10:00")
    assert events[1][0] == pd.Timestamp("2024-01-01 01:00:00")
    assert events[1][1] == pd.Timestamp("2024-01-01 01:05:00")


def test_binary_to_events_filters_short():
    idx = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 00:01:00",
    ])
    s = pd.Series([1, 1], index=idx)

    events = MetricsBase.binary_to_events(s, gap_minutes=10, min_len_minutes=5)
    assert events == []


def test_precision_at_k():
    y_true = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.1, 0.0]

    result = MetricsBase.precision_at_k(y_true, scores, 2)
    assert result == 0.5


def test_recall_at_k():
    y_true = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.1, 0.0]

    result = MetricsBase.recall_at_k(y_true, scores, 2)
    assert result == 0.5


def test_precision_at_k_invalid_k():
    with pytest.raises(ValueError, match="k must be > 0"):
        MetricsBase.precision_at_k([1, 0], [0.1, 0.2], 0)


def test_recall_at_k_invalid_k():
    with pytest.raises(ValueError, match="k must be > 0"):
        MetricsBase.recall_at_k([1, 0], [0.1, 0.2], 0)


def test_event_recall():
    true_events = [
        (pd.Timestamp("2024-01-01 10:00:00"), pd.Timestamp("2024-01-01 10:30:00")),
        (pd.Timestamp("2024-01-01 12:00:00"), pd.Timestamp("2024-01-01 12:30:00")),
    ]
    pred_events = [
        (pd.Timestamp("2024-01-01 10:10:00"), pd.Timestamp("2024-01-01 10:20:00")),
    ]

    result = MetricsBase.event_recall(true_events, pred_events, max_min=30)
    assert result == 0.5


def test_event_precision():
    true_events = [
        (pd.Timestamp("2024-01-01 10:00:00"), pd.Timestamp("2024-01-01 10:30:00")),
    ]
    pred_events = [
        (pd.Timestamp("2024-01-01 10:05:00"), pd.Timestamp("2024-01-01 10:20:00")),
        (pd.Timestamp("2024-01-01 15:00:00"), pd.Timestamp("2024-01-01 15:10:00")),
    ]

    result = MetricsBase.event_precision(true_events, pred_events, max_min=30)
    assert result == 0.5


def test_time_to_detect_mean():
    true_events = [
        (pd.Timestamp("2024-01-01 10:00:00"), pd.Timestamp("2024-01-01 10:30:00")),
    ]
    pred_events = [
        (pd.Timestamp("2024-01-01 10:10:00"), pd.Timestamp("2024-01-01 10:20:00")),
    ]

    result = MetricsBase.time_to_detect(true_events, pred_events, reduction="mean", unit="minutes", max_min=30)
    assert result == 10.0


def test_time_to_detect_absolute():
    true_events = [
        (pd.Timestamp("2024-01-01 10:00:00"), pd.Timestamp("2024-01-01 10:30:00")),
    ]
    pred_events = [
        (pd.Timestamp("2024-01-01 09:50:00"), pd.Timestamp("2024-01-01 09:55:00")),
    ]

    result = MetricsBase.time_to_detect(
        true_events,
        pred_events,
        reduction="mean",
        unit="minutes",
        max_min=30,
        absolute=True,
    )
    assert result == 10.0


def test_time_to_detect_invalid_reduction():
    true_events = [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 00:10:00"))]
    pred_events = [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 00:10:00"))]

    with pytest.raises(ValueError, match="reduction must be mean, median, or list"):
        MetricsBase.time_to_detect(true_events, pred_events, reduction="sum")


def test_convert_unit_invalid():
    with pytest.raises(ValueError, match="unit must be seconds, minutes, or hours"):
        MetricsBase._convert_unit(pd.Timedelta(minutes=1), "days")


def test_alerts_per_day():
    timestamps = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 12:00:00",
        "2024-01-02 00:00:00",
    ])
    pred_events = [
        (pd.Timestamp("2024-01-01 01:00:00"), pd.Timestamp("2024-01-01 02:00:00")),
        (pd.Timestamp("2024-01-01 13:00:00"), pd.Timestamp("2024-01-01 14:00:00")),
    ]

    result = MetricsBase.alerts_per_day(pred_events, timestamps, gap_threshold="1D")
    assert result == 2.0


def test_make_json_serializable():
    obj = {
        "a": np.int64(5),
        "b": np.float64(1.5),
        "c": pd.Timestamp("2024-01-01"),
        "d": pd.Timedelta(minutes=5),
        "e": [np.int64(1), np.float64(2.0)],
    }

    result = MetricsBase.make_json_serializable(obj)

    assert result["a"] == 5
    assert result["b"] == 1.5
    assert result["c"] == "2024-01-01T00:00:00"
    assert result["d"] == "0 days 00:05:00"
    assert result["e"] == [1, 2.0]


def test_get_anomaly_lengths():
    df = pd.DataFrame({"target": [0, 1, 1, 0, 1, 1, 1, 0]})
    min_len, median_len = MetricsBase.get_anomaly_lengths(df)

    assert min_len == 2
    assert median_len == 2.5


def test_get_anomaly_lengths_no_anomalies():
    df = pd.DataFrame({"target": [0, 0, 0]})
    min_len, median_len = MetricsBase.get_anomaly_lengths(df, default_value=0)

    assert min_len == 0
    assert median_len == 0