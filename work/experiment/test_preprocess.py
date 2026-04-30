import numpy as np
import pandas as pd
import pytest
from preprocess import preprocess_supervised

# замени import на свой модуль
# from your_module import preprocess_supervised


def make_dt_index(n=6, start="2024-01-01 00:00:00", freq="1min"):
    return pd.date_range(start=start, periods=n, freq=freq)


def test_string_numeric_columns_are_converted():
    idx = make_dt_index(4)
    df = pd.DataFrame(
        {
            "num_str": ["1,0", "2,5", "3.0", "4"],
            "cat": ["a", "b", "a", "b"],
        },
        index=idx,
    )

    result = preprocess_supervised(
        df,
        print_outliers=False,
        handle_outliers=False,
    )

    assert "num_str" in result.columns
    assert pd.api.types.is_numeric_dtype(result["num_str"])
    assert np.isclose(result["num_str"].iloc[1], 2.5)


def test_outliers_are_clipped_before_aggregation():
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:10",
            "2024-01-01 00:00:20",
            "2024-01-01 00:01:10",
            "2024-01-01 00:01:20",
        ]
    )

    df = pd.DataFrame(
        {
            "x": [1.0, 1000.0, 2.0, 3.0],
        },
        index=idx,
    )

    result = preprocess_supervised(
        df,
        freq="1min",
        print_outliers=False,
        handle_outliers=True,
        outlier_method="quantile",
        outlier_q_low=0.0,
        outlier_q_high=0.75,
    )

    # После floor + groupby mean первая минута не должна остаться огромной
    assert result.shape[0] == 2
    assert result["x"].iloc[0] < 1000.0


def test_low_cardinality_numeric_columns_are_not_clipped():
    idx = make_dt_index(6)
    df = pd.DataFrame(
        {
            "bin_col": [0, 1, 0, 1, 0, 1],  # low-cardinality
        },
        index=idx,
    )

    result = preprocess_supervised(
        df,
        print_outliers=False,
        handle_outliers=True,
        outlier_method="iqr",
    )

    assert set(result["bin_col"].unique()).issubset({0, 1})


def test_exclude_drop_keeps_all_nan_column():
    idx = make_dt_index(4)
    df = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan, np.nan],
            "x": [1.0, 2.0, 3.0, 4.0],
        },
        index=idx,
    )

    result = preprocess_supervised(
        df,
        print_outliers=False,
        handle_outliers=False,
        exclude_drop=["all_nan"],
    )

    assert "all_nan" in result.columns


def test_missing_flags_are_added():
    idx = make_dt_index(4)
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, np.nan],
            "y": [np.nan, np.nan, 1.0, 2.0],
        },
        index=idx,
    )

    result = preprocess_supervised(
        df,
        print_outliers=False,
        handle_outliers=False,
        add_miss=True,
    )

    assert "miss_all" in result.columns
    assert "miss_x" in result.columns
    assert "miss_y" in result.columns


def test_one_hot_encoding_is_applied():
    idx = make_dt_index(4)
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "state": ["on", "off", "on", "off"],
        },
        index=idx,
    )

    result = preprocess_supervised(
        df,
        print_outliers=False,
        handle_outliers=False,
    )

    state_cols = [c for c in result.columns if c.startswith("state_")]
    assert len(state_cols) >= 2


def test_fit_dummy_columns_reindexes_columns():
    idx = make_dt_index(4)
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "state": ["on", "off", "on", "off"],
        },
        index=idx,
    )

    fit_cols = ["x", "state_on", "state_off", "extra_col"]

    result = preprocess_supervised(
        df,
        print_outliers=False,
        handle_outliers=False,
        fit_dummy_columns=fit_cols,
    )

    assert list(result.columns) == fit_cols
    assert (result["extra_col"] == 0).all()


def test_return_dummy_columns():
    idx = make_dt_index(4)
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "state": ["on", "off", "on", "off"],
        },
        index=idx,
    )

    result, dummy_columns = preprocess_supervised(
        df,
        print_outliers=False,
        handle_outliers=False,
        return_dummy_columns=True,
    )

    assert isinstance(dummy_columns, list)
    assert list(result.columns) == dummy_columns


def test_add_val_returns_train_and_val():
    idx = make_dt_index(10)
    df = pd.DataFrame(
        {
            "x": np.arange(10, dtype=float),
        },
        index=idx,
    )

    train, val = preprocess_supervised(
        df,
        print_outliers=False,
        handle_outliers=False,
        add_val=True,
    )

    assert len(train) == 8
    assert len(val) == 2


def test_add_val_with_dummy_columns():
    idx = make_dt_index(10)
    df = pd.DataFrame(
        {
            "x": np.arange(10, dtype=float),
            "state": ["on", "off"] * 5,
        },
        index=idx,
    )

    train, val, dummy_columns = preprocess_supervised(
        df,
        print_outliers=False,
        handle_outliers=False,
        add_val=True,
        return_dummy_columns=True,
    )

    assert isinstance(dummy_columns, list)
    assert list(train.columns) == dummy_columns
    assert list(val.columns) == dummy_columns


def test_invalid_outlier_method_raises_error():
    idx = make_dt_index(4)
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 100.0],
        },
        index=idx,
    )

    with pytest.raises(ValueError):
        preprocess_supervised(
            df,
            print_outliers=False,
            handle_outliers=True,
            outlier_method="wrong_method",
        )