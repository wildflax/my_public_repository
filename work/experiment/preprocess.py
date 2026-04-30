import numpy as np
import pandas as pd

def preprocess_supervised(
    data,
    freq='1min',
    max_interp_gap=5,
    add_val=False,
    add_miss=True,
    exclude_drop=None,
    fit_dummy_columns=None,
    return_dummy_columns=False,
    handle_outliers=True,
    outlier_method='iqr',
    outlier_k=10,
    outlier_q_low=0.01,
    outlier_q_high=0.99,
    outlier_exclude=None,
    print_outliers=True
):
    if exclude_drop is None:
        exclude_drop = []
    if outlier_exclude is None:
        outlier_exclude = []

    df = data.copy()

    # приведение строковых числовых столбцов к numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            cleaned = df[col].astype(str).str.strip().str.replace(',', '.', regex=False)
            converted = pd.to_numeric(cleaned, errors='coerce')

            non_na_original = df[col].notna().sum()
            non_na_converted = converted.notna().sum()

            if non_na_original > 0 and non_na_converted / non_na_original >= 0.95:
                df[col] = converted

    # -------------------------------
    # ОБРАБОТКА ВЫБРОСОВ ДО АГРЕГАЦИИ
    # -------------------------------
    outlier_log = []

    if handle_outliers and outlier_method is not None:
        if outlier_method not in ['iqr', 'quantile']:
            raise ValueError("outlier_method must be 'iqr', 'quantile' or None")
        raw_num_cols = df.select_dtypes(include=['number']).columns.tolist()

        for col in raw_num_cols:
            if col in outlier_exclude:
                continue

            s = df[col]
            non_na = s.dropna()

            if non_na.empty:
                continue

            # бинарные и низкокардинальные числовые не трогаем
            if non_na.nunique() <= 10:
                continue

            if outlier_method == 'iqr':
                q1 = non_na.quantile(0.25)
                q3 = non_na.quantile(0.75)
                iqr = q3 - q1

                if iqr == 0:
                    continue

                lower = q1 - outlier_k * iqr
                upper = q3 + outlier_k * iqr

            elif outlier_method == 'quantile':
                lower = non_na.quantile(outlier_q_low)
                upper = non_na.quantile(outlier_q_high)

            else:
                raise ValueError("outlier_method must be 'iqr', 'quantile' or None")

            mask_low = s < lower
            mask_high = s > upper
            mask_outliers = mask_low | mask_high

            if mask_outliers.any():
                changed = df.loc[mask_outliers, col].copy()

                for idx, val in changed.items():
                    new_val = lower if val < lower else upper
                    outlier_log.append({
                        'column': col,
                        'index': idx,
                        'old_value': val,
                        'new_value': new_val
                    })

                df[col] = s.clip(lower=lower, upper=upper)

    if print_outliers:
        if outlier_log:
            print("=== Изменённые выбросы ===")
            outlier_df = pd.DataFrame(outlier_log)
            print(outlier_df)
        else:
            print("Выбросы не найдены")
    if exclude_drop is None:
        exclude_drop = []
    if outlier_exclude is None:
        outlier_exclude = []

    # округление до частоты
    df.index = df.index.floor(freq)

    # разделение типов столбцов
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    # агрегирование дубликатов по времени
    agg_dict = {col: 'mean' for col in num_cols}
    agg_dict.update({col: 'last' for col in cat_cols})
    df = df.groupby(df.index).agg(agg_dict)

    # приведение к равномерной частоте
    df = df.asfreq(freq)

    # маска пропусков
    original_nan_mask = df.isna().copy()

    # максимальный допустимый размер разрыва в шагах
    step_limit = int(pd.Timedelta(minutes=max_interp_gap) / pd.Timedelta(freq))
    step_limit = max(step_limit, 1)

    # удаляем полностью пустые колонки
    cols_to_drop = []
    for col in df.columns:
        if df[col].dropna().empty and col not in exclude_drop:
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # пересчитываем после drop
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    # заполнение числовых колонок
    for col in num_cols:
        non_na = df[col].dropna()
        n_unique = non_na.nunique()

        if non_na.empty:
            continue

        # непрерывные числовые признаки
        if n_unique > 10:
            df[col] = df[col].interpolate(method='linear', limit=step_limit)

            # если после интерполяции остались пропуски — медиана
            median_value = non_na.median()
            df[col] = df[col].fillna(median_value)

        # дискретные / бинарные / низкокардинальные
        else:
            local_limit = max(step_limit // 2, 1)
            df[col] = df[col].ffill(limit=local_limit).bfill(limit=local_limit)

            # если ещё остались пропуски — заполняем 0
            df[col] = df[col].fillna(0)

    # заполнение категориальных колонок
    for col in cat_cols:
        df[col] = df[col].ffill(limit=step_limit).bfill(limit=step_limit)
        mode_series = df[col].dropna().mode()
        if not mode_series.empty:
            df[col] = df[col].fillna(mode_series.iloc[0])

    # признаки пропусков
    if add_miss:
        bad_rows = original_nan_mask.reindex(df.index).mean(axis=1) > 0.5
        df['miss_all'] = bad_rows.astype(int)

        miss_cols = original_nan_mask.reindex(df.index).astype(int)
        miss_cols.columns = ['miss_' + c for c in miss_cols.columns]

        existing_miss_cols = [
            c for c in miss_cols.columns
            if c.replace('miss_', '', 1) in df.columns
        ]
        miss_cols = miss_cols[existing_miss_cols]

        df = pd.concat([df, miss_cols], axis=1)

    # one-hot encoding категориальных
    cat_cols_after_fill = df.select_dtypes(exclude=['number']).columns.tolist()
    if len(cat_cols_after_fill) > 0:
        df = pd.get_dummies(df, columns=cat_cols_after_fill, dummy_na=False)

    # выравнивание колонок по train
    if fit_dummy_columns is not None:
        df = df.reindex(columns=fit_dummy_columns, fill_value=0)

    dummy_columns = df.columns.tolist()

    # train / val split
    if add_val:
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        val = df.iloc[split_idx:]

        if return_dummy_columns:
            return train, val, dummy_columns
        return train, val

    if return_dummy_columns:
        return df, dummy_columns

    if handle_outliers:
        if outlier_log:
            print("Изменённые выбросы")
            print(pd.DataFrame(outlier_log))

    return df