from __future__ import annotations
import pandas as pd

def time_split(df: pd.DataFrame, date_col: str, train_end: str, valid_end: str):
    d = pd.to_datetime(df[date_col])
    train_mask = d <= pd.to_datetime(train_end)
    valid_mask = (d > pd.to_datetime(train_end)) & (d <= pd.to_datetime(valid_end))
    test_mask  = d > pd.to_datetime(valid_end)
    return df[train_mask], df[valid_mask], df[test_mask]