import pandas as pd 

def promote_model(models: list) -> str:
    df = pd.DataFrame(models)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(
    by=["accuracy", "latency", "timestamp"],
        ascending=[False, True, False],  # accuracy ↓, latency ↑, timestamp ↓
        inplace=True,
        ignore_index=True  # reset index after sorting
        )
    return df.iloc[0]['name']