import pandas as pd


def assign_possession_ids(events_df):
    
    if 'possession' in events_df.columns:
        events_df['possession_id'] = events_df['possession']
    else:
        raise ValueError("events_df must contain 'possession' column from StatsBomb data")
    
    return events_df