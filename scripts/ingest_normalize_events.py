import pandas as pd
import numpy as np
from statsbombpy import sb
import os
import json


def load_competition_matches(competition_id, season_id):
    
    print(f"Loading matches for competition {competition_id}, season {season_id}...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    print(f"✓ Loaded {len(matches)} matches")
    return matches


def load_match_events(match_id):
    
    events = sb.events(match_id=match_id)
    return events


def standardize_coordinates(events_df, match_id):
    
    df = events_df.copy()
    
    df['x'] = df['location'].apply(lambda loc: loc[0] if isinstance(loc, list) and len(loc) >= 2 else np.nan)
    df['y'] = df['location'].apply(lambda loc: loc[1] if isinstance(loc, list) and len(loc) >= 2 else np.nan)
    
    second_half = df['period'] == 2
    
    df.loc[second_half, 'x'] = 120 - df.loc[second_half, 'x']
    df.loc[second_half, 'y'] = 80 - df.loc[second_half, 'y']
    
    home_team = df.iloc[0]['team']
    
    away_events = df['team'] != home_team
    df.loc[away_events, 'x'] = 120 - df.loc[away_events, 'x']
    df.loc[away_events, 'y'] = 80 - df.loc[away_events, 'y']
    
    return df


def flatten_and_attach_metadata(events_df, match_id, home_team, away_team):
    
    df = events_df.copy()
    
    df['match_id'] = match_id
    
    df['team_id'] = df['team']
    
    df['player_id'] = df['player'].fillna('Unknown')
    df['player_name'] = df['player'].fillna('Unknown')
    
    df['minute'] = df['minute'].fillna(0).astype(int)
    df['second'] = df['second'].fillna(0).astype(int)
    df['period'] = df['period'].fillna(1).astype(int)
    
    df['is_home_team'] = df['team'] == home_team
    
    df['event_type'] = df['type']
    
    essential_cols = [
        'match_id', 'team_id', 'team', 'player_id', 'player_name',
        'minute', 'second', 'period', 'is_home_team',
        'event_type', 'possession', 'possession_team',
        'x', 'y', 'location', 'index', 'timestamp'
    ]
    
    existing_cols = [col for col in essential_cols if col in df.columns]
    df = df[existing_cols]
    
    return df


def validate_events(events_df):
    
    issues = []
    
    critical_fields = ['match_id', 'team_id', 'minute', 'second', 'event_type']
    for field in critical_fields:
        if field in events_df.columns:
            missing = events_df[field].isna().sum()
            if missing > 0:
                issues.append(f"Missing {missing} values in {field}")
    
    if 'index' in events_df.columns:
        if not events_df['index'].is_monotonic_increasing:
            issues.append("Events are not in chronological order (index)")
    
    if 'x' in events_df.columns and 'y' in events_df.columns:
        x_out = ((events_df['x'] < 0) | (events_df['x'] > 120)).sum()
        y_out = ((events_df['y'] < 0) | (events_df['y'] > 80)).sum()
        
        if x_out > 0:
            issues.append(f"{x_out} events have x coordinates outside [0, 120]")
        if y_out > 0:
            issues.append(f"{y_out} events have y coordinates outside [0, 80]")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


def load_and_normalize_match(match_id, home_team, away_team):
    
    print(f"\nProcessing match {match_id}: {home_team} vs {away_team}")
    
    events = load_match_events(match_id)
    print(f"  ✓ Loaded {len(events)} events")
    
    events = standardize_coordinates(events, match_id)
    print(f"  ✓ Standardized coordinates")
    
    events = flatten_and_attach_metadata(events, match_id, home_team, away_team)
    print(f"  ✓ Attached metadata")
    
    is_valid, issues = validate_events(events)
    if is_valid:
        print(f"  ✓ Validation passed")
    else:
        print(f"  ⚠ Validation issues:")
        for issue in issues:
            print(f"    - {issue}")
    
    return events


def load_and_normalize_competition(competition_id, season_id, max_matches=None):
    
    matches = load_competition_matches(competition_id, season_id)
    
    if max_matches:
        matches = matches.head(max_matches)
        print(f"\n⚠ Limited to first {max_matches} matches for testing")
    
    all_events = []
    
    for idx, match in matches.iterrows():
        match_id = match['match_id']
        home_team = match['home_team']
        away_team = match['away_team']
        
        try:
            events = load_and_normalize_match(match_id, home_team, away_team)
            all_events.append(events)
        except Exception as e:
            print(f"  ✗ Error processing match {match_id}: {e}")
            continue
    
    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        print(f"\n✓ Total events across all matches: {len(combined)}")
        return combined
    else:
        print("\n✗ No events loaded")
        return pd.DataFrame()


if __name__ == "__main__":
    print("="*80)
    print("STATSBOMB DATA INGESTION & NORMALIZATION")
    print("="*80)
    
    events_df = load_and_normalize_competition(
        competition_id=11,
        season_id=90,
        max_matches=3
    )
    
    if not events_df.empty:
        os.makedirs('data/processed', exist_ok=True)
        output_file = 'data/processed/events_normalized_sample.csv'
        events_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved to {output_file}")
        
        print("\nSample events:")
        print(events_df.head(10))
        
        print("\nEvent types distribution:")
        print(events_df['event_type'].value_counts().head(10))
