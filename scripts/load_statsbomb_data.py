from statsbombpy import sb
import pandas as pd
import os

def list_available_competitions():
    print("Fetching available competitions...")
    comps = sb.competitions()
    print(f"\nFound {len(comps)} competition-seasons")
    print("\nPreview:")
    print(comps.head(20))
    return comps

def get_la_liga_matches(seasons_to_fetch=5):
    
    print(f"\nFetching La Liga matches for last {seasons_to_fetch} seasons...")
    
    all_matches = []
    
    seasons = [90, 42, 4, 1, 2]
    season_names = ['2020/2021', '2019/2020', '2018/2019', '2017/2018', '2016/2017']
    
    for season_id, season_name in zip(seasons[:seasons_to_fetch], season_names[:seasons_to_fetch]):
        try:
            print(f"  Fetching {season_name}...", end=' ')
            matches = sb.matches(competition_id=11, season_id=season_id)
            matches['season_name'] = season_name
            all_matches.append(matches)
            print(f"✓ {len(matches)} matches")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if all_matches:
        df = pd.concat(all_matches, ignore_index=True)
        
        os.makedirs('data/statsbomb', exist_ok=True)
        filename = 'data/statsbomb/la_liga_matches.csv'
        df.to_csv(filename, index=False)
        print(f"\n✓ Saved {len(df)} matches to {filename}")
        print(f"\nColumns available: {list(df.columns)}")
        print(f"\nSample data:")
        print(df[['match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'season_name']].head(10))
        return df
    
    return None

def get_match_events(match_id):
    
    print(f"\nFetching events for match {match_id}...")
    events = sb.events(match_id=match_id)
    print(f"✓ {len(events)} events")
    print(f"\nEvent types: {events['type'].unique()}")
    print(f"\nSample events:")
    print(events[['minute', 'team', 'player', 'type', 'location']].head(10))
    return events

def main():
    
    print("="*60)
    print("STATSBOMB DATA LOADER")
    print("="*60)
    
    comps = list_available_competitions()
    
    os.makedirs('data/statsbomb', exist_ok=True)
    comps.to_csv('data/statsbomb/all_competitions.csv', index=False)
    print("\n✓ Saved competitions list to data/statsbomb/all_competitions.csv")
    
    print("\n" + "="*60)
    matches = get_la_liga_matches(seasons_to_fetch=5)
    
    # 3. Optional: Show event data structure (very detailed, use sparingly)
    # if matches is not None and len(matches) > 0:
    #     print("\n" + "="*60)
    #     print("SAMPLE EVENT DATA (one match):")
    #     print("="*60)
    #     sample_match_id = matches.iloc[0]['match_id']
    #     get_match_events(sample_match_id)
    
    print("\n" + "="*60)
    print("✓ DONE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check data/statsbomb/ folder")
    print("2. Open notebooks/01_data_exploration.ipynb")
    print("3. Load the CSV and start exploring!")
    print("\nNote: Event data is VERY detailed but huge.")
    print("For match prediction, the matches CSV is usually enough.")

if __name__ == "__main__":
    main()
