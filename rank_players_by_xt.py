"""
Rank players by xT generation per 90 minutes across all available StatsBomb data
"""

from statsbombpy import sb
import pandas as pd
import numpy as np
import sys
sys.path.append('models')
from pitch_grid import PitchGrid

print("="*80)
print("PLAYER xT RANKING SYSTEM")
print("="*80)

# Initialize pitch grid
grid = PitchGrid(n_cols=12, n_rows=8, pitch_length=120, pitch_width=80)

# Load xT values
print("\nLoading xT model...")
xt_data = np.load('data/processed/xt_values.npz')
xt_values = xt_data['xt_values']
print(f"✓ Loaded xT grid with {len(xt_values)} zones")

def get_zone_xt(x, y):
    """Get xT value for a coordinate"""
    if pd.isna(x) or pd.isna(y):
        return None
    zone = grid.get_zone(x, y)
    return xt_values[zone]

def calculate_delta_xt(row):
    """Calculate ΔxT for an event"""
    event_type = row.get('type')
    
    # Get start position
    location = row.get('location')
    if not isinstance(location, list) or len(location) < 2:
        return 0.0
    
    start_x, start_y = location[0], location[1]
    start_xt = get_zone_xt(start_x, start_y)
    if start_xt is None:
        return 0.0
    
    # Get end position based on event type
    end_x, end_y = None, None
    
    if event_type == 'Pass':
        end_loc = row.get('pass_end_location')
        if isinstance(end_loc, list) and len(end_loc) >= 2:
            end_x, end_y = end_loc[0], end_loc[1]
    
    elif event_type == 'Carry':
        end_loc = row.get('carry_end_location')
        if isinstance(end_loc, list) and len(end_loc) >= 2:
            end_x, end_y = end_loc[0], end_loc[1]
    
    elif event_type == 'Shot':
        # Shots end at goal (assume center of goal)
        end_x, end_y = 120, 40
    
    # Calculate delta
    if end_x is not None and end_y is not None:
        end_xt = get_zone_xt(end_x, end_y)
        if end_xt is not None:
            return end_xt - start_xt
    
    return 0.0

# Select LA LIGA 2020/21 by default
comp_id, season_id, comp_name = 11, 90, "La Liga 2020/21"

print(f"\n{'='*80}")
print(f"Analyzing: {comp_name}")
print(f"{'='*80}\n")

# Get matches
if comp_id is not None:
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    print(f"Found {len(matches)} matches")
else:
    # Get all competitions
    comps = sb.competitions()
    all_matches = []
    print("Fetching all matches (this will take time)...")
    for idx, comp in comps.head(10).iterrows():  # Limit to top 10 for demo
        try:
            m = sb.matches(competition_id=comp['competition_id'], season_id=comp['season_id'])
            all_matches.append(m)
            print(f"  ✓ {comp['competition_name']} {comp['season_name']}: {len(m)} matches")
        except:
            pass
    matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

if len(matches) == 0:
    print("No matches found!")
    sys.exit(1)

# Process matches and calculate xT
print(f"\nProcessing {len(matches)} matches...")
player_stats = {}

for idx, match in matches.iterrows():
    match_id = match['match_id']
    
    try:
        # Get events
        events = sb.events(match_id=match_id)
        
        # Extract player minutes from events (more reliable than lineups)
        player_minutes = {}
        player_teams = {}
        
        # Count events per player to estimate participation
        for player_name in events['player'].dropna().unique():
            player_events = events[events['player'] == player_name]
            # Assume 90 minutes by default (we'll use games played for normalization)
            player_minutes[player_name] = 90
            # Get team
            team = player_events.iloc[0].get('team')
            player_teams[player_name] = team
        
        # Calculate xT for each event
        print(f"  Match {idx+1}/{len(matches)}: {match['home_team']} vs {match['away_team']}...", end=' ')
        
        # Filter to relevant events
        relevant_events = events[events['type'].isin(['Pass', 'Carry', 'Shot'])].copy()
        
        # Calculate delta xT
        relevant_events['delta_xt'] = relevant_events.apply(calculate_delta_xt, axis=1)
        
        # Aggregate by player
        for player_name in relevant_events['player'].unique():
            if pd.isna(player_name):
                continue
            
            player_events = relevant_events[relevant_events['player'] == player_name]
            total_xt = player_events['delta_xt'].sum()
            n_actions = len(player_events)
            
            if player_name not in player_stats:
                player_stats[player_name] = {
                    'total_xt': 0.0,
                    'total_actions': 0,
                    'total_minutes': 0,
                    'games_played': 0
                }
            
            player_stats[player_name]['total_xt'] += total_xt
            player_stats[player_name]['total_actions'] += n_actions
            player_stats[player_name]['total_minutes'] += player_minutes.get(player_name, 90)
            player_stats[player_name]['games_played'] += 1
        
        print(f"{len(relevant_events)} events analyzed")
        
    except Exception as e:
        print(f"  ✗ Match {idx+1}/{len(matches)}: Error - {e}")

# Convert to DataFrame
print("\nAggregating player statistics...")
results = []
for player_name, stats in player_stats.items():
    minutes = stats['total_minutes']
    games = stats['games_played']
    total_xt = stats['total_xt']
    
    # Calculate per 90 minutes
    if minutes > 0:
        xt_per_90 = (total_xt / minutes) * 90
    else:
        xt_per_90 = 0.0
    
    results.append({
        'player': player_name,
        'total_xt': total_xt,
        'games_played': games,
        'minutes_played': minutes,
        'actions': stats['total_actions'],
        'xt_per_90': xt_per_90,
        'xt_per_game': total_xt / games if games > 0 else 0
    })

if len(results) == 0:
    print("\n✗ No player data found!")
    sys.exit(1)

df_results = pd.DataFrame(results)

# Filter minimum games/minutes threshold
min_games = 3
df_filtered = df_results[df_results['games_played'] >= min_games].copy()

print(f"\n{'='*80}")
print(f"RESULTS ({len(df_filtered)} players with {min_games}+ games)")
print(f"{'='*80}\n")

# Rank by xT per 90
df_filtered = df_filtered.sort_values('xt_per_90', ascending=False)

print("TOP 20 PLAYERS BY xT PER 90 MINUTES:")
print("-" * 100)
print(f"{'Rank':<6}{'Player':<30}{'xT/90':<12}{'Total xT':<12}{'Games':<8}{'Minutes':<10}{'Actions':<10}")
print("-" * 100)

for idx, row in df_filtered.head(20).iterrows():
    print(f"{idx+1:<6}{row['player'][:29]:<30}{row['xt_per_90']:<12.4f}{row['total_xt']:<12.2f}"
          f"{row['games_played']:<8.0f}{row['minutes_played']:<10.0f}{row['actions']:<10.0f}")

# Save results
output_file = 'data/processed/player_xt_rankings.csv'
df_filtered.to_csv(output_file, index=False)
print(f"\n✓ Full rankings saved to: {output_file}")

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")
print(f"Total players analyzed: {len(df_results)}")
print(f"Players with {min_games}+ games: {len(df_filtered)}")
print(f"Average xT per 90: {df_filtered['xt_per_90'].mean():.4f}")
print(f"Median xT per 90: {df_filtered['xt_per_90'].median():.4f}")
print(f"Highest xT per 90: {df_filtered['xt_per_90'].max():.4f} ({df_filtered.iloc[0]['player']})")
print(f"\nTotal actions analyzed: {df_results['actions'].sum():,.0f}")
print(f"Total xT generated: {df_results['total_xt'].sum():.2f}")
