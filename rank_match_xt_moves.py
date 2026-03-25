"""
Rank top xT moves in a single Barcelona match
"""

from statsbombpy import sb
import pandas as pd
import numpy as np
import sys
sys.path.append('models')
from assign_xt_to_events import xTAssigner


def find_barcelona_matches(competition_id=11, season_id=90):
    """Find all Barcelona matches in a given competition/season"""
    print("Loading matches...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    
    # Filter for Barcelona matches
    barca_matches = matches[
        (matches['home_team'] == 'Barcelona') | 
        (matches['away_team'] == 'Barcelona')
    ].copy()
    
    return barca_matches


def display_match_selection(matches):
    """Display matches for user to choose from"""
    print(f"\n{'='*80}")
    print(f"Found {len(matches)} Barcelona matches")
    print(f"{'='*80}\n")
    
    for idx, row in matches.iterrows():
        home = row['home_team']
        away = row['away_team']
        score = f"{row['home_score']}-{row['away_score']}"
        date = row['match_date']
        match_id = row['match_id']
        
        print(f"{idx+1}. {date} | {home} vs {away} ({score}) | ID: {match_id}")
    
    return matches


def analyze_match_xt(match_id):
    """Analyze xT for a single match and return ranked moves"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing Match ID: {match_id}")
    print(f"{'='*80}\n")
    
    # Load xT assigner
    print("Loading xT model...")
    assigner = xTAssigner()
    
    # Get events with xT values
    print("Processing match events...")
    events_with_xt = assigner.assign_xt_to_match(match_id)
    
    # Filter for actions that create threat (positive xT)
    threat_events = events_with_xt[
        events_with_xt['xT_delta'] > 0
    ].copy()
    
    # Sort by xT delta (highest first)
    threat_events = threat_events.sort_values('xT_delta', ascending=False)
    
    return events_with_xt, threat_events


def display_top_moves(threat_events, top_n=20, team_filter=None):
    """Display the top xT generating moves"""
    
    if team_filter:
        threat_events = threat_events[threat_events['team'] == team_filter]
        print(f"\n{'='*80}")
        print(f"TOP {top_n} xT MOVES - {team_filter}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"TOP {top_n} xT MOVES - BOTH TEAMS")
        print(f"{'='*80}\n")
    
    top_moves = threat_events.head(top_n)
    
    for idx, (i, row) in enumerate(top_moves.iterrows(), 1):
        player = row.get('player', 'Unknown')
        team = row.get('team', 'Unknown')
        event_type = row.get('event_type', 'Unknown')
        xt_delta = row.get('xT_delta', 0)
        timestamp = row.get('timestamp', 'Unknown')
        
        # Get positions if available
        start_zone = row.get('start_zone', None)
        end_zone = row.get('end_zone', None)
        
        # Format the output
        print(f"{idx:2d}. xT: {xt_delta:+.4f} | {event_type:10s} | {player:25s} | {team:20s}")
        
        # Add position details if available
        if pd.notna(row.get('x')) and pd.notna(row.get('y')):
            x_start = row['x']
            y_start = row['y']
            if pd.notna(row.get('end_x')) and pd.notna(row.get('end_y')):
                x_end = row['end_x']
                y_end = row['end_y']
                print(f"     Position: ({x_start:.1f}, {y_start:.1f}) → ({x_end:.1f}, {y_end:.1f}) | Time: {timestamp}")
            else:
                print(f"     Position: ({x_start:.1f}, {y_start:.1f}) | Time: {timestamp}")
        else:
            print(f"     Time: {timestamp}")
        
        print()
    
    # Summary statistics
    print(f"{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total positive xT moves: {len(threat_events)}")
    print(f"Total xT generated: {threat_events['xT_delta'].sum():.4f}")
    print(f"Average xT per move: {threat_events['xT_delta'].mean():.4f}")
    print(f"Max single move xT: {threat_events['xT_delta'].max():.4f}")
    
    # Breakdown by event type
    print(f"\nBreakdown by event type:")
    event_summary = threat_events.groupby('event_type')['xT_delta'].agg(['count', 'sum', 'mean'])
    event_summary = event_summary.sort_values('sum', ascending=False)
    for event_type, stats in event_summary.iterrows():
        print(f"  {event_type:15s}: {stats['count']:4.0f} moves, {stats['sum']:7.4f} total, {stats['mean']:7.4f} avg")
    
    # Top players
    if 'player' in threat_events.columns:
        print(f"\nTop xT generators:")
        player_summary = threat_events.groupby('player')['xT_delta'].sum().sort_values(ascending=False).head(10)
        for player, total_xt in player_summary.items():
            count = len(threat_events[threat_events['player'] == player])
            avg = total_xt / count
            print(f"  {player:30s}: {total_xt:7.4f} total ({count:3d} moves, {avg:.4f} avg)")


def main():
    """Main function to run the analysis"""
    
    print("="*80)
    print("BARCELONA MATCH xT ANALYZER")
    print("="*80)
    
    # Configuration
    COMPETITION_ID = 11  # La Liga
    SEASON_ID = 90       # 2020/21
    SEASON_NAME = "La Liga 2020/21"
    
    print(f"\nAnalyzing: {SEASON_NAME}")
    
    # Find Barcelona matches
    barca_matches = find_barcelona_matches(COMPETITION_ID, SEASON_ID)
    
    if len(barca_matches) == 0:
        print("No Barcelona matches found!")
        return
    
    # Display matches
    display_match_selection(barca_matches)
    
    # For demo purposes, automatically select a high-scoring match
    # Barcelona vs Villarreal (4-0) - should have many high xT moves
    # You can modify this to take user input or choose different matches
    selected_match = barca_matches.iloc[7]  # Barcelona vs Villarreal 4-0
    match_id = selected_match['match_id']
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {selected_match['home_team']} vs {selected_match['away_team']}")
    print(f"Date: {selected_match['match_date']}")
    print(f"Score: {selected_match['home_score']}-{selected_match['away_score']}")
    print(f"{'='*80}")
    
    # Analyze the match
    all_events, threat_events = analyze_match_xt(match_id)
    
    # Display top moves for Barcelona
    barcelona_team = 'Barcelona'
    display_top_moves(threat_events, top_n=20, team_filter=barcelona_team)
    
    # Optional: Save results to CSV
    print(f"\n{'='*80}")
    print("Saving results...")
    threat_events.to_csv('data/processed/top_xt_moves.csv', index=False)
    print("✓ Saved to data/processed/top_xt_moves.csv")
    
    # Also save a simplified version
    simplified = threat_events[['timestamp', 'team', 'player', 'event_type', 'xT_delta', 'x', 'y', 'end_x', 'end_y']].copy()
    simplified.to_csv('data/processed/top_xt_moves_simple.csv', index=False)
    print("✓ Saved simplified version to data/processed/top_xt_moves_simple.csv")
    

if __name__ == "__main__":
    main()
