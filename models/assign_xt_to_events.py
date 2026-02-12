import numpy as np
import pandas as pd
from statsbombpy import sb
import sys
sys.path.append('models')
from pitch_grid import PitchGrid


class xTAssigner:
    
    def __init__(self, xt_values_path='data/processed/xt_values.npz'):
        data = np.load(xt_values_path)
        self.xt_values = data['xt_values']
        self.grid = PitchGrid(n_cols=12, n_rows=8)
        
        print(f"✓ Loaded xT values for {len(self.xt_values)} zones")
    
    
    def assign_xt_to_match(self, match_id, decay_factor=0.94):
        
        events = sb.events(match_id=match_id)
        
        cols_to_keep = ['type', 'team', 'player', 'location', 'pass_end_location', 
                       'carry_end_location', 'possession', 'timestamp', 
                       'shot_statsbomb_xg', 'shot_outcome', 'dribble_outcome']
        
        cols_to_keep = [c for c in cols_to_keep if c in events.columns]
        events_df = events[cols_to_keep].copy()
        events_df.rename(columns={'type': 'event_type'}, inplace=True)
        
        events_df['x'] = events_df['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
        events_df['y'] = events_df['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
        
        events_df = self.grid.map_events_to_zones(events_df)
        
        events_df['end_x'] = np.nan
        events_df['end_y'] = np.nan
        
        pass_mask = events_df['event_type'] == 'Pass'
        events_df.loc[pass_mask, 'end_x'] = events_df.loc[pass_mask, 'pass_end_location'].apply(
            lambda loc: loc[0] if isinstance(loc, list) else np.nan
        )
        events_df.loc[pass_mask, 'end_y'] = events_df.loc[pass_mask, 'pass_end_location'].apply(
            lambda loc: loc[1] if isinstance(loc, list) else np.nan
        )
        
        carry_mask = events_df['event_type'] == 'Carry'
        events_df.loc[carry_mask, 'end_x'] = events_df.loc[carry_mask, 'carry_end_location'].apply(
            lambda loc: loc[0] if isinstance(loc, list) else np.nan
        )
        events_df.loc[carry_mask, 'end_y'] = events_df.loc[carry_mask, 'carry_end_location'].apply(
            lambda loc: loc[1] if isinstance(loc, list) else np.nan
        )
        
        events_df['end_zone'] = events_df.apply(
            lambda row: self.grid.get_zone(row['end_x'], row['end_y']) 
            if pd.notna(row['end_x']) and pd.notna(row['end_y']) else np.nan,
            axis=1
        )
        
        events_df['xT_start'] = events_df['start_zone'].apply(
            lambda z: self.xt_values[int(z)] if pd.notna(z) else np.nan
        )
        events_df['xT_end'] = events_df['end_zone'].apply(
            lambda z: self.xt_values[int(z)] if pd.notna(z) else np.nan
        )
        
        events_df['xT_delta'] = events_df['xT_end'] - events_df['xT_start']
        
        if 'shot_statsbomb_xg' in events_df.columns:
            shot_mask = events_df['event_type'] == 'Shot'
            events_df.loc[shot_mask, 'xT_end'] = events_df.loc[shot_mask, 'shot_statsbomb_xg']
            events_df.loc[shot_mask, 'xT_delta'] = (
                events_df.loc[shot_mask, 'shot_statsbomb_xg'] - 
                events_df.loc[shot_mask, 'xT_start']
            )
        
        events_df = self._assign_dribble_xt(events_df, decay_factor)
        
        return events_df
    
    
    def _assign_dribble_xt(self, events_df, decay_factor):
        
        dribble_mask = events_df['event_type'] == 'Dribble'
        
        for idx in events_df[dribble_mask].index:
            dribble_event = events_df.loc[idx]
            possession_id = dribble_event['possession']
            dribble_team = dribble_event['team']
            
            if pd.notna(dribble_event.get('dribble_outcome')):
                if dribble_event['dribble_outcome'] == 'Incomplete':
                    events_df.loc[idx, 'xT_delta'] = -events_df.loc[idx, 'xT_start']
                    events_df.loc[idx, 'xT_end'] = 0
                    continue
            
            remaining_poss = events_df[
                (events_df['possession'] == possession_id) &
                (events_df.index > idx) &
                (events_df['team'] == dribble_team)
            ].copy()
            
            if len(remaining_poss) == 0:
                events_df.loc[idx, 'xT_delta'] = 0
                events_df.loc[idx, 'xT_end'] = events_df.loc[idx, 'xT_start']
                continue
            
            next_event = remaining_poss.iloc[0]
            if pd.notna(next_event.get('start_zone')):
                next_zone_xT = self.xt_values[int(next_event['start_zone'])]
                direct_xT_delta = next_zone_xT - events_df.loc[idx, 'xT_start']
            else:
                direct_xT_delta = 0
            
            shots_in_poss = remaining_poss[remaining_poss['event_type'] == 'Shot']
            
            if len(shots_in_poss) > 0:
                shot_event = shots_in_poss.iloc[0]
                
                # Get all actions between dribble and shot (inclusive of dribble)
                actions_before_shot = events_df[
                    (events_df['possession'] == possession_id) &
                    (events_df.index >= idx) &
                    (events_df.index < shot_event.name) &
                    (events_df['team'] == dribble_team)
                ]
                
                weighted_contribs = []
                total_positive = 0
                
                for i, (action_idx, action) in enumerate(actions_before_shot.iterrows()):
                    if action_idx == idx:
                        contrib = direct_xT_delta
                    else:
                        contrib = action.get('xT_delta', 0)
                    
                    if pd.notna(contrib) and contrib > 0:
                        steps_from_shot = len(actions_before_shot) - i - 1
                        weighted_contrib = contrib * (decay_factor ** steps_from_shot)
                        weighted_contribs.append((action_idx, contrib, weighted_contrib))
                        total_positive += weighted_contrib
                
                outcome_value = shot_event.get('shot_statsbomb_xg', 0)
                if pd.notna(shot_event.get('shot_outcome')):
                    if shot_event['shot_outcome'] == 'Goal':
                        outcome_value = 1.0
                
                if total_positive > 0:
                    for action_idx, direct_contrib, weighted_contrib in weighted_contribs:
                        if action_idx == idx:
                            dribble_attributed_xt = (weighted_contrib / total_positive) * outcome_value
                            events_df.loc[idx, 'xT_delta'] = dribble_attributed_xt
                            events_df.loc[idx, 'xT_end'] = events_df.loc[idx, 'xT_start'] + dribble_attributed_xt
                            break
                else:
                    events_df.loc[idx, 'xT_delta'] = direct_xT_delta
                    events_df.loc[idx, 'xT_end'] = events_df.loc[idx, 'xT_start'] + direct_xT_delta
            else:
                events_df.loc[idx, 'xT_delta'] = direct_xT_delta
                events_df.loc[idx, 'xT_end'] = events_df.loc[idx, 'xT_start'] + direct_xT_delta
        
        return events_df


if __name__ == "__main__":
    
    import os
    
    assigner = xTAssigner()
    
    matches = sb.matches(competition_id=11, season_id=90)
    match_id = matches.iloc[0]['match_id']
    
    events_with_xt = assigner.assign_xt_to_match(match_id)
    
    output_path = f'data/processed/match_{match_id}_with_xt.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    events_with_xt.to_csv(output_path, index=False)
    
    print(f"✓ Processed match {match_id}")
    print(f"✓ Saved to {output_path}")
