import numpy as np
import pandas as pd
from statsbombpy import sb
import sys
sys.path.append('models')
from pitch_grid import PitchGrid


class TransitionMatrixBuilder:
    
    def __init__(self, n_zones=96):
        self.n_zones = n_zones
        self.grid = PitchGrid(n_cols=12, n_rows=8)
        
        self.transition_counts = np.zeros((n_zones, n_zones))
        self.goal_counts = np.zeros(n_zones)
        self.end_counts = np.zeros(n_zones)
        
        self.transition_probs = None
        self.goal_probs = None
        self.end_probs = None
    
    
    def process_match(self, match_id):
        
        events = sb.events(match_id=match_id)
        
        events_df = events[['type', 'team', 'location', 'pass_end_location', 
                            'carry_end_location', 'possession', 'shot_outcome']].copy()
        events_df.rename(columns={'type': 'event_type'}, inplace=True)
        
        events_df['x'] = events_df['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
        events_df['y'] = events_df['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)
        
        events_df = self.grid.map_events_to_zones(events_df)
        
        for poss_id in events_df['possession'].dropna().unique():
            poss_events = events_df[events_df['possession'] == poss_id].copy()
            self._process_possession(poss_events)
    
    
    def _process_possession(self, poss_events):
        
        poss_events = poss_events.dropna(subset=['start_zone']).copy()
        
        if len(poss_events) == 0:
            return
        
        scored_goal = False
        if 'shot_outcome' in poss_events.columns:
            scored_goal = (poss_events['shot_outcome'] == 'Goal').any()
        
        for i in range(len(poss_events) - 1):
            current_event = poss_events.iloc[i]
            next_event = poss_events.iloc[i + 1]
            
            start_zone = int(current_event['start_zone'])
            
            if current_event['event_type'] in ['Pass', 'Carry']:
                if current_event['event_type'] == 'Pass' and isinstance(current_event.get('pass_end_location'), list):
                    end_x = current_event['pass_end_location'][0]
                    end_y = current_event['pass_end_location'][1]
                    end_zone = self.grid.get_zone(end_x, end_y)
                    self.transition_counts[start_zone, end_zone] += 1
                    
                elif current_event['event_type'] == 'Carry' and isinstance(current_event.get('carry_end_location'), list):
                    end_x = current_event['carry_end_location'][0]
                    end_y = current_event['carry_end_location'][1]
                    end_zone = self.grid.get_zone(end_x, end_y)
                    self.transition_counts[start_zone, end_zone] += 1
        
        last_event = poss_events.iloc[-1]
        last_zone = int(last_event['start_zone'])
        
        if scored_goal:
            self.goal_counts[last_zone] += 1
        else:
            self.end_counts[last_zone] += 1
    
    
    def calculate_probabilities(self):
        
        self.transition_probs = np.zeros((self.n_zones, self.n_zones))
        self.goal_probs = np.zeros(self.n_zones)
        self.end_probs = np.zeros(self.n_zones)
        
        for zone in range(self.n_zones):
            total_from_zone = self.transition_counts[zone, :].sum() + self.goal_counts[zone] + self.end_counts[zone]
            
            if total_from_zone > 0:
                self.transition_probs[zone, :] = self.transition_counts[zone, :] / total_from_zone
                self.goal_probs[zone] = self.goal_counts[zone] / total_from_zone
                self.end_probs[zone] = self.end_counts[zone] / total_from_zone
        
        print(f"\n✓ Calculated probabilities")
        print(f"  Zones with data: {(self.transition_probs.sum(axis=1) > 0).sum()}/{self.n_zones}")
        print(f"  Average goal probability: {self.goal_probs.mean():.4f}")
    
    
    def save(self, filepath='data/processed/transition_matrix.npz'):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        np.savez(filepath,
                 transition_probs=self.transition_probs,
                 goal_probs=self.goal_probs,
                 end_probs=self.end_probs,
                 transition_counts=self.transition_counts,
                 goal_counts=self.goal_counts,
                 end_counts=self.end_counts)
        
        print(f"\n✓ Saved to {filepath}")
    
    
    @staticmethod
    def load(filepath='data/processed/transition_matrix.npz'):
        data = np.load(filepath)
        
        builder = TransitionMatrixBuilder()
        builder.transition_probs = data['transition_probs']
        builder.goal_probs = data['goal_probs']
        builder.end_probs = data['end_probs']
        builder.transition_counts = data['transition_counts']
        builder.goal_counts = data['goal_counts']
        builder.end_counts = data['end_counts']
        
        return builder


if __name__ == "__main__":
    
    matches = sb.matches(competition_id=11, season_id=90)
    n_matches = min(10, len(matches))
    
    print(f"Processing {n_matches} matches...")
    
    builder = TransitionMatrixBuilder()
    
    for idx, match in matches.head(n_matches).iterrows():
        match_id = match['match_id']
        try:
            builder.process_match(match_id)
            print(f"  ✓ Match {idx+1}/{n_matches}")
        except Exception as e:
            print(f"  ✗ Match {idx+1} failed: {e}")
    
    # Calculate probabilities
    builder.calculate_probabilities()
    
    # Save results
    builder.save()
    
    print("✓ Transition matrix complete")

