import numpy as np
import pandas as pd


class PitchGrid:
    
    def __init__(self, n_cols=12, n_rows=8, pitch_length=120, pitch_width=80):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        
        self.zone_length = pitch_length / n_cols
        self.zone_width = pitch_width / n_rows
        
        self.n_zones = n_cols * n_rows
        
        print(f"Grid created: {n_cols} x {n_rows} = {self.n_zones} zones")
        print(f"Each zone: {self.zone_length:.1f} x {self.zone_width:.1f} yards")
    
    
    def get_zone(self, x, y):
        col = int(x / self.zone_length)
        row = int(y / self.zone_width)
        
        if col >= self.n_cols:
            col = self.n_cols - 1
        if row >= self.n_rows:
            row = self.n_rows - 1
        
        zone = row * self.n_cols + col
        return zone
    
    
    def get_zone_center(self, zone):
        row = zone // self.n_cols
        col = zone % self.n_cols
        
        x_center = (col + 0.5) * self.zone_length
        y_center = (row + 0.5) * self.zone_width
        
        return (x_center, y_center)
    
    
    def map_events_to_zones(self, events_df):
        events_df['start_zone'] = events_df.apply(
            lambda row: self.get_zone(row['x'], row['y']) if pd.notna(row['x']) and pd.notna(row['y']) else None,
            axis=1
        )
        return events_df


if __name__ == "__main__":
    grid = PitchGrid(n_cols=12, n_rows=8)
    print(f"Created {grid.n_cols}×{grid.n_rows} pitch grid ({grid.total_zones} zones)")
    print(f"Zone 0 center: {grid.get_zone_center(0)}")
    print(f"Zone 95 center: {grid.get_zone_center(95)}")
