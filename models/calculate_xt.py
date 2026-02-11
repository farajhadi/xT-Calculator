import numpy as np
import pandas as pd
import sys
sys.path.append('models')
from transition_matrix import TransitionMatrixBuilder
from pitch_grid import PitchGrid


class xTCalculator:
    
    def __init__(self, transition_matrix_path='data/processed/transition_matrix.npz'):
        self.builder = TransitionMatrixBuilder.load(transition_matrix_path)
        self.grid = PitchGrid(n_cols=12, n_rows=8)
        
        self.transition_probs = self.builder.transition_probs
        self.goal_probs = self.builder.goal_probs
        
        self.xt_values = None
    
    
    def calculate_xt_iterative(self, max_iterations=100, tolerance=1e-6):
        
        n_zones = len(self.goal_probs)
        
        xt = self.goal_probs.copy()
        
        print(f"\nIterating to calculate xT...")
        print(f"Initial max xT: {xt.max():.6f}")
        
        for iteration in range(max_iterations):
            xt_new = self.goal_probs + self.transition_probs @ xt
            
            diff = np.abs(xt_new - xt).max()
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration+1}: max diff = {diff:.8f}, max xT = {xt_new.max():.6f}")
            
            if diff < tolerance:
                print(f"\n✓ Converged after {iteration+1} iterations")
                self.xt_values = xt_new
                return xt_new
            
            xt = xt_new
        
        print(f"\n⚠ Reached max iterations ({max_iterations})")
        self.xt_values = xt
        return xt
    
    
    def calculate_xt_matrix(self):
        
        n_zones = len(self.goal_probs)
        I = np.eye(n_zones)
        
        print("\nSolving matrix equation: (I - T) × xT = P(goal)")
        
        try:
            xt = np.linalg.solve(I - self.transition_probs, self.goal_probs)
            print("✓ Matrix solution successful")
            self.xt_values = xt
            return xt
        except np.linalg.LinAlgError:
            print("✗ Matrix singular, falling back to iterative")
            return self.calculate_xt_iterative()
    
    
    def get_xt_grid(self):
        
        n_rows, n_cols = 8, 12
        xt_grid = np.zeros((n_rows, n_cols))
        
        for zone in range(96):
            row = zone // n_cols
            col = zone % n_cols
            xt_grid[row, col] = self.xt_values[zone]
        
        return xt_grid
    
    
    def save(self, filepath='data/processed/xt_values.npz'):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        np.savez(filepath, xt_values=self.xt_values)
        print(f"\n✓ Saved xT values to {filepath}")
    
    
    def print_summary(self):
        print("\n" + "="*60)
        print("xT SUMMARY")
        print("="*60)
        
        print(f"\nOverall statistics:")
        print(f"  Min xT: {self.xt_values.min():.6f}")
        print(f"  Max xT: {self.xt_values.max():.6f}")
        print(f"  Mean xT: {self.xt_values.mean():.6f}")
        print(f"  Median xT: {np.median(self.xt_values):.6f}")
        
        defensive_xt = self.xt_values[0:32].mean()
        middle_xt = self.xt_values[32:64].mean()
        attacking_xt = self.xt_values[64:96].mean()
        
        print(f"\nBy pitch area:")
        print(f"  Defensive third (zones 0-31): {defensive_xt:.6f}")
        print(f"  Middle third (zones 32-63): {middle_xt:.6f}")
        print(f"  Attacking third (zones 64-95): {attacking_xt:.6f}")
        
        print(f"\nTop 5 most valuable zones:")
        top_5 = np.argsort(self.xt_values)[-5:][::-1]
        for rank, zone in enumerate(top_5, 1):
            center = self.grid.get_zone_center(zone)
            print(f"  {rank}. Zone {zone} at {center}: xT = {self.xt_values[zone]:.6f}")
        
        print(f"\nBottom 5 least valuable zones:")
        bottom_5 = np.argsort(self.xt_values)[:5]
        for rank, zone in enumerate(bottom_5, 1):
            center = self.grid.get_zone_center(zone)
            print(f"  {rank}. Zone {zone} at {center}: xT = {self.xt_values[zone]:.6f}")


if __name__ == "__main__":
    
    calc = xTCalculator()
    
    calc.calculate_xt_iterative(max_iterations=100)
    
    # Print summary
    calc.print_summary()
    
    # Save results
    calc.save()
    
    print("\n✓ xT calculation complete")

