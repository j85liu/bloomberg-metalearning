#!/usr/bin/env python3
"""
Script to calculate meta-learning targets from processed volatility data
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from target_calculator import calculate_meta_learning_targets

if __name__ == "__main__":
    print("Calculating Meta-Learning Targets...")
    print("This will create the 5 prediction targets for your meta-learning framework.")
    print("="*60)
    
    try:
        # Calculate the targets
        targets_df = calculate_meta_learning_targets()
        
        print("\n" + "="*60)
        print("🎉 TARGET CALCULATION COMPLETED SUCCESSFULLY!")
        print("The targets are ready for your meta-learning models.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check that processed_volatility_data.csv exists in data/processed/")
        sys.exit(1)