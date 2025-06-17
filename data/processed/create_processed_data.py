#!/usr/bin/env python3
"""
Script to create processed volatility dataset from 2015
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from enhanced_data_processor import create_processed_volatility_data

if __name__ == "__main__":
    print("Creating processed volatility dataset from 2015...")
    print("This will merge all your volatility, market, and macro data files.")
    print("="*60)
    
    try:
        # Create the processed dataset
        df = create_processed_volatility_data()
        
        print("\n" + "="*60)
        print("🎉 PROCESSING COMPLETED SUCCESSFULLY!")
        print("The processed dataset is ready for meta-learning target calculations.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check that all data files exist in the expected locations.")
        sys.exit(1)