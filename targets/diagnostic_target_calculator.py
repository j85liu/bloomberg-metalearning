#!/usr/bin/env python3
"""
Diagnostic Target Calculator - Enhanced with detailed debugging
to understand why realized volatility calculation is failing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosticVolatilityTargetCalculator:
    """
    Diagnostic version of target calculator with extensive debugging
    """
    
    def __init__(self, data_path: str = "data/processed/processed_volatility_data_final.csv"):
        self.data_path = Path(data_path)
        self.data = None
        self.targets = {}
        self.target_stats = {}
        self.regime_info = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate processed volatility data with comprehensive checks"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {self.data_path}")
            
        logger.info(f"Loading processed data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Check for any remaining NaNs in key columns
        key_cols = ['VIX', 'VIX3M', 'VVIX', 'OVX', 'SP500_RETURNS']
        available_cols = [col for col in key_cols if col in self.data.columns]
        
        nan_counts = self.data[available_cols].isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"Found NaN values in key columns: {nan_counts[nan_counts > 0]}")
        
        logger.info(f"Loaded {len(self.data)} rows with {len(self.data.columns)} columns")
        logger.info(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        logger.info(f"Available key columns: {available_cols}")
        
        return self.data
    
    def diagnose_returns_data(self, returns_col: str = 'SP500_RETURNS') -> None:
        """
        Comprehensive diagnosis of returns data to understand calculation issues
        """
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE RETURNS DATA DIAGNOSIS")
        print("="*80)
        
        if returns_col not in self.data.columns:
            print(f"❌ Column '{returns_col}' not found!")
            print(f"Available columns: {list(self.data.columns)}")
            
            # Try to find returns-like columns
            returns_cols = [col for col in self.data.columns if 'return' in col.lower()]
            if returns_cols:
                print(f"Found potential returns columns: {returns_cols}")
                returns_col = returns_cols[0]
                print(f"Using: {returns_col}")
            else:
                print("No returns columns found - will use VIX changes as proxy")
                return
        
        returns = self.data[returns_col].copy()
        
        print(f"\n📊 RETURNS DATA ANALYSIS ({returns_col}):")
        print(f"Total observations: {len(returns)}")
        print(f"Missing values: {returns.isnull().sum()}")
        print(f"Zero values: {(returns == 0).sum()}")
        print(f"Range: [{returns.min():.6f}, {returns.max():.6f}]")
        print(f"Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")
        
        # Show first 20 values
        print(f"\n📋 FIRST 20 RETURNS VALUES:")
        print("-" * 50)
        for i in range(min(20, len(returns))):
            date = self.data.iloc[i]['date']
            ret_val = returns.iloc[i]
            status = "NaN" if pd.isna(ret_val) else f"{ret_val:.6f}"
            print(f"Day {i+1:2d} ({date.strftime('%Y-%m-%d')}): {status}")
        
        # Test volatility calculation on first few days
        print(f"\n🧮 STEP-BY-STEP VOLATILITY CALCULATION:")
        print("-" * 60)
        
        returns_clean = returns.fillna(0)  # Fill NaNs with 0 for calculation
        
        for i in range(min(15, len(returns_clean))):
            if i == 0:
                print(f"Day {i+1:2d}: Default vol = 20.00 (can't calculate from 1 point)")
            else:
                window_returns = returns_clean.iloc[:i+1]
                n_points = len(window_returns)
                
                if n_points >= 2:
                    # Calculate statistics
                    mean_ret = window_returns.mean()
                    std_ret = window_returns.std()
                    vol_annualized = std_ret * np.sqrt(252) * 100
                    
                    # Check for issues
                    issues = []
                    if pd.isna(std_ret):
                        issues.append("std=NaN")
                    if std_ret == 0:
                        issues.append("std=0")
                    if vol_annualized == 0:
                        issues.append("vol=0")
                    
                    issue_str = f" [{', '.join(issues)}]" if issues else ""
                    
                    print(f"Day {i+1:2d}: n={n_points}, mean={mean_ret:.6f}, std={std_ret:.6f}, vol={vol_annualized:.2f}{issue_str}")
                else:
                    print(f"Day {i+1:2d}: n={n_points} (insufficient for std calculation)")
        
        # Check if all returns are identical (would cause std=0)
        unique_returns = returns_clean.nunique()
        print(f"\n🔍 POTENTIAL ISSUES:")
        print(f"Unique return values: {unique_returns}")
        if unique_returns <= 2:
            print("⚠️  WARNING: Very few unique return values - may cause std=0")
            print(f"Unique values: {returns_clean.unique()[:10]}")
        
        # Check for data patterns
        zero_count = (returns_clean == 0).sum()
        if zero_count > len(returns_clean) * 0.5:
            print(f"⚠️  WARNING: {zero_count}/{len(returns_clean)} returns are zero")
        
        consecutive_zeros = 0
        max_consecutive_zeros = 0
        for val in returns_clean:
            if val == 0:
                consecutive_zeros += 1
                max_consecutive_zeros = max(max_consecutive_zeros, consecutive_zeros)
            else:
                consecutive_zeros = 0
        
        if max_consecutive_zeros > 10:
            print(f"⚠️  WARNING: Up to {max_consecutive_zeros} consecutive zero returns")
    
    def calculate_diagnostic_realized_volatility(self, returns_col: str = 'SP500_RETURNS', 
                                                target_window: int = 30,
                                                min_periods: int = 5) -> pd.Series:
        """
        Enhanced realized volatility calculation with extensive diagnostics
        """
        print(f"\n🔬 DIAGNOSTIC REALIZED VOLATILITY CALCULATION")
        print("="*60)
        
        if returns_col not in self.data.columns:
            print(f"❌ '{returns_col}' not found, creating proxy from VIX")
            if 'VIX' in self.data.columns:
                vix_data = self.data['VIX'].copy()
                returns = vix_data.pct_change()
                print(f"Created VIX-based returns: {returns.describe()}")
            else:
                raise ValueError(f"Neither '{returns_col}' nor 'VIX' available")
        else:
            returns = self.data[returns_col].copy()
        
        # Fill NaN values
        original_nans = returns.isnull().sum()
        returns = returns.fillna(0)
        print(f"Filled {original_nans} NaN values with 0")
        
        # Diagnostic check on returns
        print(f"Returns after cleaning:")
        print(f"  Range: [{returns.min():.6f}, {returns.max():.6f}]")
        print(f"  Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")
        print(f"  Zeros: {(returns == 0).sum()}/{len(returns)}")
        
        realized_vol = pd.Series(index=returns.index, dtype=float)
        
        # Track diagnostic information
        calculation_log = []
        
        print(f"\n📊 DETAILED CALCULATION LOG (first 20 days):")
        print("-" * 80)
        print("Day | Method     | Window Size | Std      | Vol      | Status")
        print("-" * 80)
        
        for i in range(len(returns)):
            date = self.data.iloc[i]['date'] if i < len(self.data) else "N/A"
            
            if i == 0:
                # First day: use default
                vol = 20.0
                method = "Default"
                window_size = 1
                std_val = np.nan
                status = "OK"
                
            else:
                # Expanding window calculation
                window_returns = returns.iloc[:i+1]
                window_size = len(window_returns)
                method = "Expanding"
                
                if window_size >= 2:
                    std_val = window_returns.std()
                    
                    if pd.isna(std_val):
                        vol = 20.0
                        status = "NaN std -> default"
                    elif std_val == 0:
                        vol = 20.0
                        status = "Zero std -> default"
                    else:
                        vol = std_val * np.sqrt(252) * 100
                        status = "Calculated"
                        
                        # Additional checks
                        if vol < 1.0:
                            vol = 20.0
                            status = "Too low -> default"
                        elif vol > 200.0:
                            vol = 200.0
                            status = "Capped at 200"
                else:
                    vol = 20.0
                    std_val = np.nan
                    status = "Insufficient data"
            
            realized_vol.iloc[i] = vol
            
            # Log first 20 calculations
            if i < 20:
                std_str = f"{std_val:.6f}" if not pd.isna(std_val) else "N/A"
                print(f"{i+1:3d} | {method:10s} | {window_size:11d} | {std_str:8s} | {vol:8.2f} | {status}")
            
            calculation_log.append({
                'day': i+1,
                'date': date,
                'method': method,
                'window_size': window_size,
                'std': std_val,
                'vol': vol,
                'status': status
            })
        
        # Summary statistics
        print(f"\n📈 CALCULATION SUMMARY:")
        print(f"Total calculations: {len(realized_vol)}")
        
        status_counts = {}
        for log in calculation_log:
            status = log['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("Status distribution:")
        for status, count in status_counts.items():
            pct = (count / len(calculation_log)) * 100
            print(f"  {status}: {count} ({pct:.1f}%)")
        
        print(f"\nRealized vol range: {realized_vol.min():.2f} - {realized_vol.max():.2f}")
        print(f"Mean: {realized_vol.mean():.2f}, Std: {realized_vol.std():.2f}")
        
        # Check for issues
        default_count = (realized_vol == 20.0).sum()
        if default_count > len(realized_vol) * 0.5:
            print(f"⚠️  WARNING: {default_count}/{len(realized_vol)} values are using default (20.0)")
            print("This suggests the underlying returns data has issues")
        
        return realized_vol
    
    def run_diagnostic_analysis(self):
        """Run comprehensive diagnostic analysis"""
        if self.data is None:
            self.load_data()
        
        # Step 1: Diagnose returns data
        self.diagnose_returns_data()
        
        # Step 2: Test realized volatility calculation
        realized_vol = self.calculate_diagnostic_realized_volatility()
        
        # Step 3: Show impact on spread calculation
        if 'VIX' in self.data.columns:
            vix = self.data['VIX'].copy()
            spread = vix - realized_vol
            
            print(f"\n📊 REALIZED VS IMPLIED SPREAD ANALYSIS:")
            print("-" * 50)
            print("Day | Date       | VIX   | Realized | Spread")
            print("-" * 50)
            
            for i in range(min(15, len(self.data))):
                date = self.data.iloc[i]['date']
                vix_val = vix.iloc[i]
                real_val = realized_vol.iloc[i]
                spread_val = spread.iloc[i]
                print(f"{i+1:3d} | {date.strftime('%Y-%m-%d')} | {vix_val:5.2f} | {real_val:8.2f} | {spread_val:6.2f}")
        
        print(f"\n✅ DIAGNOSTIC ANALYSIS COMPLETE")
        print("="*60)

# Example usage
if __name__ == "__main__":
    print("🔬 STARTING DIAGNOSTIC TARGET CALCULATOR")
    print("="*60)
    
    # Initialize diagnostic calculator
    calculator = DiagnosticVolatilityTargetCalculator()
    
    try:
        # Run comprehensive diagnostic analysis
        calculator.run_diagnostic_analysis()
        
    except Exception as e:
        logger.error(f"❌ Error in diagnostic analysis: {e}")
        import traceback
        traceback.print_exc()