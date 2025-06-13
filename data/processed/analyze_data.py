#!/usr/bin/env python3
"""
Comprehensive analysis of processed volatility data
Run this script to get detailed insights into data quality and completeness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_processed_data(file_path='data/processed/processed_volatility_data_final.csv'):
    """
    Comprehensive analysis of the processed volatility dataset
    """
    print("=" * 80)
    print("🔍 COMPREHENSIVE ANALYSIS OF PROCESSED VOLATILITY DATA")
    print("=" * 80)
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Successfully loaded data from {file_path}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # 1. BASIC DATASET INFO
    print(f"\n📊 BASIC DATASET INFO:")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total trading days: {len(df):,}")
    
    # 2. MISSING VALUES ANALYSIS
    print(f"\n🔍 MISSING VALUES ANALYSIS:")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    # Show columns with missing values
    missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    if len(missing_cols) > 0:
        print(f"⚠️  Found {len(missing_cols)} columns with missing values:")
        print(f"{'Column':<25} {'Missing':<10} {'Percentage':<12}")
        print("-" * 50)
        for col, count in missing_cols.head(20).items():
            pct = (count / len(df)) * 100
            print(f"{col:<25} {count:<10} {pct:<12.2f}%")
        
        if len(missing_cols) > 20:
            print(f"... and {len(missing_cols) - 20} more columns with missing values")
    else:
        print("✅ No missing values found!")
    
    # 3. DATA TYPES ANALYSIS
    print(f"\n📈 DATA TYPES ANALYSIS:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"{str(dtype):<15}: {count} columns")
    
    # 4. DATE CONTINUITY CHECK
    print(f"\n📅 DATE CONTINUITY CHECK:")
    date_diff = df['date'].diff()
    
    # Check for weekends (should be 1 business day gaps)
    business_days = pd.bdate_range(df['date'].min(), df['date'].max())
    expected_days = len(business_days)
    actual_days = len(df)
    
    print(f"Expected business days: {expected_days:,}")
    print(f"Actual data days: {actual_days:,}")
    print(f"Missing trading days: {expected_days - actual_days:,}")
    
    # Find large gaps (> 5 days)
    large_gaps = date_diff[date_diff > pd.Timedelta(days=5)]
    if len(large_gaps) > 0:
        print(f"\n⚠️  Found {len(large_gaps)} large date gaps (>5 days):")
        for idx in large_gaps.index[:10]:
            gap_start = df.loc[idx-1, 'date'].date()
            gap_end = df.loc[idx, 'date'].date()
            gap_size = large_gaps.loc[idx].days
            print(f"  {gap_start} → {gap_end} ({gap_size} days)")
    else:
        print("✅ No large date gaps found")
    
    # 5. KEY VOLATILITY INDICES ANALYSIS
    print(f"\n📊 KEY VOLATILITY INDICES ANALYSIS:")
    vol_indices = ['VIX', 'VVIX', 'VIX3M', 'VIX9D', 'OVX', 'GVZ']
    
    for idx in vol_indices:
        if idx in df.columns:
            col_data = df[idx].dropna()
            if len(col_data) > 0:
                print(f"{idx:<6}: {len(col_data):>4,} values | "
                      f"Range: {col_data.min():>6.2f} - {col_data.max():>6.2f} | "
                      f"Mean: {col_data.mean():>6.2f} | "
                      f"Current: {col_data.iloc[-1]:>6.2f}")
            else:
                print(f"{idx:<6}: No valid data")
        else:
            print(f"{idx:<6}: Column not found")
    
    # 6. TECHNICAL FEATURES ANALYSIS
    print(f"\n🔧 TECHNICAL FEATURES ANALYSIS:")
    tech_features = [col for col in df.columns if any(x in col for x in ['MA_', 'ROC_', 'LAG_', 'PCT_RANK'])]
    
    if tech_features:
        print(f"Found {len(tech_features)} technical features:")
        tech_missing = df[tech_features].isnull().sum()
        tech_with_missing = tech_missing[tech_missing > 0]
        
        if len(tech_with_missing) > 0:
            print("Technical features with missing values:")
            for feat, missing in tech_with_missing.head(10).items():
                pct = (missing / len(df)) * 100
                print(f"  {feat}: {missing} ({pct:.1f}%)")
        else:
            print("✅ All technical features are complete")
    
    # 7. DATA QUALITY ISSUES
    print(f"\n⚠️  DATA QUALITY ISSUES:")
    issues = []
    
    # Check for infinite values
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            inf_count = np.isinf(df[col]).sum()
            inf_cols.append((col, inf_count))
    
    if inf_cols:
        issues.append(f"Infinite values in {len(inf_cols)} columns")
        for col, count in inf_cols[:5]:
            print(f"  {col}: {count} infinite values")
    
    # Check for duplicate dates
    dup_dates = df[df['date'].duplicated()]
    if len(dup_dates) > 0:
        issues.append(f"Duplicate dates: {len(dup_dates)}")
        print(f"  Found {len(dup_dates)} duplicate dates")
    
    # Check for negative volatility values (shouldn't happen)
    vol_cols = [col for col in df.columns if 'VIX' in col or col in ['OVX', 'GVZ', 'SKEW']]
    negative_vol = []
    for col in vol_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                negative_vol.append((col, neg_count))
    
    if negative_vol:
        issues.append(f"Negative volatility values in {len(negative_vol)} columns")
        for col, count in negative_vol:
            print(f"  {col}: {count} negative values")
    
    if not issues:
        print("✅ No major data quality issues detected!")
    
    # 8. READINESS FOR META-LEARNING TARGETS
    print(f"\n🎯 READINESS FOR META-LEARNING TARGETS:")
    
    targets_ready = {
        "VIX Term Structure Slope": ['VIX', 'VIX3M'] if all(col in df.columns for col in ['VIX', 'VIX3M']) else False,
        "Realized vs Implied Vol": ['VIX', 'SP500_RETURNS'] if all(col in df.columns for col in ['VIX', 'SP500_RETURNS']) else False,
        "Cross-Asset Correlation": ['VIX', 'OVX'] if all(col in df.columns for col in ['VIX', 'OVX']) else False,
        "Volatility Dispersion": ['VIX', 'VXAPL', 'VXEEM'] if all(col in df.columns for col in ['VIX', 'VXAPL', 'VXEEM']) else False,
        "Vol-of-Vol Ratio": ['VVIX', 'VIX'] if all(col in df.columns for col in ['VVIX', 'VIX']) else False
    }
    
    for target, status in targets_ready.items():
        if status:
            # Check data availability
            required_cols = status
            min_data_points = min([df[col].notna().sum() for col in required_cols])
            pct_available = (min_data_points / len(df)) * 100
            status_icon = "✅" if pct_available > 90 else "⚠️" if pct_available > 70 else "❌"
            print(f"  {status_icon} {target:<25}: {pct_available:>5.1f}% data available")
        else:
            print(f"  ❌ {target:<25}: Missing required columns")
    
    # 9. SAMPLE DATA PREVIEW
    print(f"\n📋 SAMPLE DATA PREVIEW (Last 5 rows):")
    key_cols = ['date', 'VIX', 'VVIX', 'VIX3M', 'OVX', 'GVZ', 'SP500_RETURNS']
    available_cols = [col for col in key_cols if col in df.columns]
    
    sample_data = df[available_cols].tail()
    print(sample_data.to_string(index=False))
    
    # 10. RECOMMENDATIONS
    print(f"\n💡 RECOMMENDATIONS:")
    
    if missing_counts.sum() > 0:
        high_missing = missing_cols[missing_cols > len(df) * 0.1]  # > 10% missing
        if len(high_missing) > 0:
            print(f"  ⚠️  Consider reviewing columns with >10% missing data:")
            for col in high_missing.head(5).index:
                print(f"     - {col}")
    
    if len(df) < 1000:
        print(f"  ⚠️  Dataset might be small for robust meta-learning ({len(df)} samples)")
    elif len(df) > 5000:
        print(f"  ✅ Good dataset size for meta-learning ({len(df)} samples)")
    
    # Check if we have enough regime diversity
    if 'VIX' in df.columns:
        vix_data = df['VIX'].dropna()
        if len(vix_data) > 0:
            low_vol_days = (vix_data < 15).sum()
            high_vol_days = (vix_data > 30).sum()
            crisis_vol_days = (vix_data > 50).sum()
            
            print(f"  📊 Market regime diversity:")
            print(f"     - Low volatility days (VIX < 15): {low_vol_days} ({low_vol_days/len(vix_data)*100:.1f}%)")
            print(f"     - High volatility days (VIX > 30): {high_vol_days} ({high_vol_days/len(vix_data)*100:.1f}%)")
            print(f"     - Crisis volatility days (VIX > 50): {crisis_vol_days} ({crisis_vol_days/len(vix_data)*100:.1f}%)")
    
    print(f"\n✅ Analysis complete! Dataset appears ready for meta-learning target calculation.")
    
    return df

if __name__ == "__main__":
    # Run the comprehensive analysis
    df = analyze_processed_data()
    
    # Optional: Save analysis results
    if df is not None:
        # Create a simple data quality report
        print(f"\n💾 Creating data quality summary...")
        
        quality_summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range_start': df['date'].min().isoformat(),
            'date_range_end': df['date'].max().isoformat(),
            'total_missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'ready_for_meta_learning': df.isnull().sum().sum() < len(df) * 0.1  # Less than 10% missing overall
        }
        
        print(f"📊 Quality Summary:")
        for key, value in quality_summary.items():
            print(f"   {key}: {value}")