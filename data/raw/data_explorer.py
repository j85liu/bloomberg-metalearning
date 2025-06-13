import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

def explore_data_structure():
    """
    Comprehensive data exploration to understand file structure and contents
    """
    print("=" * 80)
    print("BLOOMBERG METALEARNING - DATA STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Get current directory (should be the data/ folder)
    data_dir = Path.cwd()
    print(f"Current directory: {data_dir}")
    print()
    
    # 1. DIRECTORY STRUCTURE
    print("📁 DIRECTORY STRUCTURE:")
    print("-" * 40)
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(str(data_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}📂 {os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith(('.csv', '.json', '.xlsx', '.parquet')):
                print(f"{subindent}📄 {file}")
    print()
    
    # 2. CSV FILES ANALYSIS
    print("📊 CSV FILES DETAILED ANALYSIS:")
    print("-" * 40)
    
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    data_summary = {}
    
    for csv_file in csv_files:
        try:
            print(f"\n🔍 ANALYZING: {os.path.relpath(csv_file, data_dir)}")
            print("-" * 60)
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Basic info
            print(f"Shape: {df.shape} (rows x columns)")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Column information
            print(f"\nColumns ({len(df.columns)}):")
            for i, col in enumerate(df.columns):
                dtype = df[col].dtype
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                
                # Sample values
                sample_vals = df[col].dropna().head(3).tolist()
                
                print(f"  {i+1:2d}. {col:<20} | {str(dtype):<12} | Nulls: {null_count:>5} ({null_pct:>5.1f}%) | Sample: {sample_vals}")
            
            # Date columns detection
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].head(10))
                        date_cols.append(col)
                    except:
                        pass
            
            if date_cols:
                print(f"\nPotential date columns: {date_cols}")
                for date_col in date_cols:
                    try:
                        dates = pd.to_datetime(df[date_col])
                        print(f"  {date_col}: {dates.min()} to {dates.max()} ({len(dates)} records)")
                        print(f"    Frequency: {pd.infer_freq(dates.dropna()[:100])}")
                    except Exception as e:
                        print(f"  {date_col}: Could not parse dates - {e}")
            
            # Numerical columns statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"\nNumerical columns summary:")
                stats = df[numeric_cols].describe()
                print(stats.round(4))
            
            # Data sample
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string())
            
            # Store summary for JSON output
            data_summary[os.path.relpath(csv_file, data_dir)] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'date_columns': date_cols,
                'numeric_columns': numeric_cols.tolist(),
                'sample_data': df.head(3).to_dict('records')
            }
            
        except Exception as e:
            print(f"❌ Error reading {csv_file}: {e}")
            data_summary[os.path.relpath(csv_file, data_dir)] = {'error': str(e)}
    
    # 3. VOLATILITY INDICES DETECTION
    print("\n" + "=" * 80)
    print("🎯 VOLATILITY INDICES DETECTION:")
    print("-" * 40)
    
    volatility_keywords = ['vix', 'vvix', 'ovx', 'gvz', 'rvx', 'volatility', 'vol']
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, nrows=5)  # Just peek at first few rows
            file_name = os.path.basename(csv_file).lower()
            
            # Check filename
            has_vol_in_name = any(keyword in file_name for keyword in volatility_keywords)
            
            # Check column names
            vol_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in volatility_keywords)]
            
            if has_vol_in_name or vol_columns:
                print(f"📈 {os.path.relpath(csv_file, data_dir)}")
                if has_vol_in_name:
                    print(f"   - Volatility indicator in filename")
                if vol_columns:
                    print(f"   - Volatility columns: {vol_columns}")
                    
        except Exception as e:
            continue
    
    # 4. SAVE SUMMARY TO JSON
    print(f"\n💾 Saving data summary to 'data_structure_summary.json'")
    with open('data_structure_summary.json', 'w') as f:
        json.dump(data_summary, f, indent=2, default=str)
    
    # 5. RECOMMENDATIONS
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS FOR META-LEARNING SETUP:")
    print("-" * 40)
    
    total_files = len(csv_files)
    print(f"1. Found {total_files} CSV files to work with")
    
    # Check for time series structure
    has_time_series = False
    for file_path, summary in data_summary.items():
        if 'error' not in summary and summary.get('date_columns'):
            has_time_series = True
            break
    
    if has_time_series:
        print("2. ✅ Time series structure detected - good for forecasting")
    else:
        print("2. ⚠️  No clear time series structure detected - may need date column creation")
    
    # Check for multiple volatility indices
    vol_files = [f for f in csv_files if any(kw in os.path.basename(f).lower() for kw in volatility_keywords)]
    if len(vol_files) > 1:
        print(f"3. ✅ Multiple volatility files detected ({len(vol_files)}) - good for cross-asset analysis")
    else:
        print("3. ⚠️  Limited volatility data - may need additional indices")
    
    print("\n4. 🎯 NEXT STEPS:")
    print("   - Copy this entire output to Claude for code generation")
    print("   - Focus on files with volatility indicators")
    print("   - Ensure date columns are properly formatted")
    print("   - Consider data frequency (daily/hourly) for model selection")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    explore_data_structure()