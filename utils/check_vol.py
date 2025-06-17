import pandas as pd

# Load the targets
df = pd.read_csv('data/processed/meta_learning_targets.csv')
df['date'] = pd.to_datetime(df['date'])

# Load the original processed data to check VVIX and VIX values
orig_df = pd.read_csv('data/processed/processed_volatility_data.csv')
orig_df['date'] = pd.to_datetime(orig_df['date'])

print("🔍 INVESTIGATING VOL-OF-VOL RATIO VALUES")
print("="*50)

# Check extreme ratios
ratio_series = df['TARGET_VOL_OF_VOL_RATIO']
print(f"VVIX/VIX Ratio Statistics:")
print(f"  Min: {ratio_series.min():.3f}")
print(f"  Max: {ratio_series.max():.3f}")
print(f"  Mean: {ratio_series.mean():.3f}")
print(f"  Median: {ratio_series.median():.3f}")
print(f"  95th percentile: {ratio_series.quantile(0.95):.3f}")

# Find dates with extreme ratios
high_ratio_dates = df[df['TARGET_VOL_OF_VOL_RATIO'] > 8.0][['date', 'TARGET_VOL_OF_VOL_RATIO']]
print(f"\n📅 Dates with VVIX/VIX > 8.0:")
print(high_ratio_dates.head(10))

# Check the underlying VVIX and VIX values for these extreme dates
if len(high_ratio_dates) > 0:
    extreme_date = high_ratio_dates.iloc[0]['date']
    print(f"\n🔍 Checking underlying data for {extreme_date}:")
    
    orig_row = orig_df[orig_df['date'] == extreme_date]
    if len(orig_row) > 0:
        vvix_val = orig_row['VVIX'].iloc[0]
        vix_val = orig_row['VIX'].iloc[0]
        print(f"  VVIX: {vvix_val}")
        print(f"  VIX: {vix_val}")
        print(f"  Ratio: {vvix_val/vix_val:.3f}")
        
        if vix_val < 15:
            print(f"  ⚠️  Low VIX ({vix_val}) can cause inflated ratios")

# Check if extreme ratios correlate with low VIX periods
merged = df.merge(orig_df[['date', 'VIX', 'VVIX']], on='date')
low_vix_high_ratio = merged[(merged['VIX'] < 15) & (merged['TARGET_VOL_OF_VOL_RATIO'] > 7)]
print(f"\n📊 Periods with VIX < 15 AND Ratio > 7: {len(low_vix_high_ratio)} occurrences")

if len(low_vix_high_ratio) > 0:
    print("This suggests extreme ratios occur during low volatility periods (normal behavior)")
    print("VVIX becomes relatively large compared to very low VIX values")