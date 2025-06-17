import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolatilityTargetCalculator:
    """
    Calculate the 5 meta-learning targets from processed volatility data
    with zero tolerance for missing data filling - report issues instead
    """
    
    def __init__(self, data_path: str = "data/processed/processed_volatility_data.csv"):
        """Initialize with processed data"""
        self.data_path = Path(data_path)
        self.df = None
        self.targets_df = None
        self.calculation_report = {
            'targets_calculated': [],
            'targets_failed': [],
            'data_issues': {},
            'buffer_zones': {},
            'null_counts': {}
        }
        
    def load_processed_data(self) -> pd.DataFrame:
        """Load the processed volatility dataset"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Processed data not found: {self.data_path}")
        
        logger.info(f"Loading processed data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Loaded data: {df.shape}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        self.df = df
        return df
    
    def check_required_columns(self, required_cols: List[str], target_name: str) -> bool:
        """Check if required columns exist and report issues"""
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            error_msg = f"Missing columns for {target_name}: {missing_cols}"
            logger.error(error_msg)
            self.calculation_report['data_issues'][target_name] = error_msg
            return False
        
        # Check for excessive nulls in required columns
        null_issues = {}
        for col in required_cols:
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            if null_count > 0:
                null_issues[col] = {
                    'null_count': int(null_count),
                    'null_percentage': round(null_pct, 2)
                }
        
        if null_issues:
            warning_msg = f"Null values found in required columns for {target_name}: {null_issues}"
            logger.warning(warning_msg)
            self.calculation_report['data_issues'][target_name] = warning_msg
        
        return True
    
    def calculate_target_1_vix_term_structure(self) -> pd.Series:
        """
        Target 1: VIX Term Structure Slope (VIX3M - VIX)
        Should have zero NaN if both VIX and VIX3M are complete
        """
        logger.info("Calculating Target 1: VIX Term Structure Slope")
        
        required_cols = ['VIX', 'VIX3M']
        if not self.check_required_columns(required_cols, 'VIX_TERM_STRUCTURE'):
            return pd.Series(dtype=float)
        
        # Calculate the spread
        target = self.df['VIX3M'] - self.df['VIX']
        
        # Check for any NaN values
        null_count = target.isnull().sum()
        if null_count > 0:
            error_msg = f"Unexpected {null_count} NaN values in VIX Term Structure calculation"
            logger.error(error_msg)
            self.calculation_report['data_issues']['VIX_TERM_STRUCTURE'] = error_msg
        else:
            logger.info("✓ VIX Term Structure calculated with zero NaN")
            self.calculation_report['targets_calculated'].append('VIX_TERM_STRUCTURE')
        
        self.calculation_report['null_counts']['TARGET_VIX_TERM_STRUCTURE'] = int(null_count)
        return target
    
    def calculate_realized_volatility_30d(self) -> pd.Series:
        """Calculate 30-day realized volatility from SP500 returns"""
        if 'SP500_RETURNS' not in self.df.columns:
            error_msg = "SP500_RETURNS column not found for realized volatility calculation"
            logger.error(error_msg)
            return pd.Series(dtype=float)
        
        # Calculate 30-day rolling standard deviation, annualized
        returns = self.df['SP500_RETURNS']
        realized_vol = returns.rolling(window=30, min_periods=20).std() * np.sqrt(252) * 100
        
        # Track where we have valid realized vol data
        first_valid_idx = realized_vol.first_valid_index()
        if first_valid_idx is not None:
            buffer_end_date = self.df.loc[first_valid_idx, 'date']
            self.calculation_report['buffer_zones']['realized_vol_30d'] = {
                'first_valid_date': buffer_end_date.isoformat(),
                'buffer_period_days': first_valid_idx
            }
            logger.info(f"Realized volatility buffer period: {first_valid_idx} days until {buffer_end_date}")
        
        return realized_vol
    
    def calculate_target_2_realized_vs_implied(self) -> pd.Series:
        """
        Target 2: Realized vs Implied Volatility Spread (VIX - realized_vol_30d)
        Will have NaN values during buffer period (first ~30 days)
        """
        logger.info("Calculating Target 2: Realized vs Implied Volatility Spread")
        
        required_cols = ['VIX', 'SP500_RETURNS']
        if not self.check_required_columns(required_cols, 'REALIZED_VS_IMPLIED'):
            return pd.Series(dtype=float)
        
        # Calculate realized volatility
        realized_vol_30d = self.calculate_realized_volatility_30d()
        
        # Calculate the spread
        target = self.df['VIX'] - realized_vol_30d
        
        # Report on null values (expected during buffer period)
        null_count = target.isnull().sum()
        first_valid_idx = target.first_valid_index()
        
        if first_valid_idx is not None:
            buffer_end_date = self.df.loc[first_valid_idx, 'date']
            self.calculation_report['buffer_zones']['REALIZED_VS_IMPLIED'] = {
                'buffer_days': first_valid_idx,
                'first_valid_date': buffer_end_date.isoformat(),
                'total_null_values': int(null_count)
            }
            logger.info(f"✓ Realized vs Implied calculated with {null_count} NaN (buffer period)")
            self.calculation_report['targets_calculated'].append('REALIZED_VS_IMPLIED')
        else:
            error_msg = "No valid values calculated for Realized vs Implied"
            logger.error(error_msg)
            self.calculation_report['data_issues']['REALIZED_VS_IMPLIED'] = error_msg
        
        self.calculation_report['null_counts']['TARGET_REALIZED_VS_IMPLIED'] = int(null_count)
        return target
    
    def calculate_target_3_cross_asset_correlation(self) -> pd.Series:
        """
        Target 3: Cross-Asset Volatility Correlation (30-day rolling correlation VIX vs OVX)
        Will have NaN values during buffer period (first ~30 days)
        """
        logger.info("Calculating Target 3: Cross-Asset Volatility Correlation")
        
        required_cols = ['VIX', 'OVX']
        if not self.check_required_columns(required_cols, 'CROSS_ASSET_CORRELATION'):
            return pd.Series(dtype=float)
        
        # Calculate 30-day rolling correlation
        vix = self.df['VIX']
        ovx = self.df['OVX']
        
        # Calculate rolling correlation with minimum periods for stability
        target = vix.rolling(window=30, min_periods=20).corr(ovx)
        
        # Report on null values and buffer period
        null_count = target.isnull().sum()
        first_valid_idx = target.first_valid_index()
        
        if first_valid_idx is not None:
            buffer_end_date = self.df.loc[first_valid_idx, 'date']
            self.calculation_report['buffer_zones']['CROSS_ASSET_CORRELATION'] = {
                'buffer_days': first_valid_idx,
                'first_valid_date': buffer_end_date.isoformat(),
                'total_null_values': int(null_count)
            }
            
            # Check correlation range (should be between -1 and 1)
            valid_corr = target.dropna()
            if len(valid_corr) > 0:
                min_corr = valid_corr.min()
                max_corr = valid_corr.max()
                if min_corr < -1 or max_corr > 1:
                    warning_msg = f"Correlation values outside [-1,1] range: min={min_corr:.3f}, max={max_corr:.3f}"
                    logger.warning(warning_msg)
                    self.calculation_report['data_issues']['CROSS_ASSET_CORRELATION'] = warning_msg
                else:
                    logger.info(f"✓ Cross-Asset Correlation calculated with {null_count} NaN (buffer period)")
                    logger.info(f"  Correlation range: [{min_corr:.3f}, {max_corr:.3f}]")
                    self.calculation_report['targets_calculated'].append('CROSS_ASSET_CORRELATION')
        else:
            error_msg = "No valid correlation values calculated"
            logger.error(error_msg)
            self.calculation_report['data_issues']['CROSS_ASSET_CORRELATION'] = error_msg
        
        self.calculation_report['null_counts']['TARGET_CROSS_ASSET_CORRELATION'] = int(null_count)
        return target
    
    def calculate_target_4_volatility_dispersion(self) -> pd.Series:
        """
        Target 4: Volatility Dispersion (VIX - mean of individual stock volatilities)
        Should have zero NaN if all required columns are complete
        """
        logger.info("Calculating Target 4: Volatility Dispersion")
        
        # Available individual stock volatility indices
        individual_vol_cols = ['VXAPL', 'VXEEM', 'VXAZN']
        available_cols = [col for col in individual_vol_cols if col in self.df.columns]
        
        if len(available_cols) == 0:
            error_msg = f"No individual volatility indices found. Expected: {individual_vol_cols}"
            logger.error(error_msg)
            self.calculation_report['data_issues']['VOLATILITY_DISPERSION'] = error_msg
            return pd.Series(dtype=float)
        
        required_cols = ['VIX'] + available_cols
        if not self.check_required_columns(required_cols, 'VOLATILITY_DISPERSION'):
            return pd.Series(dtype=float)
        
        # Calculate mean of individual volatilities
        individual_vol_mean = self.df[available_cols].mean(axis=1)
        
        # Calculate dispersion
        target = self.df['VIX'] - individual_vol_mean
        
        # Check for null values
        null_count = target.isnull().sum()
        
        if null_count > 0:
            # Investigate which columns contributed to nulls
            null_sources = {}
            for col in required_cols:
                col_nulls = self.df[col].isnull().sum()
                if col_nulls > 0:
                    null_sources[col] = int(col_nulls)
            
            error_msg = f"Unexpected {null_count} NaN values in Volatility Dispersion. Null sources: {null_sources}"
            logger.error(error_msg)
            self.calculation_report['data_issues']['VOLATILITY_DISPERSION'] = error_msg
        else:
            logger.info(f"✓ Volatility Dispersion calculated with zero NaN using columns: {available_cols}")
            self.calculation_report['targets_calculated'].append('VOLATILITY_DISPERSION')
        
        self.calculation_report['null_counts']['TARGET_VOLATILITY_DISPERSION'] = int(null_count)
        return target
    
    def calculate_target_5_vol_of_vol_ratio(self) -> pd.Series:
        """
        Target 5: Vol-of-Vol Ratio (VVIX/VIX)
        Should have zero NaN if both VVIX and VIX are complete and VIX has no zeros
        """
        logger.info("Calculating Target 5: Vol-of-Vol Ratio")
        
        required_cols = ['VVIX', 'VIX']
        if not self.check_required_columns(required_cols, 'VOL_OF_VOL_RATIO'):
            return pd.Series(dtype=float)
        
        # Check for zero or negative VIX values (would cause division issues)
        zero_vix_count = (self.df['VIX'] <= 0).sum()
        if zero_vix_count > 0:
            warning_msg = f"Found {zero_vix_count} zero or negative VIX values - will cause NaN in ratio"
            logger.warning(warning_msg)
            self.calculation_report['data_issues']['VOL_OF_VOL_RATIO'] = warning_msg
        
        # Calculate the ratio
        target = self.df['VVIX'] / self.df['VIX']
        
        # Check for null or infinite values
        null_count = target.isnull().sum()
        inf_count = np.isinf(target).sum()
        
        if null_count > 0 or inf_count > 0:
            error_msg = f"Issues in Vol-of-Vol Ratio: {null_count} NaN, {inf_count} infinite values"
            logger.error(error_msg)
            self.calculation_report['data_issues']['VOL_OF_VOL_RATIO'] = error_msg
        else:
            # Check if ratio values are reasonable (typically 0.5 to 3.0)
            valid_ratios = target.dropna()
            if len(valid_ratios) > 0:
                min_ratio = valid_ratios.min()
                max_ratio = valid_ratios.max()
                median_ratio = valid_ratios.median()
                
                if min_ratio < 0.1 or max_ratio > 10:
                    warning_msg = f"Unusual Vol-of-Vol ratios: range [{min_ratio:.3f}, {max_ratio:.3f}], median={median_ratio:.3f}"
                    logger.warning(warning_msg)
                    self.calculation_report['data_issues']['VOL_OF_VOL_RATIO'] = warning_msg
                else:
                    logger.info(f"✓ Vol-of-Vol Ratio calculated with zero NaN")
                    logger.info(f"  Ratio range: [{min_ratio:.3f}, {max_ratio:.3f}], median: {median_ratio:.3f}")
                    self.calculation_report['targets_calculated'].append('VOL_OF_VOL_RATIO')
        
        self.calculation_report['null_counts']['TARGET_VOL_OF_VOL_RATIO'] = int(null_count)
        return target
    
    def calculate_all_targets(self) -> pd.DataFrame:
        """Calculate all 5 targets and return dataframe with targets"""
        logger.info("="*60)
        logger.info("🎯 CALCULATING META-LEARNING TARGETS")
        logger.info("="*60)
        
        if self.df is None:
            self.load_processed_data()
        
        # Initialize targets dataframe with date column
        targets_df = self.df[['date']].copy()
        
        # Calculate each target
        targets_df['TARGET_VIX_TERM_STRUCTURE'] = self.calculate_target_1_vix_term_structure()
        targets_df['TARGET_REALIZED_VS_IMPLIED'] = self.calculate_target_2_realized_vs_implied()
        targets_df['TARGET_CROSS_ASSET_CORRELATION'] = self.calculate_target_3_cross_asset_correlation()
        targets_df['TARGET_VOLATILITY_DISPERSION'] = self.calculate_target_4_volatility_dispersion()
        targets_df['TARGET_VOL_OF_VOL_RATIO'] = self.calculate_target_5_vol_of_vol_ratio()
        
        # Also include realized volatility for reference
        targets_df['REALIZED_VOL_30D'] = self.calculate_realized_volatility_30d()
        
        self.targets_df = targets_df
        
        # Final summary
        logger.info("="*60)
        logger.info("📊 TARGET CALCULATION SUMMARY")
        logger.info(f"✅ Successfully calculated: {len(self.calculation_report['targets_calculated'])}")
        for target in self.calculation_report['targets_calculated']:
            logger.info(f"   • {target}")
        
        if self.calculation_report['targets_failed']:
            logger.info(f"❌ Failed to calculate: {len(self.calculation_report['targets_failed'])}")
            for target in self.calculation_report['targets_failed']:
                logger.info(f"   • {target}")
        
        if self.calculation_report['data_issues']:
            logger.info("⚠️  Data Issues Found:")
            for target, issue in self.calculation_report['data_issues'].items():
                logger.info(f"   • {target}: {issue}")
        
        logger.info("="*60)
        
        return targets_df
    
    def save_targets(self, output_path: str = "data/processed/meta_learning_targets.csv"):
        """Save targets and calculation report"""
        if self.targets_df is None:
            logger.error("No targets calculated yet. Run calculate_all_targets() first.")
            return
        
        # Create output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save targets CSV
        self.targets_df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved targets to: {output_path}")
        
        # Create detailed statistics for each target
        target_stats = {}
        target_cols = [col for col in self.targets_df.columns if col.startswith('TARGET_')]
        
        for col in target_cols:
            series = self.targets_df[col].dropna()
            if len(series) > 0:
                target_stats[col] = {
                    'count': len(series),
                    'null_count': int(self.targets_df[col].isnull().sum()),
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'median': float(series.median()),
                    'q25': float(series.quantile(0.25)),
                    'q75': float(series.quantile(0.75))
                }
            else:
                target_stats[col] = {'error': 'No valid values'}
        
        # Combine calculation report with statistics
        full_report = {
            'calculation_summary': self.calculation_report,
            'target_statistics': target_stats,
            'dataset_info': {
                'total_rows': len(self.targets_df),
                'date_range': {
                    'start': self.targets_df['date'].min().isoformat(),
                    'end': self.targets_df['date'].max().isoformat()
                },
                'targets_calculated': len(target_cols)
            }
        }
        
        # Save report
        report_path = output_path.parent / f"{output_path.stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        
        logger.info(f"📋 Saved calculation report to: {report_path}")
        
        return output_path, report_path

# Usage function
def calculate_meta_learning_targets(data_path: str = "data/processed/processed_volatility_data.csv"):
    """Main function to calculate meta-learning targets"""
    
    calculator = VolatilityTargetCalculator(data_path)
    
    # Calculate all targets
    targets_df = calculator.calculate_all_targets()
    
    # Save results
    targets_path, report_path = calculator.save_targets()
    
    print(f"\n🎯 TARGET CALCULATION COMPLETED!")
    print(f"📊 Targets saved to: {targets_path}")
    print(f"📋 Report saved to: {report_path}")
    print(f"🎯 Shape: {targets_df.shape}")
    
    # Show null counts for each target
    target_cols = [col for col in targets_df.columns if col.startswith('TARGET_')]
    print(f"\n📈 NULL COUNTS BY TARGET:")
    for col in target_cols:
        null_count = targets_df[col].isnull().sum()
        total_count = len(targets_df)
        null_pct = (null_count / total_count) * 100
        print(f"   {col}: {null_count}/{total_count} ({null_pct:.1f}%)")
    
    return targets_df

if __name__ == "__main__":
    targets_df = calculate_meta_learning_targets()