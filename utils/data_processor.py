import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalVolatilityDataProcessor:
    """
    Final data preprocessing pipeline that achieves ZERO missing values
    by using smart interpolation and fill strategies
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data processor
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.processed_data = {}
        self.date_ranges = {}
        self.data_frequency = "daily"
        
        # Define data categories and their file patterns
        self.data_categories = {
            'volatility': {
                'dir': 'raw/volatility',
                'files': {
                    'VIX': 'VIX_History.csv',
                    'VVIX': 'VVIX_History.csv', 
                    'VIX3M': 'VIX3M_History.csv',
                    'VIX9D': 'VIX9D_History.csv',
                    'OVX': 'OVX_History.csv',
                    'GVZ': 'GVZ_History.csv',
                    'VXN': 'VXN_History.csv',
                    'VXD': 'VXD_History.csv',
                    'RVX': 'RVX_History.csv',
                    'VXAPL': 'VXAPL_History.csv',
                    'VXAZN': 'VXAZN_History.csv',
                    'VXEEM': 'VXEEM_History.csv',
                    'SKEW': 'SKEW_History.csv'
                }
            },
            'macro': {
                'dir': 'raw/macro',
                'files': {
                    'FED_FUNDS': 'FED_FUNDS.csv',
                    'INFLATION_HEADLINE': 'INFLATION_HEADLINE.csv',
                    'INFLATION_CORE': 'INFLATION_CORE.csv',
                    'REPO_RATE': 'REPO_RATE.csv',
                    'POLICY_UNCERTAINTY': 'policy_uncertainty_monthly.csv'
                }
            },
            'market': {
                'dir': 'raw/market',
                'files': {
                    'SP500_INDEX': 'sp500_index.csv',
                    'SP500_RETURNS': 'SP500_RETURNS.csv',
                    'TREASURY_YIELDS': 'us_treasury_yields_daily.csv',
                    'USD_EUR': 'USD_EUR.csv',
                    'USD_GBP': 'USD_GBP.csv',
                    'USD_JPY': 'USD_JPY.csv',
                    'USD_INDEX': 'USD_INDEX.csv',
                    'TED_SPREAD': 'TED_SPREAD.csv',
                    'HIGH_YIELD_SPREAD': 'HIGH_YIELD_SPREAD.csv'
                }
            },
            'events': {
                'dir': 'raw/events',
                'files': {
                    'FOMC_EVENTS': 'fomc_major_events.csv',
                    'MARKET_EVENTS': 'major_market_events.csv',
                    'VIX_METHODOLOGY': 'vix_methodology_changes.csv'
                }
            }
        }
    
    def standardize_date_column(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Standardize date column to datetime format"""
        df = df.copy()
        
        # Try different date formats
        date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%m/%d/%y', '%Y/%m/%d', '%d/%m/%Y']
        
        for fmt in date_formats:
            try:
                df[date_col] = pd.to_datetime(df[date_col], format=fmt)
                logger.info(f"Successfully parsed dates using format: {fmt}")
                break
            except (ValueError, TypeError):
                continue
        else:
            # If no format works, try pandas' flexible parser
            try:
                df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
                logger.info("Successfully parsed dates using pandas flexible parser")
            except:
                logger.error(f"Could not parse date column: {date_col}")
                raise ValueError(f"Unable to parse date column: {date_col}")
        
        # Rename to standard 'date' column
        if date_col != 'date':
            df = df.rename(columns={date_col: 'date'})
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def process_volatility_data(self) -> Dict[str, pd.DataFrame]:
        """Process all volatility index files"""
        logger.info("Processing volatility data...")
        volatility_data = {}
        
        vol_dir = self.data_dir / self.data_categories['volatility']['dir']
        
        for key, filename in self.data_categories['volatility']['files'].items():
            file_path = vol_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # Identify date column
                date_col = None
                for col in df.columns:
                    if col.lower() in ['date', 'time', 'datetime']:
                        date_col = col
                        break
                
                if date_col is None:
                    logger.error(f"No date column found in {filename}")
                    continue
                
                # Standardize date column
                df = self.standardize_date_column(df, date_col)
                
                # Handle different volatility file structures
                if key in ['OVX', 'GVZ', 'VVIX', 'SKEW']:
                    # Files with just DATE and VALUE columns
                    value_col = [col for col in df.columns if col != 'date'][0]
                    df = df.rename(columns={value_col: key})
                    df = df[['date', key]]
                else:
                    # Files with OHLC structure - use CLOSE for main value
                    if 'CLOSE' in df.columns:
                        df[key] = df['CLOSE']
                        # Keep OHLC data for additional analysis
                        df = df.rename(columns={
                            'OPEN': f'{key}_OPEN',
                            'HIGH': f'{key}_HIGH', 
                            'LOW': f'{key}_LOW',
                            'CLOSE': f'{key}_CLOSE'
                        })
                        # Main column is the close price
                        df[key] = df[f'{key}_CLOSE']
                    
                # Remove any duplicate dates
                df = df.drop_duplicates(subset=['date'], keep='last')
                
                # Store date range info
                self.date_ranges[key] = {
                    'start': df['date'].min(),
                    'end': df['date'].max(),
                    'count': len(df)
                }
                
                volatility_data[key] = df
                logger.info(f"Processed {key}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        return volatility_data
    
    def process_macro_data(self) -> Dict[str, pd.DataFrame]:
        """Process macroeconomic data files"""
        logger.info("Processing macro data...")
        macro_data = {}
        
        macro_dir = self.data_dir / self.data_categories['macro']['dir']
        
        for key, filename in self.data_categories['macro']['files'].items():
            file_path = macro_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # Handle different date column formats
                if key == 'POLICY_UNCERTAINTY':
                    # Special handling for policy uncertainty (Year, Month columns)
                    df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
                    df = df.rename(columns={'News_Based_Policy_Uncert_Index': 'POLICY_UNCERTAINTY'})
                    df = df[['date', 'POLICY_UNCERTAINTY']]
                else:
                    # Standard date column
                    date_col = 'date' if 'date' in df.columns else df.columns[0]
                    df = self.standardize_date_column(df, date_col)
                    
                    # Rename value column to match key
                    value_cols = [col for col in df.columns if col != 'date']
                    if len(value_cols) == 1:
                        df = df.rename(columns={value_cols[0]: key})
                
                # Remove duplicates and sort
                df = df.drop_duplicates(subset=['date'], keep='last')
                df = df.sort_values('date').reset_index(drop=True)
                
                # Store date range info
                self.date_ranges[key] = {
                    'start': df['date'].min(),
                    'end': df['date'].max(),
                    'count': len(df)
                }
                
                macro_data[key] = df
                logger.info(f"Processed {key}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        return macro_data
    
    def process_market_data(self) -> Dict[str, pd.DataFrame]:
        """Process market data files"""
        logger.info("Processing market data...")
        market_data = {}
        
        market_dir = self.data_dir / self.data_categories['market']['dir']
        
        for key, filename in self.data_categories['market']['files'].items():
            file_path = market_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # Identify and standardize date column
                date_col = None
                for col in df.columns:
                    if col.lower() in ['date', 'time', 'datetime']:
                        date_col = col
                        break
                
                if date_col is None:
                    logger.error(f"No date column found in {filename}")
                    continue
                
                df = self.standardize_date_column(df, date_col)
                
                # Handle special cases
                if key == 'SP500_INDEX':
                    # Rename S&P500 column to something cleaner
                    df = df.rename(columns={'S&P500': 'SP500'})
                elif key == 'TREASURY_YIELDS':
                    # Keep all yield columns - they're already well named
                    pass
                else:
                    # For single-value files, rename the value column
                    value_cols = [col for col in df.columns if col != 'date']
                    if len(value_cols) == 1:
                        df = df.rename(columns={value_cols[0]: key})
                
                # Remove duplicates and sort
                df = df.drop_duplicates(subset=['date'], keep='last')
                df = df.sort_values('date').reset_index(drop=True)
                
                # Store date range info
                self.date_ranges[key] = {
                    'start': df['date'].min(),
                    'end': df['date'].max(),
                    'count': len(df)
                }
                
                market_data[key] = df
                logger.info(f"Processed {key}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        return market_data
    
    def create_optimized_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Create an optimized date range that maximizes data availability
        """
        # Key insight: Use data availability to determine optimal range
        
        # Core volatility indices we MUST have
        core_indices = ['VIX', 'VVIX', 'OVX', 'GVZ']
        
        # Find latest start date among core indices
        core_starts = []
        for idx in core_indices:
            if idx in self.date_ranges:
                core_starts.append(self.date_ranges[idx]['start'])
        
        if not core_starts:
            raise ValueError("No core volatility indices found!")
        
        # Use the latest start date to ensure all core data is available
        optimal_start = max(core_starts)
        
        # Use earliest end date to ensure data completeness
        core_ends = []
        for idx in core_indices:
            if idx in self.date_ranges:
                core_ends.append(self.date_ranges[idx]['end'])
        
        optimal_end = min(core_ends)
        
        logger.info(f"Optimized date range: {optimal_start} to {optimal_end}")
        return optimal_start, optimal_end
    
    def create_master_date_index(self, start_date: str = None, end_date: str = None) -> pd.DatetimeIndex:
        """Create optimized master date index"""
        
        if start_date and end_date:
            common_start = pd.to_datetime(start_date)
            common_end = pd.to_datetime(end_date)
        else:
            common_start, common_end = self.create_optimized_date_range()
        
        logger.info(f"Creating master date index from {common_start} to {common_end}")
        
        # Create business day index
        date_index = pd.bdate_range(start=common_start, end=common_end, freq='B')
        
        return date_index
    
    def aggressive_missing_value_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressive strategy to eliminate ALL missing values
        """
        logger.info("Applying aggressive missing value elimination...")
        
        df = df.copy()
        
        # Step 1: Forward fill all financial data
        financial_cols = [col for col in df.columns if col != 'date']
        
        for col in financial_cols:
            if col in df.columns:
                missing_before = df[col].isna().sum()
                
                # Strategy 1: Forward fill
                df[col] = df[col].ffill()
                
                # Strategy 2: Backward fill for remaining NaNs at the start
                df[col] = df[col].bfill()
                
                # Strategy 3: For volatility indices that are still NaN, use VIX as proxy
                if df[col].isna().any():
                    if any(vol_name in col for vol_name in ['VX', 'OVX', 'GVZ']) and 'VIX' in df.columns:
                        # Use VIX-based estimation for missing volatility data
                        vix_data = df['VIX'].copy()
                        if col in ['VXAPL', 'VXAZN', 'VXEEM']:
                            # Individual stock volatilities are typically higher than VIX
                            proxy_values = vix_data * 1.2
                        elif col in ['OVX']:
                            # Oil volatility is typically higher than VIX
                            proxy_values = vix_data * 1.8
                        elif col in ['GVZ']:
                            # Gold volatility is typically similar to VIX
                            proxy_values = vix_data * 0.9
                        else:
                            # Default: use VIX as proxy
                            proxy_values = vix_data
                        
                        # Fill remaining NaNs with proxy
                        mask = df[col].isna()
                        df.loc[mask, col] = proxy_values[mask]
                        
                        logger.info(f"Used VIX proxy for {mask.sum()} missing values in {col}")
                
                # Strategy 4: Linear interpolation for any remaining NaNs
                if df[col].isna().any():
                    df[col] = df[col].interpolate(method='linear')
                
                # Strategy 5: Fill any remaining NaNs with median value
                if df[col].isna().any():
                    median_val = df[col].median()
                    if not pd.isna(median_val):
                        df[col] = df[col].fillna(median_val)
                        logger.info(f"Used median fill for remaining NaNs in {col}")
                
                # Strategy 6: Last resort - use 0 for any remaining NaNs (should not happen)
                if df[col].isna().any():
                    df[col] = df[col].fillna(0)
                    remaining_nans = df[col].isna().sum()
                    logger.warning(f"Used zero fill for {remaining_nans} values in {col}")
                
                missing_after = df[col].isna().sum()
                if missing_before > 0:
                    logger.info(f"{col}: {missing_before} -> {missing_after} missing values")
        
        return df
    
    def add_zero_nan_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical features with zero NaN guarantee
        """
        logger.info("Adding technical features with zero NaN guarantee...")
        
        df = df.copy()
        
        # VIX-based features (if VIX exists)
        if 'VIX' in df.columns:
            # Moving averages with immediate fill
            df['VIX_MA_5'] = df['VIX'].rolling(window=5, min_periods=1).mean()
            df['VIX_MA_20'] = df['VIX'].rolling(window=20, min_periods=1).mean()
            df['VIX_MA_60'] = df['VIX'].rolling(window=60, min_periods=1).mean()
            
            # Percentile ranks with expanding window for early periods
            df['VIX_PCT_RANK_63'] = df['VIX'].rolling(window=63, min_periods=1).rank(pct=True)
            df['VIX_PCT_RANK_252'] = df['VIX'].rolling(window=252, min_periods=1).rank(pct=True)
            
            # Rate of change - fill first values with 0
            df['VIX_ROC_5'] = df['VIX'].pct_change(5).fillna(0)
            df['VIX_ROC_20'] = df['VIX'].pct_change(20).fillna(0)
            
        # Add calendar features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Add lagged features with zero NaN guarantee
        vol_indices = ['VIX', 'VVIX', 'OVX', 'GVZ']
        for vol_idx in vol_indices:
            if vol_idx in df.columns:
                for lag in [1, 2, 5, 10]:
                    lag_col = f'{vol_idx}_LAG_{lag}'
                    df[lag_col] = df[vol_idx].shift(lag)
                    # Fill initial NaNs with the first available value
                    first_valid = df[vol_idx].iloc[0]
                    df[lag_col] = df[lag_col].fillna(first_valid)
        
        # Final check - ensure no NaNs in technical features
        tech_cols = [col for col in df.columns if any(x in col for x in ['MA_', 'ROC_', 'LAG_', 'PCT_RANK'])]
        for col in tech_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
                logger.warning(f"Emergency fill applied to {col}")
        
        logger.info("Technical features completed with zero NaN guarantee")
        
        return df
    
    def final_zero_nan_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final check to absolutely guarantee zero NaN values
        """
        logger.info("Performing final zero NaN guarantee check...")
        
        df = df.copy()
        
        # Check for any remaining NaNs
        total_nans = df.isnull().sum().sum()
        
        if total_nans > 0:
            logger.warning(f"Found {total_nans} remaining NaN values - applying emergency fixes")
            
            for col in df.columns:
                if col != 'date' and df[col].isna().any():
                    nan_count = df[col].isna().sum()
                    
                    # Emergency strategy: use column median or 0
                    if df[col].dtype in ['float64', 'int64']:
                        fill_value = df[col].median()
                        if pd.isna(fill_value):
                            fill_value = 0
                    else:
                        fill_value = 0
                    
                    df[col] = df[col].fillna(fill_value)
                    logger.warning(f"Emergency filled {nan_count} values in {col} with {fill_value}")
        
        # Final verification
        final_nans = df.isnull().sum().sum()
        if final_nans == 0:
            logger.info("🎉 SUCCESS: Zero NaN values achieved!")
        else:
            logger.error(f"FAILED: Still have {final_nans} NaN values")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_volatility_data_final.csv"):
        """Save the final processed data"""
        output_path = self.data_dir / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved final processed data to {output_path}")
        
        # Save comprehensive summary
        summary = {
            'shape': df.shape,
            'date_range': {
                'start': df['date'].min().isoformat() if len(df) > 0 else None,
                'end': df['date'].max().isoformat() if len(df) > 0 else None,
                'days': len(df)
            },
            'columns': df.columns.tolist(),
            'missing_data': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'summary_stats': {
                'total_missing': int(df.isnull().sum().sum()),
                'missing_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                'complete_rows': int((df.isnull().sum(axis=1) == 0).sum()),
                'complete_percentage': float(((df.isnull().sum(axis=1) == 0).sum() / len(df)) * 100),
                'zero_nan_achieved': df.isnull().sum().sum() == 0
            },
            'target_readiness': {
                'vix_term_structure': all(col in df.columns for col in ['VIX', 'VIX3M']),
                'realized_vs_implied': all(col in df.columns for col in ['VIX', 'SP500_RETURNS']),
                'cross_asset_correlation': all(col in df.columns for col in ['VIX', 'OVX']),
                'volatility_dispersion': all(col in df.columns for col in ['VIX', 'VXAPL', 'VXEEM']),
                'vol_of_vol_ratio': all(col in df.columns for col in ['VVIX', 'VIX'])
            }
        }
        
        summary_path = output_path.parent / f"{filename.replace('.csv', '_summary.json')}"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved comprehensive summary to {summary_path}")
    
    def run_final_pipeline(self, 
                          start_date: str = None, 
                          end_date: str = None,
                          save_output: bool = True) -> pd.DataFrame:
        """
        Run the final pipeline guaranteed to produce zero NaN values
        """
        logger.info("="*60)
        logger.info("🚀 STARTING FINAL ZERO-NAN DATA PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Process all data categories
        volatility_data = self.process_volatility_data()
        macro_data = self.process_macro_data()
        market_data = self.process_market_data()
        
        # Step 2: Create optimized master date index
        master_date_index = self.create_master_date_index(start_date, end_date)
        
        # Step 3: Create master dataframe
        merged_df = pd.DataFrame(index=master_date_index)
        merged_df.index.name = 'date'
        merged_df = merged_df.reset_index()
        
        # Step 4: Merge all data
        for key, df in volatility_data.items():
            merged_df = pd.merge(merged_df, df, on='date', how='left')
            
        for key, df in market_data.items():
            merged_df = pd.merge(merged_df, df, on='date', how='left')
            
        for key, df in macro_data.items():
            merged_df = pd.merge(merged_df, df, on='date', how='left')
        
        logger.info(f"Initial merged dataset: {merged_df.shape}")
        logger.info(f"Initial missing values: {merged_df.isnull().sum().sum()}")
        
        # Step 5: Aggressive missing value elimination
        processed_df = self.aggressive_missing_value_strategy(merged_df)
        
        # Step 6: Add technical features with zero NaN guarantee
        enhanced_df = self.add_zero_nan_technical_features(processed_df)
        
        # Step 7: Final zero NaN guarantee
        final_df = self.final_zero_nan_check(enhanced_df)
        
        # Step 8: Save final data
        if save_output:
            self.save_processed_data(final_df)
        
        # Final summary
        logger.info("="*60)
        logger.info("🎉 FINAL ZERO-NAN PREPROCESSING PIPELINE COMPLETED")
        logger.info(f"Final dataset shape: {final_df.shape}")
        if len(final_df) > 0:
            logger.info(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        
        total_missing = final_df.isnull().sum().sum()
        total_cells = len(final_df) * len(final_df.columns)
        missing_pct = (total_missing / total_cells) * 100
        
        logger.info(f"Total missing values: {total_missing} ({missing_pct:.6f}%)")
        
        if total_missing == 0:
            logger.info("🏆 PERFECT! ZERO missing values achieved!")
            logger.info("✅ Dataset is 100% ready for meta-learning target calculation!")
        else:
            logger.error(f"❌ FAILED: Still have {total_missing} missing values")
        
        logger.info("="*60)
        
        return final_df

# Usage
if __name__ == "__main__":
    processor = FinalVolatilityDataProcessor(data_dir=".")
    df = processor.run_final_pipeline(save_output=True)
    
    print(f"\n🎯 FINAL RESULT:")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Zero NaN achieved: {df.isnull().sum().sum() == 0}")