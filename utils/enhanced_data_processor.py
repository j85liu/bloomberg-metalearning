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

class VolatilityDataProcessor2015:
    """
    Enhanced data processor that preserves OHLC data and handles date ranges properly
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data processor"""
        self.data_dir = Path(data_dir)
        self.processed_data = {}
        self.date_ranges = {}
        
        # We'll determine the actual start date based on data availability
        self.buffer_start = pd.to_datetime('2014-12-01')  # Buffer for rolling calculations
        
        logger.info(f"Processor initialized with buffer start: {self.buffer_start}")
    
    def standardize_date_column(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Standardize date column to datetime format"""
        df = df.copy()
        
        # Try different date formats commonly found in financial data
        date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%m/%d/%y', '%Y/%m/%d', '%d/%m/%Y']
        
        for fmt in date_formats:
            try:
                df[date_col] = pd.to_datetime(df[date_col], format=fmt)
                logger.debug(f"Successfully parsed dates using format: {fmt}")
                break
            except (ValueError, TypeError):
                continue
        else:
            # If no format works, try pandas' flexible parser
            try:
                df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
                logger.debug("Successfully parsed dates using pandas flexible parser")
            except:
                logger.error(f"Could not parse date column: {date_col}")
                raise ValueError(f"Unable to parse date column: {date_col}")
        
        # Rename to standard 'date' column and sort
        if date_col != 'date':
            df = df.rename(columns={date_col: 'date'})
        
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def load_volatility_data(self) -> Dict[str, pd.DataFrame]:
        """Load all volatility index files, preserving OHLC structure"""
        logger.info("Loading volatility data with OHLC preservation...")
        volatility_data = {}
        
        vol_files = {
            'VIX': 'data/raw/volatility/VIX_History.csv',
            'VVIX': 'data/raw/volatility/VVIX_History.csv',
            'VIX3M': 'data/raw/volatility/VIX3M_History.csv',
            'VIX9D': 'data/raw/volatility/VIX9D_History.csv',
            'OVX': 'data/raw/volatility/OVX_History.csv',
            'GVZ': 'data/raw/volatility/GVZ_History.csv',
            'VXN': 'data/raw/volatility/VXN_History.csv',
            'VXD': 'data/raw/volatility/VXD_History.csv',
            'RVX': 'data/raw/volatility/RVX_History.csv',
            'VXAPL': 'data/raw/volatility/VXAPL_History.csv',
            'VXAZN': 'data/raw/volatility/VXAZN_History.csv',
            'VXEEM': 'data/raw/volatility/VXEEM_History.csv',
            'SKEW': 'data/raw/volatility/SKEW_History.csv'
        }
        
        for key, file_path in vol_files.items():
            full_path = Path(file_path)
            
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                continue
            
            try:
                df = pd.read_csv(full_path)
                
                # Find date column
                date_col = None
                for col in df.columns:
                    if col.upper() in ['DATE', 'TIME', 'DATETIME']:
                        date_col = col
                        break
                
                if date_col is None:
                    logger.error(f"No date column found in {file_path}")
                    continue
                
                # Standardize dates
                df = self.standardize_date_column(df, date_col)
                
                # Filter to our buffer start date
                df = df[df['date'] >= self.buffer_start].copy()
                
                # Handle different file structures - PRESERVE OHLC
                if key in ['OVX', 'GVZ', 'VVIX', 'SKEW']:
                    # Files with DATE and single value column
                    value_col = [col for col in df.columns if col != 'date'][0]
                    df = df.rename(columns={value_col: key})
                    df = df[['date', key]]
                else:
                    # Files with OHLC structure - KEEP ALL COLUMNS
                    if all(col in df.columns for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE']):
                        # Rename OHLC columns with prefix
                        df = df.rename(columns={
                            'OPEN': f'{key}_OPEN',
                            'HIGH': f'{key}_HIGH', 
                            'LOW': f'{key}_LOW',
                            'CLOSE': f'{key}_CLOSE'
                        })
                        # Add main column as CLOSE value for backwards compatibility
                        df[key] = df[f'{key}_CLOSE']
                    else:
                        logger.warning(f"Expected OHLC columns not found in {key}")
                
                # Remove duplicates
                df = df.drop_duplicates(subset=['date'], keep='last')
                
                volatility_data[key] = df
                logger.info(f"Loaded {key}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
                logger.debug(f"  Columns: {list(df.columns)}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        return volatility_data
    
    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data files"""
        logger.info("Loading market data...")
        market_data = {}
        
        market_files = {
            'SP500_RETURNS': 'data/raw/market/SP500_RETURNS.csv',
            'SP500_INDEX': 'data/raw/market/sp500_index.csv',
            'TREASURY_YIELDS': 'data/raw/market/us_treasury_yields_daily.csv',
            'USD_EUR': 'data/raw/market/USD_EUR.csv',
            'USD_GBP': 'data/raw/market/USD_GBP.csv',
            'USD_JPY': 'data/raw/market/USD_JPY.csv',
            'USD_INDEX': 'data/raw/market/USD_INDEX.csv',
            'TED_SPREAD': 'data/raw/market/TED_SPREAD.csv',
            'HIGH_YIELD_SPREAD': 'data/raw/market/HIGH_YIELD_SPREAD.csv'
        }
        
        for key, file_path in market_files.items():
            full_path = Path(file_path)
            
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                continue
            
            try:
                df = pd.read_csv(full_path)
                
                # Find date column
                date_col = None
                for col in df.columns:
                    if col.lower() in ['date', 'time', 'datetime']:
                        date_col = col
                        break
                
                if date_col is None:
                    logger.error(f"No date column found in {file_path}")
                    continue
                
                # Standardize dates
                df = self.standardize_date_column(df, date_col)
                
                # Filter to our buffer start date
                df = df[df['date'] >= self.buffer_start].copy()
                
                # Handle specific cases
                if key == 'SP500_INDEX':
                    # Rename S&P500 column to SP500
                    if 'S&P500' in df.columns:
                        df = df.rename(columns={'S&P500': 'SP500'})
                elif key == 'TREASURY_YIELDS':
                    # Keep treasury yield columns as they are (well named)
                    pass
                else:
                    # For single-value files, rename the value column
                    value_cols = [col for col in df.columns if col != 'date']
                    if len(value_cols) == 1:
                        df = df.rename(columns={value_cols[0]: key})
                
                # Remove duplicates
                df = df.drop_duplicates(subset=['date'], keep='last')
                
                market_data[key] = df
                logger.info(f"Loaded {key}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
                
                # Special logging for SP500_RETURNS to confirm start date
                if key == 'SP500_RETURNS':
                    logger.info(f"  ⚠️  SP500_RETURNS actual start: {df['date'].min()}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        return market_data
    
    def load_macro_data(self) -> Dict[str, pd.DataFrame]:
        """Load macroeconomic data"""
        logger.info("Loading macro data...")
        macro_data = {}
        
        macro_files = {
            'FED_FUNDS': 'data/raw/macro/FED_FUNDS.csv',
            'INFLATION_HEADLINE': 'data/raw/macro/INFLATION_HEADLINE.csv',
            'INFLATION_CORE': 'data/raw/macro/INFLATION_CORE.csv',
            'REPO_RATE': 'data/raw/macro/REPO_RATE.csv',
            'POLICY_UNCERTAINTY': 'data/raw/macro/policy_uncertainty_monthly.csv'
        }
        
        for key, file_path in macro_files.items():
            full_path = Path(file_path)
            
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                continue
            
            try:
                df = pd.read_csv(full_path)
                
                # Handle special case for policy uncertainty
                if key == 'POLICY_UNCERTAINTY':
                    # Convert Year/Month to date
                    df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
                    df = df.rename(columns={'News_Based_Policy_Uncert_Index': 'POLICY_UNCERTAINTY'})
                    df = df[['date', 'POLICY_UNCERTAINTY']]
                else:
                    # Standard date processing
                    date_col = 'date' if 'date' in df.columns else df.columns[0]
                    df = self.standardize_date_column(df, date_col)
                    
                    # Rename value column to match expected names
                    if key == 'INFLATION_HEADLINE' and 'CPIAUCSL' in df.columns:
                        df = df.rename(columns={'CPIAUCSL': 'INFLATION_HEADLINE'})
                    elif key == 'INFLATION_CORE' and 'CPILFESL' in df.columns:
                        df = df.rename(columns={'CPILFESL': 'INFLATION_CORE'})
                    elif key == 'FED_FUNDS' and 'FEDFUNDS' in df.columns:
                        df = df.rename(columns={'FEDFUNDS': 'FED_FUNDS'})
                    elif key == 'REPO_RATE' and 'SOFR' in df.columns:
                        df = df.rename(columns={'SOFR': 'REPO_RATE'})
                
                # Filter to date range
                df = df[df['date'] >= self.buffer_start].copy()
                
                # Remove duplicates
                df = df.drop_duplicates(subset=['date'], keep='last')
                
                macro_data[key] = df
                logger.info(f"Loaded {key}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        return macro_data
    
    def determine_optimal_start_date(self, market_data: Dict) -> pd.Timestamp:
        """Determine the optimal start date based on critical data availability"""
        
        # Check SP500_RETURNS availability (critical for realized vol calculation)
        if 'SP500_RETURNS' in market_data:
            sp500_start = market_data['SP500_RETURNS']['date'].min()
            logger.info(f"SP500_RETURNS starts: {sp500_start}")
        else:
            logger.warning("SP500_RETURNS not available!")
            sp500_start = pd.to_datetime('2015-01-01')
        
        # Use SP500_RETURNS start date directly (no buffer)
        actual_start = sp500_start  # <-- REMOVED THE BUFFER
        
        logger.info(f"Optimal start date (data availability): {actual_start}")
        return actual_start
    
    def create_master_date_index(self, start_date: pd.Timestamp) -> pd.DatetimeIndex:
        """Create business day index from start date to present"""
        end_date = pd.to_datetime('2024-12-31')  # Conservative end date
        
        logger.info(f"Creating master date index from {start_date} to {end_date}")
        
        # Create business day index (excluding weekends)
        date_index = pd.bdate_range(start=start_date, end=end_date, freq='B')
        
        return date_index
    
    def merge_all_data(self, volatility_data: Dict, market_data: Dict, macro_data: Dict) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """Merge all data sources into a single dataframe"""
        logger.info("Merging all data sources...")
        
        # Determine optimal start date
        optimal_start = self.determine_optimal_start_date(market_data)
        
        # Create master dataframe with business day index
        master_index = self.create_master_date_index(optimal_start)
        df = pd.DataFrame(index=master_index)
        df.index.name = 'date'
        df = df.reset_index()
        
        # Track merging statistics
        merge_stats = {'successful': [], 'failed': [], 'columns_added': 0}
        
        # Merge volatility data
        for key, data in volatility_data.items():
            try:
                before_cols = len(df.columns)
                df = pd.merge(df, data, on='date', how='left')
                after_cols = len(df.columns)
                merge_stats['successful'].append(key)
                merge_stats['columns_added'] += (after_cols - before_cols)
                logger.debug(f"Merged {key}: added {after_cols - before_cols} columns")
            except Exception as e:
                logger.error(f"Failed to merge {key}: {e}")
                merge_stats['failed'].append(key)
        
        # Merge market data
        for key, data in market_data.items():
            try:
                before_cols = len(df.columns)
                df = pd.merge(df, data, on='date', how='left')
                after_cols = len(df.columns)
                merge_stats['successful'].append(key)
                merge_stats['columns_added'] += (after_cols - before_cols)
                logger.debug(f"Merged {key}: added {after_cols - before_cols} columns")
            except Exception as e:
                logger.error(f"Failed to merge {key}: {e}")
                merge_stats['failed'].append(key)
        
        # Merge macro data (using forward fill for monthly data)
        for key, data in macro_data.items():
            try:
                before_cols = len(df.columns)
                df = pd.merge(df, data, on='date', how='left')
                after_cols = len(df.columns)
                
                # Forward fill macro data (since it's often monthly)
                macro_cols = [col for col in data.columns if col != 'date']
                for col in macro_cols:
                    if col in df.columns:
                        df[col] = df[col].ffill()
                
                merge_stats['successful'].append(key)
                merge_stats['columns_added'] += (after_cols - before_cols)
                logger.debug(f"Merged {key}: added {after_cols - before_cols} columns")
            except Exception as e:
                logger.error(f"Failed to merge {key}: {e}")
                merge_stats['failed'].append(key)
        
        logger.info(f"Merge complete: {df.shape}")
        logger.info(f"Successfully merged: {len(merge_stats['successful'])} datasets")
        logger.info(f"Total columns added: {merge_stats['columns_added']}")
        if merge_stats['failed']:
            logger.warning(f"Failed to merge: {merge_stats['failed']}")
        
        return df, optimal_start
    
    def calculate_realized_volatility(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """Calculate realized volatility from SP500 returns"""
        if 'SP500_RETURNS' not in df.columns:
            logger.warning("SP500_RETURNS not available for realized volatility calculation")
            return pd.Series(index=df.index, dtype=float)
        
        # Calculate rolling standard deviation and annualize
        realized_vol = df['SP500_RETURNS'].rolling(window=window, min_periods=10).std() * np.sqrt(252) * 100
        return realized_vol
    
    def calculate_prediction_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the top 5 prediction targets identified in your research"""
        logger.info("Calculating prediction targets...")
        
        df = df.copy()
        targets_created = []
        
        # 1. VIX Term Structure Slope (VIX3M - VIX)
        if 'VIX3M' in df.columns and 'VIX' in df.columns:
            df['TARGET_VIX_TERM_STRUCTURE'] = df['VIX3M'] - df['VIX']
            targets_created.append('TARGET_VIX_TERM_STRUCTURE')
            logger.info("✓ Calculated VIX Term Structure target")
        
        # 2. Realized vs Implied Volatility Spread (VIX - 30-day realized vol)
        if 'VIX' in df.columns:
            realized_vol_30d = self.calculate_realized_volatility(df, window=30)
            df['REALIZED_VOL_30D'] = realized_vol_30d
            df['TARGET_REALIZED_VS_IMPLIED'] = df['VIX'] - realized_vol_30d
            targets_created.append('TARGET_REALIZED_VS_IMPLIED')
            logger.info("✓ Calculated Realized vs Implied Volatility target")
        
        # 3. Cross-Asset Volatility Correlation (30-day rolling correlation VIX vs OVX)
        if 'VIX' in df.columns and 'OVX' in df.columns:
            # Calculate 30-day rolling correlation
            correlation = df['VIX'].rolling(window=30, min_periods=15).corr(df['OVX'])
            df['TARGET_CROSS_ASSET_CORRELATION'] = correlation
            targets_created.append('TARGET_CROSS_ASSET_CORRELATION')
            logger.info("✓ Calculated Cross-Asset Volatility Correlation target")
        
        # 4. Volatility Dispersion (VIX - average of individual stock volatilities)
        individual_vols = ['VXAPL', 'VXEEM', 'VXAZN']  # Available individual vol indices
        available_individual_vols = [vol for vol in individual_vols if vol in df.columns]
        
        if len(available_individual_vols) > 0 and 'VIX' in df.columns:
            # Calculate average of available individual volatilities
            df['AVG_INDIVIDUAL_VOL'] = df[available_individual_vols].mean(axis=1)
            df['TARGET_VOLATILITY_DISPERSION'] = df['VIX'] - df['AVG_INDIVIDUAL_VOL']
            targets_created.append('TARGET_VOLATILITY_DISPERSION')
            logger.info(f"✓ Calculated Volatility Dispersion target using {available_individual_vols}")
        
        # 5. Volatility-of-Volatility Regime Indicator (VVIX/VIX ratio)
        if 'VVIX' in df.columns and 'VIX' in df.columns:
            # Avoid division by zero
            df['TARGET_VOL_OF_VOL_RATIO'] = df['VVIX'] / df['VIX'].replace(0, np.nan)
            targets_created.append('TARGET_VOL_OF_VOL_RATIO')
            logger.info("✓ Calculated Vol-of-Vol Ratio target")
        
        logger.info(f"Created {len(targets_created)} prediction targets: {targets_created}")
        return df
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and features"""
        logger.info("Adding technical features...")
        
        df = df.copy()
        
        # VIX-based features
        if 'VIX' in df.columns:
            # Moving averages
            df['VIX_MA_5'] = df['VIX'].rolling(window=5, min_periods=1).mean()
            df['VIX_MA_20'] = df['VIX'].rolling(window=20, min_periods=1).mean()
            df['VIX_MA_60'] = df['VIX'].rolling(window=60, min_periods=1).mean()
            
            # Volatility of VIX
            df['VIX_VOLATILITY_20D'] = df['VIX'].rolling(window=20, min_periods=10).std()
            
            # VIX percentile ranks (using your expected column names)
            df['VIX_PCT_RANK_63'] = df['VIX'].rolling(window=63, min_periods=20).rank(pct=True)
            df['VIX_PCT_RANK_252'] = df['VIX'].rolling(window=252, min_periods=50).rank(pct=True)
            
            # Rate of change
            df['VIX_ROC_5'] = df['VIX'].pct_change(5)
            df['VIX_ROC_20'] = df['VIX'].pct_change(20)
        
        # Add calendar features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        # Add lagged features for key volatility indices
        vol_indices = ['VIX', 'VVIX', 'OVX', 'GVZ']
        for idx in vol_indices:
            if idx in df.columns:
                for lag in [1, 2, 5, 10]:
                    df[f'{idx}_LAG_{lag}'] = df[idx].shift(lag)
        
        logger.info("Technical features added successfully")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values with detailed tracking"""
        logger.info("Handling missing values...")
        
        df = df.copy()
        initial_missing = df.isnull().sum()
        total_initial_missing = initial_missing.sum()
        
        logger.info(f"Initial missing values: {total_initial_missing}")
        
        # Create detailed missing value report
        missing_report = {
            'initial_missing': initial_missing.to_dict(),
            'strategies_applied': [],
            'final_missing': {},
            'columns_with_remaining_nulls': []
        }
        
        # Strategy 1: Forward fill for all financial time series
        financial_cols = [col for col in df.columns if col not in ['date']]
        strategy_1_filled = 0
        
        for col in financial_cols:
            if col in df.columns:
                before_fill = df[col].isnull().sum()
                # Forward fill
                df[col] = df[col].ffill()
                # Backward fill for initial NaNs
                df[col] = df[col].bfill()
                after_fill = df[col].isnull().sum()
                strategy_1_filled += (before_fill - after_fill)
        
        missing_report['strategies_applied'].append({
            'strategy': 'Forward/Backward Fill',
            'values_filled': strategy_1_filled
        })
        
        # Strategy 2: Use VIX as proxy for missing volatility indices
        strategy_2_filled = 0
        if 'VIX' in df.columns:
            vol_proxies = {
                'VXAPL': 1.2,    # Individual stock vol typically higher
                'VXAZN': 1.2,
                'VXEEM': 1.5,    # Emerging market vol higher
                'OVX': 1.8,      # Oil vol typically much higher
                'GVZ': 0.9,      # Gold vol similar to equity
                'VXN': 1.1,      # NASDAQ vol slightly higher
                'VXD': 0.95,     # Dow vol slightly lower
                'RVX': 1.3       # Russell vol higher
            }
            
            for vol_idx, multiplier in vol_proxies.items():
                if vol_idx in df.columns:
                    before_proxy = df[vol_idx].isnull().sum()
                    mask = df[vol_idx].isnull()
                    if mask.any():
                        df.loc[mask, vol_idx] = df.loc[mask, 'VIX'] * multiplier
                        after_proxy = df[vol_idx].isnull().sum()
                        filled_count = before_proxy - after_proxy
                        strategy_2_filled += filled_count
                        if filled_count > 0:
                            logger.info(f"Used VIX proxy for {filled_count} values in {vol_idx}")
        
        missing_report['strategies_applied'].append({
            'strategy': 'VIX Proxy Fill',
            'values_filled': strategy_2_filled
        })
        
        # Strategy 3: Linear interpolation for remaining gaps
        strategy_3_filled = 0
        for col in financial_cols:
            if col in df.columns:
                before_interp = df[col].isnull().sum()
                if before_interp > 0:
                    df[col] = df[col].interpolate(method='linear')
                    after_interp = df[col].isnull().sum()
                    strategy_3_filled += (before_interp - after_interp)
        
        missing_report['strategies_applied'].append({
            'strategy': 'Linear Interpolation',
            'values_filled': strategy_3_filled
        })
        
        # Strategy 4: Final fill with median for any remaining NaNs
        strategy_4_filled = 0
        for col in financial_cols:
            if col in df.columns:
                before_median = df[col].isnull().sum()
                if before_median > 0:
                    median_val = df[col].median()
                    if not pd.isna(median_val):
                        df[col] = df[col].fillna(median_val)
                        after_median = df[col].isnull().sum()
                        strategy_4_filled += (before_median - after_median)
        
        missing_report['strategies_applied'].append({
            'strategy': 'Median Fill',
            'values_filled': strategy_4_filled
        })
        
        # Final missing value assessment
        final_missing = df.isnull().sum()
        total_final_missing = final_missing.sum()
        
        missing_report['final_missing'] = final_missing.to_dict()
        missing_report['columns_with_remaining_nulls'] = final_missing[final_missing > 0].index.tolist()
        
        logger.info(f"Final missing values: {total_final_missing}")
        logger.info(f"Reduced missing values by: {total_initial_missing - total_final_missing}")
        
        if total_final_missing > 0:
            logger.warning(f"Remaining nulls in columns: {missing_report['columns_with_remaining_nulls']}")
        else:
            logger.info("🎉 All missing values handled successfully!")
        
        return df, missing_report
    
    def verify_expected_columns(self, df: pd.DataFrame) -> Dict:
        """Verify that all expected columns are present"""
        
        expected_columns = [
            'date', 'VIX_OPEN', 'VIX_HIGH', 'VIX_LOW', 'VIX_CLOSE', 'VIX', 'VVIX',
            'VIX3M_OPEN', 'VIX3M_HIGH', 'VIX3M_LOW', 'VIX3M_CLOSE', 'VIX3M',
            'VIX9D_OPEN', 'VIX9D_HIGH', 'VIX9D_LOW', 'VIX9D_CLOSE', 'VIX9D',
            'OVX', 'GVZ', 'VXN_OPEN', 'VXN_HIGH', 'VXN_LOW', 'VXN_CLOSE', 'VXN',
            'VXD_OPEN', 'VXD_HIGH', 'VXD_LOW', 'VXD_CLOSE', 'VXD',
            'RVX_OPEN', 'RVX_HIGH', 'RVX_LOW', 'RVX_CLOSE', 'RVX',
            'VXAPL_OPEN', 'VXAPL_HIGH', 'VXAPL_LOW', 'VXAPL_CLOSE', 'VXAPL',
            'VXAZN_OPEN', 'VXAZN_HIGH', 'VXAZN_LOW', 'VXAZN_CLOSE', 'VXAZN',
            'VXEEM_OPEN', 'VXEEM_HIGH', 'VXEEM_LOW', 'VXEEM_CLOSE', 'VXEEM',
            'SKEW', 'SP500', 'SP500_RETURNS', 'US1M', 'US3M', 'US6M', 'US1Y',
            'US2Y', 'US3Y', 'US5Y', 'US7Y', 'US10Y', 'US20Y', 'US30Y',
            'USD_EUR', 'USD_GBP', 'USD_JPY', 'USD_INDEX', 'TED_SPREAD',
            'HIGH_YIELD_SPREAD', 'FED_FUNDS', 'INFLATION_HEADLINE', 'INFLATION_CORE',
            'REPO_RATE', 'POLICY_UNCERTAINTY', 'VIX_MA_5', 'VIX_MA_20', 'VIX_MA_60',
            'VIX_PCT_RANK_63', 'VIX_PCT_RANK_252', 'VIX_ROC_5', 'VIX_ROC_20',
            'day_of_week', 'month', 'quarter', 'VIX_LAG_1', 'VIX_LAG_2', 'VIX_LAG_5',
            'VIX_LAG_10', 'VVIX_LAG_1', 'VVIX_LAG_2', 'VVIX_LAG_5', 'VVIX_LAG_10',
            'OVX_LAG_1', 'OVX_LAG_2', 'OVX_LAG_5', 'OVX_LAG_10', 'GVZ_LAG_1',
            'GVZ_LAG_2', 'GVZ_LAG_5', 'GVZ_LAG_10'
        ]
        
        present_columns = df.columns.tolist()
        missing_columns = [col for col in expected_columns if col not in present_columns]
        unexpected_columns = [col for col in present_columns if col not in expected_columns and not col.startswith('TARGET_')]
        
        verification_report = {
            'total_expected': len(expected_columns),
            'total_present': len(present_columns),
            'missing_columns': missing_columns,
            'unexpected_columns': unexpected_columns,
            'all_expected_present': len(missing_columns) == 0
        }
        
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
        if unexpected_columns:
            logger.info(f"Additional columns found: {unexpected_columns}")
        
        return verification_report
    
    def create_processed_dataset(self) -> Tuple[pd.DataFrame, Dict]:
        """Main method to create the processed dataset with comprehensive reporting"""
        logger.info("="*60)
        logger.info("🚀 CREATING PROCESSED VOLATILITY DATASET")
        logger.info("="*60)
        
        # Initialize comprehensive report
        processing_report = {
            'start_time': datetime.now().isoformat(),
            'data_loading': {},
            'merging': {},
            'missing_values': {},
            'column_verification': {},
            'final_stats': {}
        }
        
        # Load all data
        volatility_data = self.load_volatility_data()
        market_data = self.load_market_data()
        macro_data = self.load_macro_data()
        
        # Record loading stats
        processing_report['data_loading'] = {
            'volatility_files_loaded': len(volatility_data),
            'market_files_loaded': len(market_data),
            'macro_files_loaded': len(macro_data)
        }
        
        # Merge all data
        merged_df, optimal_start = self.merge_all_data(volatility_data, market_data, macro_data)
        processing_report['merging'] = {
            'optimal_start_date': optimal_start.isoformat(),
            'initial_shape': merged_df.shape,
            'initial_date_range': {
                'start': merged_df['date'].min().isoformat(),
                'end': merged_df['date'].max().isoformat()
            }
        }
        
        # Calculate prediction targets
        df_with_targets = self.calculate_prediction_targets(merged_df)
        
        # Add technical features
        df_with_features = self.add_technical_features(df_with_targets)
        
        # Handle missing values
        final_df, missing_report = self.handle_missing_values(df_with_features)
        processing_report['missing_values'] = missing_report
        
        # Verify expected columns
        column_verification = self.verify_expected_columns(final_df)
        processing_report['column_verification'] = column_verification
        
        # Final statistics
        processing_report['final_stats'] = {
            'final_shape': final_df.shape,
            'final_date_range': {
                'start': final_df['date'].min().isoformat(),
                'end': final_df['date'].max().isoformat(),
                'total_days': len(final_df)
            },
            'total_missing_values': int(final_df.isnull().sum().sum()),
            'missing_percentage': float((final_df.isnull().sum().sum() / (len(final_df) * len(final_df.columns))) * 100),
            'target_columns': [col for col in final_df.columns if col.startswith('TARGET_')],
            'processing_completed': True
        }
        
        logger.info("="*60)
        logger.info("✅ DATASET CREATION COMPLETED")
        logger.info(f"📊 Final dataset shape: {final_df.shape}")
        logger.info(f"📅 Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        logger.info(f"🎯 Prediction targets: {len(processing_report['final_stats']['target_columns'])}")
        logger.info(f"❌ Missing values: {processing_report['final_stats']['total_missing_values']} ({processing_report['final_stats']['missing_percentage']:.2f}%)")
        
        if not column_verification['all_expected_present']:
            logger.warning(f"⚠️  Missing expected columns: {len(column_verification['missing_columns'])}")
        else:
            logger.info("✅ All expected columns present!")
        
        logger.info("="*60)
        
        return final_df, processing_report
    
    def save_processed_data(self, df: pd.DataFrame, processing_report: Dict, filename: str = "processed_volatility_data.csv"):
        """Save the processed dataset with comprehensive reporting"""
        # Create processed data directory
        output_dir = self.data_dir / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        # Save main dataset
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved processed dataset to {output_path}")
        
        # Create detailed column mapping report
        column_report = {
            'volatility_indices': {
                'core_indices': [col for col in df.columns if col in ['VIX', 'VVIX', 'OVX', 'GVZ']],
                'term_structure': [col for col in df.columns if col in ['VIX9D', 'VIX', 'VIX3M']],
                'sector_indices': [col for col in df.columns if col in ['VXN', 'VXD', 'RVX']],
                'individual_stock': [col for col in df.columns if col in ['VXAPL', 'VXAZN', 'VXEEM']],
                'ohlc_data': [col for col in df.columns if any(ohlc in col for ohlc in ['_OPEN', '_HIGH', '_LOW', '_CLOSE'])]
            },
            'market_data': {
                'equity': [col for col in df.columns if 'SP500' in col],
                'treasury_yields': [col for col in df.columns if col.startswith('US') and col[2:].replace('M', '').replace('Y', '').isdigit()],
                'fx_rates': [col for col in df.columns if col.startswith('USD_')],
                'credit_spreads': [col for col in df.columns if 'SPREAD' in col]
            },
            'macro_indicators': [col for col in df.columns if col in ['FED_FUNDS', 'INFLATION_HEADLINE', 'INFLATION_CORE', 'REPO_RATE', 'POLICY_UNCERTAINTY']],
            'technical_features': {
                'moving_averages': [col for col in df.columns if '_MA_' in col],
                'percentile_ranks': [col for col in df.columns if '_PCT_RANK_' in col],
                'rate_of_change': [col for col in df.columns if '_ROC_' in col],
                'lagged_features': [col for col in df.columns if '_LAG_' in col]
            },
            'prediction_targets': [col for col in df.columns if col.startswith('TARGET_')],
            'calendar_features': [col for col in df.columns if col in ['day_of_week', 'month', 'quarter']]
        }
        
        # Combine processing report with column mapping
        comprehensive_report = {
            'processing_report': processing_report,
            'column_mapping': column_report,
            'data_quality': {
                'null_counts_by_column': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'basic_statistics': {
                    'vix_stats': df['VIX'].describe().to_dict() if 'VIX' in df.columns else None,
                    'sp500_returns_stats': df['SP500_RETURNS'].describe().to_dict() if 'SP500_RETURNS' in df.columns else None
                }
            },
            'answers_to_questions': {
                'do_we_have_all_ohlc_columns': len([col for col in df.columns if any(ohlc in col for ohlc in ['_OPEN', '_HIGH', '_LOW', '_CLOSE'])]) > 0,
                'sp500_returns_start_date': df['SP500_RETURNS'].first_valid_index() if 'SP500_RETURNS' in df.columns else None,
                'actual_start_date_used': processing_report['merging']['optimal_start_date'],
                'total_null_values': int(df.isnull().sum().sum()),
                'null_handling_strategies': [strategy['strategy'] for strategy in processing_report['missing_values']['strategies_applied']],
                'missing_expected_columns': processing_report['column_verification']['missing_columns']
            }
        }
        
        # Save comprehensive report
        report_path = output_dir / f"{filename.replace('.csv', '_comprehensive_report.json')}"
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        logger.info(f"📋 Saved comprehensive report to {report_path}")
        
        # Create a simple summary for quick reference
        quick_summary = {
            'dataset_shape': df.shape,
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'total_missing_values': int(df.isnull().sum().sum()),
            'expected_columns_present': processing_report['column_verification']['all_expected_present'],
            'prediction_targets_created': len([col for col in df.columns if col.startswith('TARGET_')]),
            'ohlc_data_preserved': len([col for col in df.columns if any(ohlc in col for ohlc in ['_OPEN', '_HIGH', '_LOW', '_CLOSE'])]) > 0
        }
        
        summary_path = output_dir / f"{filename.replace('.csv', '_quick_summary.json')}"
        with open(summary_path, 'w') as f:
            json.dump(quick_summary, f, indent=2, default=str)
        
        logger.info(f"⚡ Saved quick summary to {summary_path}")
        
        return output_path, report_path, summary_path

# Usage function
def create_processed_volatility_data():
    """Main function to create the processed volatility dataset"""
    processor = VolatilityDataProcessor2015(data_dir=".")
    
    # Create the processed dataset
    processed_df, processing_report = processor.create_processed_dataset()
    
    # Save the dataset
    data_path, report_path, summary_path = processor.save_processed_data(processed_df, processing_report)
    
    print(f"\n🎯 SUCCESS!")
    print(f"📊 Processed dataset saved to: {data_path}")
    print(f"📋 Comprehensive report: {report_path}")
    print(f"⚡ Quick summary: {summary_path}")
    print(f"📈 Dataset shape: {processed_df.shape}")
    print(f"📅 Date range: {processed_df['date'].min()} to {processed_df['date'].max()}")
    
    # Answer the user's specific questions
    print(f"\n🔍 ANSWERS TO YOUR QUESTIONS:")
    print(f"1. OHLC columns preserved: {len([col for col in processed_df.columns if any(ohlc in col for ohlc in ['_OPEN', '_HIGH', '_LOW', '_CLOSE'])]) > 0}")
    print(f"2. Total null values: {processed_df.isnull().sum().sum()}")
    print(f"3. SP500_RETURNS start date: {processed_df[processed_df['SP500_RETURNS'].notna()]['date'].min() if 'SP500_RETURNS' in processed_df.columns else 'Not available'}")
    print(f"4. Actual dataset start date: {processed_df['date'].min()}")
    
    # Show target columns created
    target_cols = [col for col in processed_df.columns if col.startswith('TARGET_')]
    print(f"\n🎯 Prediction targets created: {len(target_cols)}")
    for target in target_cols:
        valid_count = processed_df[target].count()
        print(f"   • {target}: {valid_count} valid values")
    
    return processed_df

if __name__ == "__main__":
    df = create_processed_volatility_data()