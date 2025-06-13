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

class VolatilityDataProcessor:
    """
    Comprehensive data preprocessing pipeline for volatility forecasting meta-learning framework.
    
    Handles all CBOE volatility indices, macro data, market data, and event data.
    Standardizes dates, handles missing values, and creates aligned datasets.
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
                'dir': 'volatility',
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
                'dir': 'macro',
                'files': {
                    'FED_FUNDS': 'FED_FUNDS.csv',
                    'INFLATION_HEADLINE': 'INFLATION_HEADLINE.csv',
                    'INFLATION_CORE': 'INFLATION_CORE.csv',
                    'REPO_RATE': 'REPO_RATE.csv',
                    'POLICY_UNCERTAINTY': 'policy_uncertainty_monthly.csv'
                }
            },
            'market': {
                'dir': 'market',
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
                'dir': 'events',
                'files': {
                    'FOMC_EVENTS': 'fomc_major_events.csv',
                    'MARKET_EVENTS': 'major_market_events.csv',
                    'VIX_METHODOLOGY': 'vix_methodology_changes.csv'
                }
            }
        }
    
    def standardize_date_column(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Standardize date column to datetime format
        
        Args:
            df: DataFrame with date column
            date_col: Name of the date column
            
        Returns:
            DataFrame with standardized date column
        """
        df = df.copy()
        
        # Try different date formats
        date_formats = [
            '%m/%d/%Y',    # MM/DD/YYYY (like VIX data)
            '%Y-%m-%d',    # YYYY-MM-DD (like SKEW data)
            '%m/%d/%y',    # MM/DD/YY
            '%Y/%m/%d',    # YYYY/MM/DD
            '%d/%m/%Y',    # DD/MM/YYYY
        ]
        
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
        """
        Process all volatility index files
        
        Returns:
            Dictionary of processed volatility DataFrames
        """
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
                
                # Identify date column (usually 'DATE' or 'date')
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
        """
        Process macroeconomic data files
        
        Returns:
            Dictionary of processed macro DataFrames
        """
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
        """
        Process market data files
        
        Returns:
            Dictionary of processed market DataFrames
        """
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
    
    def process_event_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process event data files
        
        Returns:
            Dictionary of processed event DataFrames
        """
        logger.info("Processing event data...")
        event_data = {}
        
        events_dir = self.data_dir / self.data_categories['events']['dir']
        
        for key, filename in self.data_categories['events']['files'].items():
            file_path = events_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # Standardize date column
                date_col = 'date' if 'date' in df.columns else df.columns[0]
                df = self.standardize_date_column(df, date_col)
                
                # Sort by date
                df = df.sort_values('date').reset_index(drop=True)
                
                # Store date range info
                self.date_ranges[key] = {
                    'start': df['date'].min(),
                    'end': df['date'].max(),
                    'count': len(df)
                }
                
                event_data[key] = df
                logger.info(f"Processed {key}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        return event_data
    
    def create_master_date_index(self, start_date: str = None, end_date: str = None) -> pd.DatetimeIndex:
        """
        Create a master date index focusing on the main volatility datasets
        
        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            DatetimeIndex for aligning all datasets
        """
        # Focus on key volatility datasets for date range
        key_datasets = ['VIX', 'VVIX', 'OVX', 'GVZ']  # Most important for meta-learning
        
        # Get date ranges from key datasets only
        key_starts = []
        key_ends = []
        
        for dataset in key_datasets:
            if dataset in self.date_ranges:
                key_starts.append(self.date_ranges[dataset]['start'])
                key_ends.append(self.date_ranges[dataset]['end'])
        
        if not key_starts:
            # Fallback to all datasets
            key_starts = [info['start'] for info in self.date_ranges.values()]
            key_ends = [info['end'] for info in self.date_ranges.values()]
        
        # Use reasonable date range - later start date, earlier end date for overlap
        # But don't be too restrictive - find a good balance
        if start_date:
            common_start = pd.to_datetime(start_date)
        else:
            # Use the latest start date among key volatility datasets (ensures we have data)
            common_start = max(key_starts)
            # But not later than 2015 to have sufficient history
            earliest_reasonable = pd.to_datetime('2015-01-01')
            if common_start > earliest_reasonable:
                # Use VIX as baseline since it has the longest history
                if 'VIX' in self.date_ranges:
                    common_start = max(earliest_reasonable, self.date_ranges['VIX']['start'])
                else:
                    common_start = earliest_reasonable
        
        if end_date:
            common_end = pd.to_datetime(end_date)
        else:
            # Use the earliest end date to ensure data availability
            common_end = min(key_ends)
            # But not earlier than 2020 to have recent data
            latest_reasonable = pd.to_datetime('2020-01-01')
            if common_end < latest_reasonable:
                common_end = min([pd.to_datetime('2024-12-31')] + key_ends)
        
        logger.info(f"Creating master date index from {common_start} to {common_end}")
        
        # Ensure start is before end
        if common_start >= common_end:
            logger.warning(f"Start date {common_start} is not before end date {common_end}")
            # Use VIX data range as fallback
            if 'VIX' in self.date_ranges:
                common_start = self.date_ranges['VIX']['start']
                common_end = self.date_ranges['VIX']['end']
                logger.info(f"Using VIX date range as fallback: {common_start} to {common_end}")
            else:
                raise ValueError("Cannot create valid date range")
        
        # Create business day index (Monday-Friday, excluding weekends)
        date_index = pd.bdate_range(start=common_start, end=common_end, freq='B')
        
        return date_index
    
    def align_and_merge_data(self, 
                           volatility_data: Dict[str, pd.DataFrame],
                           macro_data: Dict[str, pd.DataFrame],
                           market_data: Dict[str, pd.DataFrame],
                           master_date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align all datasets to the master date index and merge
        
        Args:
            volatility_data: Processed volatility data
            macro_data: Processed macro data  
            market_data: Processed market data
            master_date_index: Master date index for alignment
            
        Returns:
            Merged DataFrame with all data aligned
        """
        logger.info("Aligning and merging all datasets...")
        
        # Start with master date index
        merged_df = pd.DataFrame(index=master_date_index)
        merged_df.index.name = 'date'
        merged_df = merged_df.reset_index()
        
        # Merge volatility data
        for key, df in volatility_data.items():
            merged_df = pd.merge(merged_df, df, on='date', how='left')
            non_null_count = merged_df[key].notna().sum() if key in merged_df.columns else 0
            logger.info(f"Merged {key}: {non_null_count}/{len(merged_df)} non-null values")
        
        # Merge market data
        for key, df in market_data.items():
            merged_df = pd.merge(merged_df, df, on='date', how='left')
            
        # Merge macro data (forward fill for monthly data)
        for key, df in macro_data.items():
            merged_df = pd.merge(merged_df, df, on='date', how='left')
            # Forward fill macro data (monthly -> daily)
            if key in merged_df.columns:
                merged_df[key] = merged_df[key].ffill()  # Updated method
        
        return merged_df
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data with appropriate strategies
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing data...")
        
        df = df.copy()
        
        # Strategy 1: Forward fill for most financial time series
        financial_cols = [col for col in df.columns if col not in ['date']]
        
        for col in financial_cols:
            if col in df.columns:
                missing_before = df[col].isna().sum()
                
                # Forward fill first
                df[col] = df[col].ffill()  # Updated method
                
                # Backward fill for any remaining at the beginning
                df[col] = df[col].bfill()  # Updated method
                
                missing_after = df[col].isna().sum()
                
                if missing_before > 0:
                    logger.info(f"{col}: {missing_before} -> {missing_after} missing values")
        
        return df
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical features like moving averages, volatility measures
        
        Args:
            df: DataFrame with price/volatility data
            
        Returns:
            DataFrame with additional technical features
        """
        logger.info("Adding technical features...")
        
        df = df.copy()
        
        # VIX-based features (if VIX exists)
        if 'VIX' in df.columns:
            # VIX percentiles for regime detection
            df['VIX_PCT_RANK_252'] = df['VIX'].rolling(252).rank(pct=True)
            df['VIX_PCT_RANK_63'] = df['VIX'].rolling(63).rank(pct=True)
            
            # VIX moving averages
            df['VIX_MA_5'] = df['VIX'].rolling(5).mean()
            df['VIX_MA_20'] = df['VIX'].rolling(20).mean()
            df['VIX_MA_60'] = df['VIX'].rolling(60).mean()
            
            # VIX momentum features
            df['VIX_ROC_5'] = df['VIX'].pct_change(5)
            df['VIX_ROC_20'] = df['VIX'].pct_change(20)
            
        # Add day of week and month features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Add lagged features for key volatility indices
        vol_indices = ['VIX', 'VVIX', 'OVX', 'GVZ']
        for vol_idx in vol_indices:
            if vol_idx in df.columns:
                for lag in [1, 2, 5, 10]:
                    df[f'{vol_idx}_LAG_{lag}'] = df[vol_idx].shift(lag)
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_volatility_data.csv"):
        """
        Save processed data to CSV
        
        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        output_path = self.data_dir / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Save data summary
        summary = {
            'shape': df.shape,
            'date_range': {
                'start': df['date'].min().isoformat() if len(df) > 0 else None,
                'end': df['date'].max().isoformat() if len(df) > 0 else None,
                'days': len(df)
            },
            'columns': df.columns.tolist(),
            'missing_data': df.isnull().sum().to_dict()
        }
        
        summary_path = output_path.parent / f"{filename.replace('.csv', '_summary.json')}"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved data summary to {summary_path}")
    
    def run_full_pipeline(self, 
                         start_date: str = None, 
                         end_date: str = None,
                         save_output: bool = True) -> pd.DataFrame:
        """
        Run the complete data preprocessing pipeline
        
        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)  
            save_output: Whether to save processed data
            
        Returns:
            Fully processed DataFrame ready for modeling
        """
        logger.info("="*60)
        logger.info("STARTING VOLATILITY DATA PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Process all data categories
        volatility_data = self.process_volatility_data()
        macro_data = self.process_macro_data()
        market_data = self.process_market_data()
        event_data = self.process_event_data()
        
        # Step 2: Create master date index
        master_date_index = self.create_master_date_index(start_date, end_date)
        
        # Step 3: Align and merge all data
        merged_df = self.align_and_merge_data(
            volatility_data, macro_data, market_data, master_date_index
        )
        
        # Step 4: Handle missing data
        processed_df = self.handle_missing_data(merged_df)
        
        # Step 5: Add technical features
        final_df = self.add_technical_features(processed_df)
        
        # Step 6: Save processed data
        if save_output:
            self.save_processed_data(final_df)
        
        # Final summary
        logger.info("="*60)
        logger.info("PREPROCESSING PIPELINE COMPLETED")
        logger.info(f"Final dataset shape: {final_df.shape}")
        if len(final_df) > 0:
            logger.info(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        logger.info(f"Total missing values: {final_df.isnull().sum().sum()}")
        logger.info("="*60)
        
        return final_df

# Usage example
if __name__ == "__main__":
    # Initialize processor
    processor = VolatilityDataProcessor(data_dir=".")
    
    # Run full pipeline
    df = processor.run_full_pipeline(
        start_date="2015-01-01",  # Reasonable start date
        end_date="2024-12-31",    # Recent end date
        save_output=True
    )
    
    # Display sample of processed data
    print("\nSample of processed data:")
    print(df.head())
    
    print(f"\nColumns available: {list(df.columns)}")
    print(f"Shape: {df.shape}")