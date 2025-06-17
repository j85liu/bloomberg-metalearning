#!/usr/bin/env python3
"""
Comprehensive Data Processor for Bloomberg Meta-Learning Project

Processes all raw data files into clean processed_volatility_data.csv
- Uses SPX.csv for complete S&P 500 returns
- Finds optimal date range where all data is available
- Reports any null values instead of filling them
- Creates exact column format specified
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ComprehensiveDataProcessor:
    """
    Process all raw data into the exact format specified
    """
    
    def __init__(self, data_dir: str = "data", spx_file: str = "data/raw/market/SPX.csv"):
        """
        Initialize processor
        
        Args:
            data_dir: Path to data directory
            spx_file: Path to SPX data file
        """
        self.data_dir = Path(data_dir)
        self.spx_file = Path(spx_file)
        self.raw_data = {}
        self.date_ranges = {}
        
        # Define expected column format
        self.expected_columns = [
            'date', 'VIX_OPEN', 'VIX_HIGH', 'VIX_LOW', 'VIX_CLOSE', 'VIX', 'VVIX',
            'VIX3M_OPEN', 'VIX3M_HIGH', 'VIX3M_LOW', 'VIX3M_CLOSE', 'VIX3M',
            'VIX9D_OPEN', 'VIX9D_HIGH', 'VIX9D_LOW', 'VIX9D_CLOSE', 'VIX9D',
            'OVX', 'GVZ', 'VXN_OPEN', 'VXN_HIGH', 'VXN_LOW', 'VXN_CLOSE', 'VXN',
            'VXD_OPEN', 'VXD_HIGH', 'VXD_LOW', 'VXD_CLOSE', 'VXD',
            'RVX_OPEN', 'RVX_HIGH', 'RVX_LOW', 'RVX_CLOSE', 'RVX',
            'VXAPL_OPEN', 'VXAPL_HIGH', 'VXAPL_LOW', 'VXAPL_CLOSE', 'VXAPL',
            'VXAZN_OPEN', 'VXAZN_HIGH', 'VXAZN_LOW', 'VXAZN_CLOSE', 'VXAZN',
            'VXEEM_OPEN', 'VXEEM_HIGH', 'VXEEM_LOW', 'VXEEM_CLOSE', 'VXEEM',
            'SKEW', 'SP500', 'SP500_RETURNS',
            'US1M', 'US3M', 'US6M', 'US1Y', 'US2Y', 'US3Y', 'US5Y', 'US7Y', 'US10Y', 'US20Y', 'US30Y',
            'USD_EUR', 'USD_GBP', 'USD_JPY', 'USD_INDEX', 'TED_SPREAD', 'HIGH_YIELD_SPREAD',
            'FED_FUNDS', 'INFLATION_HEADLINE', 'INFLATION_CORE', 'REPO_RATE', 'POLICY_UNCERTAINTY',
            'VIX_MA_5', 'VIX_MA_20', 'VIX_MA_60', 'VIX_PCT_RANK_63', 'VIX_PCT_RANK_252',
            'VIX_ROC_5', 'VIX_ROC_20', 'day_of_week', 'month', 'quarter',
            'VIX_LAG_1', 'VIX_LAG_2', 'VIX_LAG_5', 'VIX_LAG_10',
            'VVIX_LAG_1', 'VVIX_LAG_2', 'VVIX_LAG_5', 'VVIX_LAG_10',
            'OVX_LAG_1', 'OVX_LAG_2', 'OVX_LAG_5', 'OVX_LAG_10',
            'GVZ_LAG_1', 'GVZ_LAG_2', 'GVZ_LAG_5', 'GVZ_LAG_10'
        ]
        
        # Define file mappings
        self.file_mappings = {
            # Volatility files
            'volatility/VIX_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'VIX'},
            'volatility/VVIX_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'VVIX'},
            'volatility/VIX3M_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'VIX3M'},
            'volatility/VIX9D_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'VIX9D'},
            'volatility/OVX_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'OVX'},
            'volatility/GVZ_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'GVZ'},
            'volatility/VXN_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'VXN'},
            'volatility/VXD_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'VXD'},
            'volatility/RVX_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'RVX'},
            'volatility/VXAPL_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'VXAPL'},
            'volatility/VXAZN_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'VXAZN'},
            'volatility/VXEEM_History.csv': {'date_col': 'DATE', 'format': '%m/%d/%Y', 'prefix': 'VXEEM'},
            'volatility/SKEW_History.csv': {'date_col': 'DATE', 'format': '%Y-%m-%d', 'prefix': 'SKEW'},
            
            # Market files
            'market/us_treasury_yields_daily.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'TREASURY'},
            'market/USD_EUR.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'USD_EUR'},
            'market/USD_GBP.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'USD_GBP'},
            'market/USD_JPY.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'USD_JPY'},
            'market/USD_INDEX.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'USD_INDEX'},
            'market/TED_SPREAD.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'TED_SPREAD'},
            'market/HIGH_YIELD_SPREAD.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'HIGH_YIELD_SPREAD'},
            
            # Macro files
            'macro/FED_FUNDS.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'FED_FUNDS'},
            'macro/INFLATION_HEADLINE.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'INFLATION_HEADLINE'},
            'macro/INFLATION_CORE.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'INFLATION_CORE'},
            'macro/REPO_RATE.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'REPO_RATE'},
        }
    
    def load_spx_data(self) -> pd.DataFrame:
        """Load and process SPX data for S&P 500 returns"""
        logger.info(f"Loading SPX data from {self.spx_file}")
        
        if not self.spx_file.exists():
            raise FileNotFoundError(f"SPX file not found: {self.spx_file}")
        
        spx_df = pd.read_csv(self.spx_file)
        spx_df['Date'] = pd.to_datetime(spx_df['Date'])
        
        # Calculate returns from Adj Close
        spx_df['SP500_RETURNS'] = spx_df['Adj Close'].pct_change()
        
        # Rename columns to match expected format
        spx_df = spx_df.rename(columns={
            'Date': 'date',
            'Adj Close': 'SP500'
        })
        
        # Keep only what we need
        spx_processed = spx_df[['date', 'SP500', 'SP500_RETURNS']].copy()
        
        logger.info(f"SPX data: {len(spx_processed)} rows from {spx_processed['date'].min()} to {spx_processed['date'].max()}")
        
        self.raw_data['SPX'] = spx_processed
        self.date_ranges['SPX'] = {
            'start': spx_processed['date'].min(),
            'end': spx_processed['date'].max(),
            'count': len(spx_processed)
        }
        
        return spx_processed
    
    def load_single_file(self, rel_path: str, config: dict) -> Optional[pd.DataFrame]:
        """Load and process a single data file"""
        file_path = self.data_dir / "raw" / rel_path
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            
            # Parse date column
            date_col = config['date_col']
            if date_col not in df.columns:
                logger.error(f"Date column '{date_col}' not found in {rel_path}")
                return None
            
            df[date_col] = pd.to_datetime(df[date_col], format=config['format'])
            df = df.rename(columns={date_col: 'date'})
            df = df.sort_values('date').reset_index(drop=True)
            
            # Handle different file types
            prefix = config['prefix']
            
            if prefix in ['VIX', 'VIX3M', 'VIX9D', 'VXN', 'VXD', 'RVX', 'VXAPL', 'VXAZN', 'VXEEM']:
                # OHLC volatility files
                if all(col in df.columns for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE']):
                    df = df.rename(columns={
                        'OPEN': f'{prefix}_OPEN',
                        'HIGH': f'{prefix}_HIGH',
                        'LOW': f'{prefix}_LOW',
                        'CLOSE': f'{prefix}_CLOSE'
                    })
                    df[prefix] = df[f'{prefix}_CLOSE']  # Main column is close price
                    processed_df = df[['date', f'{prefix}_OPEN', f'{prefix}_HIGH', f'{prefix}_LOW', f'{prefix}_CLOSE', prefix]]
                else:
                    logger.error(f"OHLC columns not found in {rel_path}")
                    return None
                    
            elif prefix in ['VVIX', 'OVX', 'GVZ', 'SKEW']:
                # Single value files
                value_col = [col for col in df.columns if col != 'date'][0]
                df = df.rename(columns={value_col: prefix})
                processed_df = df[['date', prefix]]
                
            elif prefix == 'TREASURY':
                # Treasury yields file
                yield_cols = ['US1M', 'US3M', 'US6M', 'US1Y', 'US2Y', 'US3Y', 'US5Y', 'US7Y', 'US10Y', 'US20Y', 'US30Y']
                available_cols = ['date'] + [col for col in yield_cols if col in df.columns]
                processed_df = df[available_cols]
                
            else:
                # Single value market/macro files
                value_col = [col for col in df.columns if col != 'date'][0]
                df = df.rename(columns={value_col: prefix})
                processed_df = df[['date', prefix]]
            
            # Store date range info
            self.date_ranges[prefix] = {
                'start': processed_df['date'].min(),
                'end': processed_df['date'].max(),
                'count': len(processed_df)
            }
            
            logger.info(f"Loaded {prefix}: {len(processed_df)} rows from {processed_df['date'].min().date()} to {processed_df['date'].max().date()}")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error loading {rel_path}: {e}")
            return None
    
    def load_all_raw_data(self):
        """Load all raw data files"""
        logger.info("Loading all raw data files...")
        
        # Load SPX data first
        self.load_spx_data()
        
        # Load all other files
        for rel_path, config in self.file_mappings.items():
            df = self.load_single_file(rel_path, config)
            if df is not None:
                self.raw_data[config['prefix']] = df
    
    def find_optimal_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Find the optimal date range where all important data is available"""
        logger.info("Finding optimal date range...")
        
        # Core data sources that we must have
        core_sources = ['VIX', 'VVIX', 'OVX', 'GVZ', 'SPX']
        
        # Extended data sources (nice to have but can live without)
        extended_sources = ['VIX3M', 'VIX9D', 'VXN', 'VXD', 'RVX', 'VXAPL', 'VXAZN', 'VXEEM']
        
        # Find latest start date among core sources
        core_starts = []
        for source in core_sources:
            if source in self.date_ranges:
                core_starts.append(self.date_ranges[source]['start'])
            else:
                logger.error(f"Core source {source} not available!")
        
        if not core_starts:
            raise ValueError("No core data sources available!")
        
        # Find latest start date among ALL sources (including extended)
        all_starts = []
        available_sources = []
        
        for source in core_sources + extended_sources:
            if source in self.date_ranges:
                all_starts.append(self.date_ranges[source]['start'])
                available_sources.append(source)
        
        # Use the latest start date to ensure all data is available
        optimal_start = max(all_starts)
        
        # Find earliest end date among available sources
        all_ends = []
        for source in available_sources:
            all_ends.append(self.date_ranges[source]['end'])
        
        optimal_end = min(all_ends)
        
        logger.info(f"Optimal date range: {optimal_start.date()} to {optimal_end.date()}")
        logger.info(f"Available sources: {available_sources}")
        
        # Show which sources limit our date range
        limiting_start_sources = [s for s in available_sources if self.date_ranges[s]['start'] == optimal_start]
        limiting_end_sources = [s for s in available_sources if self.date_ranges[s]['end'] == optimal_end]
        
        logger.info(f"Start date limited by: {limiting_start_sources}")
        logger.info(f"End date limited by: {limiting_end_sources}")
        
        return optimal_start, optimal_end
    
    def create_master_dataset(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Create the master dataset with all columns in the correct format"""
        logger.info(f"Creating master dataset from {start_date.date()} to {end_date.date()}")
        
        # Create business day date range
        date_range = pd.bdate_range(start=start_date, end=end_date, freq='B')
        master_df = pd.DataFrame({'date': date_range})
        
        logger.info(f"Created master date index with {len(master_df)} business days")
        
        # Merge all data sources
        for source_name, source_df in self.raw_data.items():
            logger.info(f"Merging {source_name}...")
            master_df = pd.merge(master_df, source_df, on='date', how='left')
        
        # Handle special cases for macro data (forward fill monthly data)
        monthly_cols = ['FED_FUNDS', 'INFLATION_HEADLINE', 'INFLATION_CORE']
        for col in monthly_cols:
            if col in master_df.columns:
                master_df[col] = master_df[col].fillna(method='ffill')
                logger.info(f"Forward-filled monthly data for {col}")
        
        # Handle policy uncertainty (need special processing)
        if 'POLICY_UNCERTAINTY' not in master_df.columns:
            logger.info("Processing policy uncertainty data...")
            policy_file = self.data_dir / "raw" / "macro" / "policy_uncertainty_monthly.csv"
            if policy_file.exists():
                policy_df = pd.read_csv(policy_file)
                policy_df['date'] = pd.to_datetime(policy_df[['Year', 'Month']].assign(day=1))
                policy_df = policy_df.rename(columns={'News_Based_Policy_Uncert_Index': 'POLICY_UNCERTAINTY'})
                policy_df = policy_df[['date', 'POLICY_UNCERTAINTY']]
                
                master_df = pd.merge(master_df, policy_df, on='date', how='left')
                master_df['POLICY_UNCERTAINTY'] = master_df['POLICY_UNCERTAINTY'].fillna(method='ffill')
                logger.info("Added and forward-filled policy uncertainty data")
        
        return master_df
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features and derived columns"""
        logger.info("Adding technical features...")
        
        # VIX moving averages
        if 'VIX' in df.columns:
            df['VIX_MA_5'] = df['VIX'].rolling(window=5, min_periods=1).mean()
            df['VIX_MA_20'] = df['VIX'].rolling(window=20, min_periods=1).mean()
            df['VIX_MA_60'] = df['VIX'].rolling(window=60, min_periods=1).mean()
            
            # Percentile ranks
            df['VIX_PCT_RANK_63'] = df['VIX'].rolling(window=63, min_periods=1).rank(pct=True)
            df['VIX_PCT_RANK_252'] = df['VIX'].rolling(window=252, min_periods=1).rank(pct=True)
            
            # Rate of change
            df['VIX_ROC_5'] = df['VIX'].pct_change(5)
            df['VIX_ROC_20'] = df['VIX'].pct_change(20)
        
        # Calendar features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Lag features
        lag_cols = ['VIX', 'VVIX', 'OVX', 'GVZ']
        for col in lag_cols:
            if col in df.columns:
                for lag in [1, 2, 5, 10]:
                    df[f'{col}_LAG_{lag}'] = df[col].shift(lag)
        
        logger.info("Technical features added")
        return df
    
    def check_nulls_and_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for null values and report them"""
        logger.info("Checking for null values...")
        
        # Ensure we have all expected columns in the right order
        final_df = pd.DataFrame()
        final_df['date'] = df['date']
        
        missing_columns = []
        null_report = {}
        
        for col in self.expected_columns[1:]:  # Skip 'date' column
            if col in df.columns:
                final_df[col] = df[col]
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    null_report[col] = {
                        'null_count': null_count,
                        'null_percentage': (null_count / len(df)) * 100
                    }
            else:
                final_df[col] = np.nan
                missing_columns.append(col)
                null_report[col] = {
                    'null_count': len(df),
                    'null_percentage': 100.0,
                    'reason': 'Column missing from data'
                }
        
        # Report findings
        print("\n" + "="*80)
        print("📊 NULL VALUE REPORT")
        print("="*80)
        
        if missing_columns:
            print(f"\n❌ MISSING COLUMNS ({len(missing_columns)}):")
            for col in missing_columns:
                print(f"  - {col}")
        
        if null_report:
            print(f"\n⚠️  NULL VALUES FOUND:")
            for col, info in null_report.items():
                if 'reason' in info:
                    print(f"  {col}: {info['reason']}")
                else:
                    print(f"  {col}: {info['null_count']:,} nulls ({info['null_percentage']:.1f}%)")
        else:
            print("\n✅ NO NULL VALUES FOUND!")
        
        total_nulls = sum(info['null_count'] for info in null_report.values())
        total_cells = len(df) * len(self.expected_columns)
        null_percentage = (total_nulls / total_cells) * 100
        
        print(f"\n📈 SUMMARY:")
        print(f"Total cells: {total_cells:,}")
        print(f"Null cells: {total_nulls:,}")
        print(f"Null percentage: {null_percentage:.2f}%")
        print(f"Dataset shape: {final_df.shape}")
        print(f"Date range: {final_df['date'].min().date()} to {final_df['date'].max().date()}")
        
        return final_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_volatility_data.csv"):
        """Save the processed dataset"""
        output_path = self.data_dir / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Save processing summary
        summary = {
            'processing_date': datetime.now().isoformat(),
            'shape': df.shape,
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat(),
                'days': len(df)
            },
            'columns': df.columns.tolist(),
            'data_sources': list(self.date_ranges.keys()),
            'null_counts': df.isnull().sum().to_dict()
        }
        
        summary_path = output_path.parent / f"{filename.replace('.csv', '_summary.json')}"
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved processing summary to {summary_path}")
    
    def run_complete_processing(self):
        """Run the complete data processing pipeline"""
        logger.info("🚀 STARTING COMPREHENSIVE DATA PROCESSING")
        logger.info("="*60)
        
        try:
            # Step 1: Load all raw data
            self.load_all_raw_data()
            
            # Step 2: Find optimal date range
            start_date, end_date = self.find_optimal_date_range()
            
            # Step 3: Create master dataset
            master_df = self.create_master_dataset(start_date, end_date)
            
            # Step 4: Add technical features
            enhanced_df = self.add_technical_features(master_df)
            
            # Step 5: Check nulls and create final format
            final_df = self.check_nulls_and_report(enhanced_df)
            
            # Step 6: Save processed data
            self.save_processed_data(final_df)
            
            logger.info("\n✅ DATA PROCESSING COMPLETED SUCCESSFULLY!")
            return final_df
            
        except Exception as e:
            logger.error(f"❌ Error in data processing: {e}")
            raise

if __name__ == "__main__":
    # Initialize and run processor
    processor = ComprehensiveDataProcessor()
    processed_data = processor.run_complete_processing()
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"📁 Output: data/processed/processed_volatility_data.csv")
    print(f"📊 Shape: {processed_data.shape}")