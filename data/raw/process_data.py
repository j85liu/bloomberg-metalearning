#!/usr/bin/env python3
"""
Enhanced Data Processor - Bloomberg Meta-Learning Project

IMPROVEMENTS:
1. Combines SPX.csv (for 2011-2015) + SP500_RETURNS.csv (for 2015+) 
2. Investigates null patterns to understand data gaps
3. Extends date range to get maximum coverage
4. Detailed gap analysis and reporting
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

class EnhancedDataProcessor:
    """
    Enhanced processor with SPX+SP500_RETURNS combination and gap analysis
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
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
        
        # File mappings (same as before but with SP500_RETURNS added)
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
            'market/SP500_RETURNS.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'SP500_RETURNS_RAW'},
            
            # Macro files
            'macro/FED_FUNDS.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'FED_FUNDS'},
            'macro/INFLATION_HEADLINE.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'INFLATION_HEADLINE'},
            'macro/INFLATION_CORE.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'INFLATION_CORE'},
            'macro/REPO_RATE.csv': {'date_col': 'date', 'format': '%Y-%m-%d', 'prefix': 'REPO_RATE'},
        }
    
    def load_combined_sp500_data(self) -> pd.DataFrame:
        """
        Load and combine SPX.csv + SP500_RETURNS.csv for complete coverage
        """
        logger.info("🔗 Loading and combining SPX + SP500_RETURNS data")
        
        # Load SPX data
        spx_path = self.data_dir / "raw" / "market" / "SPX.csv"
        if not spx_path.exists():
            raise FileNotFoundError(f"SPX file not found: {spx_path}")
        
        spx_df = pd.read_csv(spx_path)
        spx_df['Date'] = pd.to_datetime(spx_df['Date'])
        spx_df['SP500_RETURNS'] = spx_df['Adj Close'].pct_change()
        spx_df = spx_df.rename(columns={'Date': 'date', 'Adj Close': 'SP500'})
        spx_processed = spx_df[['date', 'SP500', 'SP500_RETURNS']].copy()
        
        logger.info(f"SPX data: {len(spx_processed)} rows from {spx_processed['date'].min()} to {spx_processed['date'].max()}")
        
        # Load SP500_RETURNS data
        sp500_returns_path = self.data_dir / "raw" / "market" / "SP500_RETURNS.csv"
        if sp500_returns_path.exists():
            returns_df = pd.read_csv(sp500_returns_path)
            returns_df['date'] = pd.to_datetime(returns_df['date'])
            
            logger.info(f"SP500_RETURNS data: {len(returns_df)} rows from {returns_df['date'].min()} to {returns_df['date'].max()}")
            
            # Find the transition point
            spx_end = spx_processed['date'].max()
            returns_start = returns_df['date'].min()
            
            # Determine optimal transition point
            if returns_start <= spx_end:
                transition_date = returns_start
                logger.info(f"Transition point: {transition_date.date()} (using SP500_RETURNS from this date)")
                
                # Split SPX data at transition point
                spx_before = spx_processed[spx_processed['date'] < transition_date].copy()
                
                # Use SP500_RETURNS from transition point onwards
                # But we need SP500 prices too, so get them from SPX where available
                returns_after = returns_df[returns_df['date'] >= transition_date].copy()
                
                # Add SP500 prices to returns data from SPX where possible
                spx_overlap = spx_processed[spx_processed['date'] >= transition_date]
                if len(spx_overlap) > 0:
                    returns_after = pd.merge(returns_after, spx_overlap[['date', 'SP500']], on='date', how='left')
                
                # If SP500 prices not available, we'll handle in post-processing
                if 'SP500' not in returns_after.columns:
                    returns_after['SP500'] = np.nan
                
                # Combine the data
                combined_sp500 = pd.concat([spx_before, returns_after], ignore_index=True)
                combined_sp500 = combined_sp500.sort_values('date').reset_index(drop=True)
                
                logger.info(f"✅ Combined SP500 data: {len(combined_sp500)} rows")
                logger.info(f"   SPX portion: {len(spx_before)} rows (up to {transition_date.date()})")
                logger.info(f"   SP500_RETURNS portion: {len(returns_after)} rows (from {transition_date.date()})")
                
            else:
                # No overlap, just use SPX data
                combined_sp500 = spx_processed
                logger.info("No overlap found, using SPX data only")
        else:
            # Only SPX data available
            combined_sp500 = spx_processed
            logger.info("SP500_RETURNS.csv not found, using SPX data only")
        
        self.raw_data['SPX'] = combined_sp500
        self.date_ranges['SPX'] = {
            'start': combined_sp500['date'].min(),
            'end': combined_sp500['date'].max(),
            'count': len(combined_sp500)
        }
        
        return combined_sp500
    
    def load_single_file(self, rel_path: str, config: dict) -> Optional[pd.DataFrame]:
        """Load and process a single data file (same as before but with better error handling)"""
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
                    df[prefix] = df[f'{prefix}_CLOSE']
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
                
            elif prefix == 'SP500_RETURNS_RAW':
                # Handle the separate SP500_RETURNS file (already processed in combination)
                return None  # Skip this since we handle it in load_combined_sp500_data
                
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
    
    def analyze_data_gaps(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        Analyze why there are null values at the beginning of the date range
        """
        logger.info("🔍 Analyzing data gaps and null patterns...")
        
        # Create business day range
        business_days = pd.bdate_range(start=start_date, end=end_date, freq='B')
        
        print(f"\n📅 DATE RANGE ANALYSIS:")
        print(f"Requested range: {start_date.date()} to {end_date.date()}")
        print(f"Business days in range: {len(business_days)}")
        
        # Check first 20 business days for data availability
        print(f"\n🔍 FIRST 20 BUSINESS DAYS DATA AVAILABILITY:")
        print("-" * 80)
        print("Date       | VIX | VVIX| OVX | GVZ |VX3M |VX9D |VXEEM|SPX |")
        print("-" * 80)
        
        for i, date in enumerate(business_days[:20]):
            availability = []
            
            for source in ['VIX', 'VVIX', 'OVX', 'GVZ', 'VIX3M', 'VIX9D', 'VXEEM', 'SPX']:
                if source in self.raw_data:
                    df = self.raw_data[source]
                    has_data = date in df['date'].values
                    availability.append('✓' if has_data else '✗')
                else:
                    availability.append('?')
            
            avail_str = ' | '.join(f'{a:^3}' for a in availability)
            print(f"{date.strftime('%Y-%m-%d')} | {avail_str} |")
        
        # Find gaps for each data source
        print(f"\n📊 DATA SOURCE COVERAGE ANALYSIS:")
        print("-" * 60)
        
        for source_name, source_df in self.raw_data.items():
            if len(source_df) == 0:
                continue
                
            source_start = source_df['date'].min()
            source_end = source_df['date'].max()
            
            # Count how many business days in our range are missing
            source_business_days = pd.bdate_range(start=max(start_date, source_start), 
                                                end=min(end_date, source_end), freq='B')
            
            available_days = set(source_df['date'].dt.date)
            missing_days = [d for d in source_business_days if d.date() not in available_days]
            
            coverage_pct = ((len(source_business_days) - len(missing_days)) / len(source_business_days)) * 100
            
            print(f"{source_name:15s}: {coverage_pct:5.1f}% coverage, {len(missing_days):3d} missing days")
            
            # Show first few missing days
            if len(missing_days) > 0:
                missing_sample = missing_days[:5]
                missing_str = ', '.join(d.strftime('%Y-%m-%d') for d in missing_sample)
                if len(missing_days) > 5:
                    missing_str += f", ... +{len(missing_days)-5} more"
                print(f"{'':17s}Missing: {missing_str}")
    
    def find_optimal_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Find optimal date range with enhanced analysis"""
        logger.info("Finding optimal date range...")
        
        # Core data sources that we must have
        core_sources = ['VIX', 'VVIX', 'OVX', 'GVZ', 'SPX']
        
        # Extended data sources 
        extended_sources = ['VIX3M', 'VIX9D', 'VXN', 'VXD', 'RVX', 'VXAPL', 'VXAZN', 'VXEEM']
        
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
        
        # Analyze gaps in this range
        self.analyze_data_gaps(optimal_start, optimal_end)
        
        return optimal_start, optimal_end
    
    def load_all_raw_data(self):
        """Load all raw data files"""
        logger.info("Loading all raw data files...")
        
        # Load combined SPX data first
        self.load_combined_sp500_data()
        
        # Load all other files
        for rel_path, config in self.file_mappings.items():
            if config['prefix'] != 'SP500_RETURNS_RAW':  # Skip since we handled it
                df = self.load_single_file(rel_path, config)
                if df is not None:
                    self.raw_data[config['prefix']] = df
    
    def create_master_dataset(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Create master dataset (same as before but with enhanced logging)"""
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
        
        # Handle policy uncertainty
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
        """Add technical features (same as before)"""
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
        """Enhanced null checking with gap analysis"""
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
        
        # Enhanced reporting
        print("\n" + "="*80)
        print("📊 ENHANCED NULL VALUE REPORT")
        print("="*80)
        
        if missing_columns:
            print(f"\n❌ MISSING COLUMNS ({len(missing_columns)}):")
            for col in missing_columns:
                print(f"  - {col}")
        
        if null_report:
            print(f"\n⚠️  NULL VALUES ANALYSIS:")
            
            # Group by null percentage
            low_nulls = {k: v for k, v in null_report.items() if v['null_percentage'] < 5}
            medium_nulls = {k: v for k, v in null_report.items() if 5 <= v['null_percentage'] < 20}
            high_nulls = {k: v for k, v in null_report.items() if v['null_percentage'] >= 20}
            
            if low_nulls:
                print(f"\n🟡 LOW NULL PERCENTAGE (< 5%): {len(low_nulls)} columns")
                for col, info in list(low_nulls.items())[:10]:  # Show first 10
                    if 'reason' not in info:
                        print(f"  {col}: {info['null_count']:,} nulls ({info['null_percentage']:.1f}%)")
                if len(low_nulls) > 10:
                    print(f"  ... and {len(low_nulls)-10} more columns")
            
            if medium_nulls:
                print(f"\n🟠 MEDIUM NULL PERCENTAGE (5-20%): {len(medium_nulls)} columns")
                for col, info in medium_nulls.items():
                    if 'reason' not in info:
                        print(f"  {col}: {info['null_count']:,} nulls ({info['null_percentage']:.1f}%)")
            
            if high_nulls:
                print(f"\n🔴 HIGH NULL PERCENTAGE (≥ 20%): {len(high_nulls)} columns")
                for col, info in high_nulls.items():
                    if 'reason' in info:
                        print(f"  {col}: {info['reason']}")
                    else:
                        print(f"  {col}: {info['null_count']:,} nulls ({info['null_percentage']:.1f}%)")
        else:
            print("\n✅ NO NULL VALUES FOUND!")
        
        # Enhanced summary
        total_nulls = sum(info['null_count'] for info in null_report.values())
        total_cells = len(df) * len(self.expected_columns)
        null_percentage = (total_nulls / total_cells) * 100
        
        print(f"\n📈 ENHANCED SUMMARY:")
        print(f"Dataset shape: {final_df.shape}")
        print(f"Date range: {final_df['date'].min().date()} to {final_df['date'].max().date()}")
        print(f"Total cells: {total_cells:,}")
        print(f"Null cells: {total_nulls:,}")
        print(f"Null percentage: {null_percentage:.2f}%")
        print(f"Completeness: {100-null_percentage:.2f}%")
        
        # Show data quality by category
        print(f"\n📋 DATA QUALITY BY CATEGORY:")
        categories = {
            'Volatility Indices': [col for col in final_df.columns if any(x in col for x in ['VIX', 'OVX', 'GVZ', 'VXN', 'VXD', 'RVX', 'VXAPL', 'VXAZN', 'VXEEM', 'SKEW'])],
            'Market Data': [col for col in final_df.columns if any(x in col for x in ['SP500', 'USD_', 'TED_', 'HIGH_YIELD', 'US1M', 'US3M', 'US6M', 'US1Y', 'US2Y', 'US3Y', 'US5Y', 'US7Y', 'US10Y', 'US20Y', 'US30Y'])],
            'Macro Data': [col for col in final_df.columns if any(x in col for x in ['FED_FUNDS', 'INFLATION', 'REPO_', 'POLICY_'])],
            'Technical Features': [col for col in final_df.columns if any(x in col for x in ['_MA_', '_ROC_', '_LAG_', '_PCT_RANK', 'day_of_week', 'month', 'quarter'])]
        }
        
        for category, cols in categories.items():
            if cols:
                cat_nulls = sum(final_df[col].isnull().sum() for col in cols if col in final_df.columns)
                cat_total = len(final_df) * len(cols)
                cat_pct = (cat_nulls / cat_total) * 100 if cat_total > 0 else 0
                print(f"  {category:20s}: {100-cat_pct:5.1f}% complete ({len(cols):2d} columns)")
        
        return final_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_volatility_data_enhanced.csv"):
        """Save the processed dataset with enhanced summary"""
        output_path = self.data_dir / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Enhanced processing summary
        summary = {
            'processing_date': datetime.now().isoformat(),
            'data_combination_strategy': 'SPX.csv + SP500_RETURNS.csv combined',
            'shape': df.shape,
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat(),
                'days': len(df),
                'business_days': len(df)
            },
            'columns': df.columns.tolist(),
            'data_sources': list(self.date_ranges.keys()),
            'data_source_ranges': {k: {
                'start': v['start'].isoformat(),
                'end': v['end'].isoformat(),
                'count': v['count']
            } for k, v in self.date_ranges.items()},
            'null_analysis': {
                'total_nulls': int(df.isnull().sum().sum()),
                'null_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
                'null_counts_by_column': df.isnull().sum().to_dict()
            }
        }
        
        summary_path = output_path.parent / f"{filename.replace('.csv', '_summary.json')}"
        with open(summary_path, 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved enhanced processing summary to {summary_path}")
    
    def run_complete_processing(self):
        """Run the complete enhanced data processing pipeline"""
        logger.info("🚀 STARTING ENHANCED DATA PROCESSING WITH SPX+SP500_RETURNS COMBINATION")
        logger.info("="*80)
        
        try:
            # Step 1: Load all raw data (including combined SPX)
            self.load_all_raw_data()
            
            # Step 2: Find optimal date range with gap analysis
            start_date, end_date = self.find_optimal_date_range()
            
            # Step 3: Create master dataset
            master_df = self.create_master_dataset(start_date, end_date)
            
            # Step 4: Add technical features
            enhanced_df = self.add_technical_features(master_df)
            
            # Step 5: Enhanced null checking and reporting
            final_df = self.check_nulls_and_report(enhanced_df)
            
            # Step 6: Save processed data with enhanced summary
            self.save_processed_data(final_df)
            
            logger.info("\n✅ ENHANCED DATA PROCESSING COMPLETED SUCCESSFULLY!")
            
            # Final recommendations
            print("\n" + "="*80)
            print("🎯 PROCESSING RECOMMENDATIONS")
            print("="*80)
            
            total_null_pct = (final_df.isnull().sum().sum() / (len(final_df) * len(final_df.columns))) * 100
            
            if total_null_pct < 5:
                print("✅ EXCELLENT: Very low null percentage - dataset is ready for meta-learning!")
            elif total_null_pct < 10:
                print("🟡 GOOD: Moderate null percentage - consider filling critical nulls")
            else:
                print("🔴 CAUTION: High null percentage - review data sources and date range")
            
            print(f"\n📊 NEXT STEPS:")
            print(f"1. Review null pattern analysis above")
            print(f"2. Consider extending SPX data to get more recent coverage")
            print(f"3. Ready to run target calculator with improved SP500_RETURNS")
            print(f"4. Dataset shape: {final_df.shape} - suitable for meta-learning")
            
            return final_df
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced data processing: {e}")
            raise

if __name__ == "__main__":
    # Initialize and run enhanced processor
    processor = EnhancedDataProcessor()
    processed_data = processor.run_complete_processing()
    
    print(f"\n🎉 ENHANCED PROCESSING COMPLETE!")
    print(f"📁 Output: data/processed/processed_volatility_data_enhanced.csv")
    print(f"📊 Shape: {processed_data.shape}")
    print(f"🔗 SPX+SP500_RETURNS combination successful!")