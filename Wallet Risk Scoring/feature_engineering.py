import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def extract_risk_features(wallet_transactions):
    """
    Extract risk-related features from wallet transactions
    
    Args:
        wallet_transactions: List of Compound transactions for a wallet
    
    Returns:
        dict: Feature dictionary for risk scoring
    """
    features = {}
    
    if not wallet_transactions:
        return get_default_features()
    
    # Convert transactions to DataFrame for easier analysis
    df = pd.DataFrame(wallet_transactions)
    df['timestamp'] = pd.to_datetime(df['timeStamp'], unit='s')
    df['value_eth'] = df['value'].astype(float) / 1e18
    df['gas_cost'] = (df['gasUsed'].astype(float) * df['gasPrice'].astype(float)) / 1e18
    
    # === LIQUIDITY RISK FEATURES ===
    features['total_transactions'] = len(df)
    features['unique_contracts'] = df['to'].nunique()
    features['total_volume_eth'] = df['value_eth'].sum()
    features['avg_transaction_size'] = df['value_eth'].mean() if len(df) > 0 else 0
    
    # === BEHAVIORAL RISK FEATURES ===
    # Transaction frequency patterns
    df_sorted = df.sort_values('timestamp')
    if len(df_sorted) > 1:
        time_diffs = df_sorted['timestamp'].diff().dt.total_seconds() / 3600  # hours
        features['avg_time_between_txs'] = time_diffs.mean()
        features['transaction_frequency_std'] = time_diffs.std()
    else:
        features['avg_time_between_txs'] = 0
        features['transaction_frequency_std'] = 0
    
    # Activity recency (days since last transaction)
    features['days_since_last_tx'] = (datetime.now() - df['timestamp'].max()).days
    
    # Failed transaction ratio
    failed_txs = df[df['isError'] == '1']
    features['failed_tx_ratio'] = len(failed_txs) / len(df) if len(df) > 0 else 0
    
    # === MARKET RISK FEATURES ===
    # Gas spending patterns (proxy for desperation/urgency)
    features['total_gas_spent'] = df['gas_cost'].sum()
    features['avg_gas_price'] = df['gasPrice'].astype(float).mean()
    features['high_gas_tx_ratio'] = len(df[df['gasPrice'].astype(float) > df['gasPrice'].astype(float).quantile(0.8)]) / len(df)
    
    # === TIME-BASED RISK FEATURES ===
    # Activity during different time periods
    recent_30d = df[df['timestamp'] > (datetime.now() - timedelta(days=30))]
    features['recent_activity_ratio'] = len(recent_30d) / len(df) if len(df) > 0 else 0
    features['recent_volume_ratio'] = recent_30d['value_eth'].sum() / df['value_eth'].sum() if df['value_eth'].sum() > 0 else 0
    
    # === COMPLEXITY RISK FEATURES ===
    # Transaction pattern complexity
    features['weekend_activity_ratio'] = len(df[df['timestamp'].dt.weekday >= 5]) / len(df) if len(df) > 0 else 0
    
    # Hour-based activity (night activity might indicate automation/bots)
    night_hours = df[(df['timestamp'].dt.hour >= 22) | (df['timestamp'].dt.hour <= 6)]
    features['night_activity_ratio'] = len(night_hours) / len(df) if len(df) > 0 else 0
    
    return features

def get_default_features():
    """Return default features for wallets with no transactions"""
    return {
        'total_transactions': 0,
        'unique_contracts': 0,
        'total_volume_eth': 0,
        'avg_transaction_size': 0,
        'avg_time_between_txs': 0,
        'transaction_frequency_std': 0,
        'days_since_last_tx': 9999,  # Very high risk
        'failed_tx_ratio': 0,
        'total_gas_spent': 0,
        'avg_gas_price': 0,
        'high_gas_tx_ratio': 0,
        'recent_activity_ratio': 0,
        'recent_volume_ratio': 0,
        'weekend_activity_ratio': 0,
        'night_activity_ratio': 0
    }

def analyze_compound_specific_risks(wallet_transactions):
    """
    Analyze Compound-specific risk patterns
    
    Args:
        wallet_transactions: List of Compound transactions
    
    Returns:
        dict: Compound-specific risk features
    """
    compound_features = {}
    
    if not wallet_transactions:
        return {
            'liquidation_events': 0,
            'borrow_repay_ratio': 0,
            'supply_withdraw_ratio': 0,
            'leverage_indicator': 0,
            'market_diversity': 0
        }
    
    # Analyze function calls (would require transaction input decoding)
    # For now, we'll use heuristics based on value and patterns
    
    df = pd.DataFrame(wallet_transactions)
    df['value_eth'] = df['value'].astype(float) / 1e18
    
    # Proxy indicators for Compound operations
    zero_value_txs = df[df['value_eth'] == 0]  # Likely borrow/repay operations
    non_zero_txs = df[df['value_eth'] > 0]     # Likely supply operations
    
    compound_features['liquidation_events'] = 0  # Would need to decode transaction data
    compound_features['borrow_repay_ratio'] = len(zero_value_txs) / len(df) if len(df) > 0 else 0
    compound_features['supply_withdraw_ratio'] = len(non_zero_txs) / len(df) if len(df) > 0 else 0
    
    # Market diversity (number of different Compound markets used)
    compound_features['market_diversity'] = df['to'].nunique()
    
    # Leverage indicator (high frequency of operations might indicate leverage)
    if len(df) > 10:
        compound_features['leverage_indicator'] = 1 if df['value_eth'].std() / df['value_eth'].mean() > 2 else 0
    else:
        compound_features['leverage_indicator'] = 0
    
    return compound_features



print("Step 2: Feature Engineering Framework Ready")
print("Features extracted:")
print("- Liquidity Risk: transaction volume, frequency, size")
print("- Behavioral Risk: timing patterns, failed transactions")
print("- Market Risk: gas spending, urgency indicators") 
print("- Compound-specific: borrow/supply patterns, leverage indicators")