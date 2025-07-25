import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class CompoundRiskScorer:
    """
    Risk scoring system for Compound protocol wallets
    Score range: 0-1000 (0 = lowest risk, 1000 = highest risk)
    """
    
    def __init__(self):
        self.feature_weights = {
            # Liquidity Risk (30% weight)
            'total_transactions': 0.05,
            'total_volume_eth': 0.10,
            'avg_transaction_size': 0.05,
            'unique_contracts': 0.10,
            
            # Behavioral Risk (25% weight)
            'days_since_last_tx': 0.10,
            'failed_tx_ratio': 0.05,
            'transaction_frequency_std': 0.05,
            'night_activity_ratio': 0.05,
            
            # Market Risk (20% weight)
            'total_gas_spent': 0.05,
            'high_gas_tx_ratio': 0.10,
            'recent_activity_ratio': 0.05,
            
            # Compound-Specific Risk (25% weight)
            'borrow_repay_ratio': 0.10,
            'leverage_indicator': 0.10,
            'market_diversity': 0.05
        }
        
        self.scaler = MinMaxScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def clean_data(self, features_df):
        """
        Clean data by handling NaN, inf, and other problematic values
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            DataFrame with cleaned features
        """
        cleaned_df = features_df.copy()
        
        # Replace infinite values with NaN first
        cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with appropriate defaults
        for column in cleaned_df.columns:
            if column in ['days_since_last_tx']:
                # For days since last transaction, use high risk value
                cleaned_df[column] = cleaned_df[column].fillna(9999)
            elif column in ['total_transactions', 'unique_contracts', 'total_volume_eth', 
                           'total_gas_spent', 'liquidation_events', 'market_diversity']:
                # For count/volume features, use 0
                cleaned_df[column] = cleaned_df[column].fillna(0)
            elif column in ['failed_tx_ratio', 'borrow_repay_ratio', 'supply_withdraw_ratio',
                           'high_gas_tx_ratio', 'recent_activity_ratio', 'weekend_activity_ratio',
                           'night_activity_ratio', 'recent_volume_ratio']:
                # For ratio features, use 0
                cleaned_df[column] = cleaned_df[column].fillna(0)
            elif column in ['leverage_indicator']:
                # For binary indicators, use 0
                cleaned_df[column] = cleaned_df[column].fillna(0)
            else:
                # For other features, use median or 0
                median_val = cleaned_df[column].median()
                if pd.isna(median_val):
                    cleaned_df[column] = cleaned_df[column].fillna(0)
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(median_val)
        
        # Ensure all values are numeric
        for column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce').fillna(0)
            
        # Cap extreme values to prevent scoring issues
        for column in cleaned_df.columns:
            if column == 'days_since_last_tx':
                cleaned_df[column] = cleaned_df[column].clip(0, 9999)
            else:
                # Cap other values at 99th percentile to handle outliers
                upper_cap = cleaned_df[column].quantile(0.99)
                if upper_cap > 0:
                    cleaned_df[column] = cleaned_df[column].clip(0, upper_cap)
        
        return cleaned_df
        
    def normalize_features(self, features_df):
        """
        Normalize features to 0-1 range with risk-based transformations
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            DataFrame with normalized features
        """
        # First clean the data
        cleaned_df = self.clean_data(features_df)
        normalized_df = cleaned_df.copy()
        
        # Apply risk-based transformations
        for feature in cleaned_df.columns:
            if feature in self.get_inverse_risk_features():
                # For features where higher value = lower risk, invert them
                max_val = cleaned_df[feature].max()
                if max_val > 0:
                    normalized_df[feature] = 1 - (cleaned_df[feature] / max_val)
                else:
                    normalized_df[feature] = 0.5  # Neutral risk
            elif feature in self.get_threshold_features():
                # For features with optimal ranges
                normalized_df[feature] = self.apply_threshold_scoring(cleaned_df[feature], feature)
            else:
                # Standard normalization for direct risk features
                max_val = cleaned_df[feature].max()
                if max_val > 0:
                    normalized_df[feature] = cleaned_df[feature] / max_val
                else:
                    normalized_df[feature] = 0.0
        
        # Ensure all values are between 0 and 1
        normalized_df = normalized_df.clip(0, 1)
        
        # Final check for any remaining NaN or inf values
        normalized_df = normalized_df.fillna(0.5)  # Neutral risk for any remaining NaN
        
        return normalized_df
    
    def get_inverse_risk_features(self):
        """Features where higher values indicate lower risk"""
        return [
            'total_transactions',
            'total_volume_eth', 
            'unique_contracts',
            'market_diversity',
            'recent_activity_ratio'
        ]
    
    def get_threshold_features(self):
        """Features with optimal ranges"""
        return [
            'days_since_last_tx',
            'transaction_frequency_std',
            'avg_transaction_size'
        ]
    
    def apply_threshold_scoring(self, values, feature_name):
        """Apply threshold-based scoring for specific features"""
        if feature_name == 'days_since_last_tx':
            # 0-7 days: low risk, 7-30: medium, 30+: high risk
            return np.where(values <= 7, 0.1,
                   np.where(values <= 30, 0.5, 
                   np.minimum(values / 365, 1.0)))  # Cap at 1 year
        
        elif feature_name == 'transaction_frequency_std':
            # Very erratic patterns indicate higher risk
            q95 = values.quantile(0.95) if len(values) > 0 else 1
            if q95 > 0:
                return np.minimum(values / q95, 1.0)
            else:
                return np.zeros_like(values)
        
        elif feature_name == 'avg_transaction_size':
            # Extremely large transactions might indicate higher risk
            median_val = values.median()
            if median_val > 0:
                return np.where(values > median_val * 10, 0.8, 0.3)
            else:
                return np.full_like(values, 0.3)
        
        return values
    
    def calculate_base_score(self, normalized_features):
        """
        Calculate base risk score using weighted features
        
        Args:
            normalized_features: DataFrame with normalized features
            
        Returns:
            Series with base scores (0-1 range)
        """
        base_scores = pd.Series(0.0, index=normalized_features.index)
        
        for feature, weight in self.feature_weights.items():
            if feature in normalized_features.columns:
                feature_contribution = normalized_features[feature] * weight
                # Clean any NaN values that might have slipped through
                feature_contribution = feature_contribution.fillna(0)
                base_scores += feature_contribution
        
        return base_scores.clip(0, 1)
    
    def apply_anomaly_detection(self, features_df, base_scores):
        """
        Apply anomaly detection to identify unusual patterns
        
        Args:
            features_df: Original features DataFrame
            base_scores: Base risk scores
            
        Returns:
            Series with anomaly-adjusted scores
        """
        # Select key features for anomaly detection
        anomaly_features = [
            'total_transactions', 'total_volume_eth', 'failed_tx_ratio',
            'days_since_last_tx', 'high_gas_tx_ratio'
        ]
        
        available_features = [f for f in anomaly_features if f in features_df.columns]
        
        if len(available_features) < 3 or len(features_df) < 2:
            return base_scores
        
        try:
            anomaly_data = features_df[available_features].fillna(0)
            
            # Additional cleaning for anomaly detection
            anomaly_data = anomaly_data.replace([np.inf, -np.inf], 0)
            
            # Check if we have enough variation in the data
            if anomaly_data.std().sum() == 0:
                return base_scores
            
            # Fit anomaly detector
            self.anomaly_detector.fit(anomaly_data)
            anomaly_scores = self.anomaly_detector.decision_function(anomaly_data)
            
            # Convert anomaly scores to risk multiplier (0.8 to 1.2)
            score_range = anomaly_scores.max() - anomaly_scores.min()
            if score_range > 0:
                anomaly_multiplier = 1.1 - (anomaly_scores - anomaly_scores.min()) / score_range * 0.3
            else:
                anomaly_multiplier = pd.Series(1.0, index=base_scores.index)
            
            return base_scores * anomaly_multiplier
            
        except Exception as e:
            print(f"Anomaly detection failed: {e}, using base scores")
            return base_scores
    
    def apply_compound_specific_adjustments(self, scores, features_df):
        """
        Apply Compound protocol specific risk adjustments
        
        Args:
            scores: Current risk scores
            features_df: Features DataFrame
            
        Returns:
            Series with adjusted scores
        """
        adjusted_scores = scores.copy()
        
        # High leverage penalty
        if 'leverage_indicator' in features_df.columns:
            high_leverage_mask = features_df['leverage_indicator'] == 1
            adjusted_scores[high_leverage_mask] *= 1.2
        
        # No activity penalty (zombie wallets)
        if 'total_transactions' in features_df.columns:
            no_activity_mask = features_df['total_transactions'] == 0
            adjusted_scores[no_activity_mask] = 0.9  # High risk for inactive wallets
        
        # Liquidation risk (high borrow activity without recent supply)
        if 'borrow_repay_ratio' in features_df.columns and 'recent_activity_ratio' in features_df.columns:
            high_borrow_mask = (features_df['borrow_repay_ratio'] > 0.7) & (features_df['recent_activity_ratio'] < 0.3)
            adjusted_scores[high_borrow_mask] *= 1.15
        
        return adjusted_scores.clip(0, 1)
    
    def convert_to_score_range(self, normalized_scores):
        """
        Convert normalized scores (0-1) to final score range (0-1000)
        
        Args:
            normalized_scores: Scores in 0-1 range
            
        Returns:
            Series with scores in 0-1000 range
        """
        # Clean scores before conversion
        clean_scores = normalized_scores.fillna(0.5).clip(0, 1)
        final_scores = (clean_scores * 1000).round()
        
        # Ensure all scores are valid integers
        return final_scores.astype(int)
    
    def calculate_risk_scores(self, features_df):
        """
        Main method to calculate risk scores for all wallets
        
        Args:
            features_df: DataFrame with extracted features for all wallets
            
        Returns:
            DataFrame with wallet addresses and risk scores
        """
        # Handle missing features
        for feature in self.feature_weights.keys():
            if feature not in features_df.columns:
                features_df[feature] = 0.5  # Neutral risk for missing features
        
        # Normalize features
        normalized_features = self.normalize_features(features_df)
        
        # Calculate base scores
        base_scores = self.calculate_base_score(normalized_features)
        
        # Apply anomaly detection
        anomaly_adjusted_scores = self.apply_anomaly_detection(features_df, base_scores)
        
        # Apply Compound-specific adjustments
        final_scores = self.apply_compound_specific_adjustments(anomaly_adjusted_scores, features_df)
        
        # Convert to 0-1000 range
        risk_scores = self.convert_to_score_range(final_scores)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'wallet_id': features_df.index,
            'score': risk_scores
        })
        
        return results
    
    def get_risk_explanation(self, wallet_features, score):
        """
        Generate risk explanation for a specific wallet
        
        Args:
            wallet_features: Features for a single wallet
            score: Risk score for the wallet
            
        Returns:
            dict with risk explanation
        """
        explanation = {
            'score': score,
            'risk_level': 'Low' if score < 300 else 'Medium' if score < 700 else 'High',
            'key_factors': []
        }
        
        # Identify key risk factors
        if wallet_features.get('days_since_last_tx', 0) > 90:
            explanation['key_factors'].append('Long inactivity period')
            
        if wallet_features.get('failed_tx_ratio', 0) > 0.1:
            explanation['key_factors'].append('High failed transaction rate')
            
        if wallet_features.get('leverage_indicator', 0) == 1:
            explanation['key_factors'].append('High leverage activity detected')
            
        if wallet_features.get('total_transactions', 0) == 0:
            explanation['key_factors'].append('No transaction history')
        
        return explanation

print("Fixed Risk Scoring Module Ready")