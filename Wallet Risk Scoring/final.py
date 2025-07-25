#!/usr/bin/env python3
"""
Complete Compound Protocol Wallet Risk Scoring System - FIXED VERSION
"""

import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import our previous modules
from load import load_wallet_addresses, get_wallet_transactions, filter_compound_transactions
from feature_engineering import extract_risk_features, analyze_compound_specific_risks
from risk_scoring import CompoundRiskScorer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompoundRiskAnalyzer:
    """Complete wallet risk analysis system for Compound protocol"""
    
    def __init__(self, etherscan_api_key: str):
        self.api_key = etherscan_api_key
        self.scorer = CompoundRiskScorer()
        self.wallets = []
        self.transaction_data = {}
        self.features_data = {}
        
    def load_wallets(self):
        """Load wallet addresses from CSV file"""
        logger.info("Loading wallet addresses...")
        self.wallets = load_wallet_addresses()
        logger.info(f"Loaded {len(self.wallets)} wallet addresses")
        
    def collect_transaction_data(self, max_wallets=None):
        """Collect transaction data for all wallets"""
        wallets_to_process = self.wallets[:max_wallets] if max_wallets else self.wallets
        logger.info(f"Collecting transaction data for {len(wallets_to_process)} wallets...")
        
        for i, wallet in enumerate(wallets_to_process):
            try:
                logger.info(f"Processing wallet {i+1}/{len(wallets_to_process)}: {wallet}")
                
                # Get all transactions for wallet
                all_transactions = get_wallet_transactions(wallet, self.api_key)
                
                # Filter only Compound transactions
                compound_transactions = filter_compound_transactions(all_transactions)
                
                self.transaction_data[wallet] = compound_transactions
                
                logger.info(f"Found {len(compound_transactions)} Compound transactions for {wallet}")
                
                # Rate limiting to avoid API limits
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error processing wallet {wallet}: {str(e)}")
                self.transaction_data[wallet] = []
                
        logger.info("Transaction data collection completed")
        
    def extract_features(self):
        """Extract risk features from transaction data"""
        logger.info("Extracting risk features...")
        
        for wallet, transactions in self.transaction_data.items():
            try:
                # Extract basic risk features
                basic_features = extract_risk_features(transactions)
                
                # Extract Compound-specific features
                compound_features = analyze_compound_specific_risks(transactions)
                
                # Combine all features
                all_features = {**basic_features, **compound_features}
                self.features_data[wallet] = all_features
                
            except Exception as e:
                logger.error(f"Error extracting features for {wallet}: {str(e)}")
                self.features_data[wallet] = self._get_default_features()
        
        logger.info(f"Feature extraction completed for {len(self.features_data)} wallets")
        
    def _get_default_features(self):
        """Get default features for wallets with errors"""
        return {
            'total_transactions': 0,
            'unique_contracts': 0,
            'total_volume_eth': 0,
            'avg_transaction_size': 0,
            'avg_time_between_txs': 0,
            'transaction_frequency_std': 0,
            'days_since_last_tx': 9999,
            'failed_tx_ratio': 0,
            'total_gas_spent': 0,
            'avg_gas_price': 0,
            'high_gas_tx_ratio': 0,
            'recent_activity_ratio': 0,
            'recent_volume_ratio': 0,
            'weekend_activity_ratio': 0,
            'night_activity_ratio': 0,
            'liquidation_events': 0,
            'borrow_repay_ratio': 0,
            'supply_withdraw_ratio': 0,
            'leverage_indicator': 0,
            'market_diversity': 0
        }
        
    def calculate_risk_scores(self):
        """Calculate risk scores for all wallets"""
        logger.info("Calculating risk scores...")
        
        # Convert features to DataFrame
        features_df = pd.DataFrame.from_dict(self.features_data, orient='index')
        
        # Calculate risk scores
        risk_scores = self.scorer.calculate_risk_scores(features_df)
        
        logger.info("Risk score calculation completed")
        return risk_scores
        
    def generate_summary_report(self, risk_scores):
        """Generate summary statistics and insights"""
        logger.info("Generating summary report...")
        
        summary = {
            'total_wallets': len(risk_scores),
            'avg_score': risk_scores['score'].mean(),
            'median_score': risk_scores['score'].median(),
            'low_risk_wallets': len(risk_scores[risk_scores['score'] < 300]),
            'medium_risk_wallets': len(risk_scores[(risk_scores['score'] >= 300) & (risk_scores['score'] < 700)]),
            'high_risk_wallets': len(risk_scores[risk_scores['score'] >= 700]),
            'highest_score': risk_scores['score'].max(),
            'lowest_score': risk_scores['score'].min()
        }
        
        return summary
        
    def save_results(self, risk_scores, filename='wallet_risk_scores.csv'):
        """Save results to CSV file"""
        logger.info(f"Saving results to {filename}...")
        
        # Sort by score (highest risk first)
        risk_scores_sorted = risk_scores.sort_values('score', ascending=False)
        
        # Save to CSV
        risk_scores_sorted.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        
        return risk_scores_sorted
        
    def run_complete_analysis(self, max_wallets=None, output_filename='wallet_risk_scores.csv'):
        """Run the complete risk analysis pipeline"""
        try:
            # Step 1: Load wallets
            self.load_wallets()
            
            # Step 2: Collect transaction data
            self.collect_transaction_data(max_wallets)
            
            # Step 3: Extract features
            self.extract_features()
            
            # Step 4: Calculate risk scores
            risk_scores = self.calculate_risk_scores()
            
            # Step 5: Generate summary
            summary = self.generate_summary_report(risk_scores)
            
            # Step 6: Save results
            final_results = self.save_results(risk_scores, output_filename)
            
            # Print summary
            logger.info("="*50)
            logger.info("ANALYSIS COMPLETE - SUMMARY REPORT")
            logger.info("="*50)
            logger.info(f"Total Wallets Analyzed: {summary['total_wallets']}")
            logger.info(f"Average Risk Score: {summary['avg_score']:.1f}")
            logger.info(f"Median Risk Score: {summary['median_score']:.1f}")
            logger.info(f"Low Risk Wallets (0-299): {summary['low_risk_wallets']}")
            logger.info(f"Medium Risk Wallets (300-699): {summary['medium_risk_wallets']}")
            logger.info(f"High Risk Wallets (700-1000): {summary['high_risk_wallets']}")
            logger.info(f"Score Range: {summary['lowest_score']} - {summary['highest_score']}")
            logger.info("="*50)
            
            return final_results, summary
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    
    # Configuration
    ETHERSCAN_API_KEY = "IFW4DAXKV5G1UNQ9QR6G3WFJN6E78FNAAB"
    MAX_WALLETS = 100  # Start with 10 wallets for testing
    OUTPUT_FILE = "wallet_risk_scores.csv"
    
    # FIXED: Check if API key is placeholder, not the actual key
    if ETHERSCAN_API_KEY == "YOUR_API_KEY_HERE" or not ETHERSCAN_API_KEY:
        logger.error("Please set your Etherscan API key in the ETHERSCAN_API_KEY variable")
        return
    
    logger.info(f"Starting analysis with API key: {ETHERSCAN_API_KEY[:10]}...")
    
    # Initialize analyzer
    analyzer = CompoundRiskAnalyzer(ETHERSCAN_API_KEY)
    
    # Run complete analysis
    try:
        results, summary = analyzer.run_complete_analysis(
            max_wallets=MAX_WALLETS,
            output_filename=OUTPUT_FILE
        )
        
        # Display  score for  risk wallets
        print("\n Scores for  Risk Wallets:")
        print(results.head(100).to_string(index=False))
        
        print(f"\nComplete results saved to: {OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()