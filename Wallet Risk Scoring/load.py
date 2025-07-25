import pandas as pd
import requests
import time

# Read wallet addresses from local CSV file
def load_wallet_addresses():
    """Load wallet addresses from the CSV file"""
    try:
        df = pd.read_csv('Wallet id - Sheet1.csv')
        wallets = df['wallet_id'].dropna().tolist()
        print(f"Loaded {len(wallets)} wallet addresses from CSV file")
        return wallets
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []

# Load all 100 wallet addresses
wallets = load_wallet_addresses()

# Compound contract addresses
compound_contracts = {
    'cETH': '0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5',
    'cDAI': '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643',
    'cUSDC': '0x39aa39c021dfbae8fac545936693ac917cf5e8a3',
    'cUSDT': '0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9',
    'cWBTC': '0xc11b1268c1a384e55c48c2391d8bb682b3e43f2e'
}

def get_wallet_transactions(address, api_key):
    """Fetch Compound transactions for a wallet"""
    url = "https://api.etherscan.io/api"
    params = {
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'desc',
        'apikey': api_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['result']
    return []

def filter_compound_transactions(transactions):
    """Filter only Compound protocol transactions"""
    compound_txs = []
    all_compound_addresses = set(compound_contracts.values())
    
    for tx in transactions:
        if tx['to'].lower() in [addr.lower() for addr in all_compound_addresses]:
            compound_txs.append(tx)
    
    return compound_txs


print(f"Setup complete. Ready to collect data for {len(wallets)} wallets")
print("Next: Run data collection with your Etherscan API key")