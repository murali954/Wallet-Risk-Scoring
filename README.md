# 🧠 Wallet Risk Scoring Assignment

## 🎯 Problem Statement
Given 100 Ethereum wallet addresses, assess each wallet's risk based on its transaction history with the Compound V2/V3 protocol. The final output is a CSV file with the format:

| wallet_id | score |
|-----------|--------|
| 0xfaa0768bde629806739c3a4620656c5d26f44ef2 | 732 |

---

## 📥 Data Collection Method
- Used **Compound V2 Subgraph API** hosted on The Graph or similar Ethereum RPC providers (Alchemy, Infura)
- For each wallet address, collected data for:
  - `borrow` transactions
  - `repayBorrow` 
  - `liquidation` events
  - `supply` and `redeem` transactions
- Data fetched in batches and stored locally using Python

---

## 🔍 Feature Selection Rationale
The following features were engineered from raw transaction data:

1. **Total Borrows** - Higher borrows indicate more leverage, increasing risk
2. **Total Repays** - Responsible repayment behavior; higher repays imply lower risk
3. **Number of Liquidations** - Direct risk indicator; more liquidations suggest poor financial management
4. **Collateral to Borrow Ratio (CBR)** - Healthy CBR implies better buffer and lower risk
5. **Transaction Frequency** - Active users may manage risk better with regular rebalancing


---

## ⚖️ Normalization Method
- Features like borrow/repay amounts, liquidation counts, and CBR were **min-max normalized** to [0, 1] range
- Ensures all features contribute proportionately to the final score

---

## 📊 Scoring Method
Each normalized feature contributes to a final weighted score:

```python
score = (
    0.3 * (1 - normalized_liquidations) +
    0.25 * normalized_repay_score +
    0.2 * normalized_cbr +
    0.15 * normalized_transaction_frequency +
    0.1 * (1 - normalized_borrow_score)
)
```

The score is scaled from [0, 1] to [0, 1000].  
**Higher score = Lower risk | Lower score = Higher risk**

---

## 📌 Risk Indicators Justification

| Feature | Why It Indicates Risk |
|---------|----------------------|
| **Liquidation Count** | Direct risk indicator — more liquidations imply bad debt management |
| **Repayment History** | More consistent repayments show disciplined financial behavior |
| **Borrow Volume** | Higher borrow levels, if not matched by collateral, signal risk |
| **Collateral-to-Borrow Ratio** | Ensures safety buffer; lower CBR means higher liquidation risk |
| **Transaction Frequency** | Active users can rebalance portfolios better, managing risk proactively |

## Output overview
| Wallet ID                                    | Score |
| -------------------------------------------- | ----- |
| `0x8be38ea2b22b706aef313c2de81f7d179024dd30` | 900   |
| `0x06b51c6882b27cb05e712185531c1f74996dd988` | 826   |


