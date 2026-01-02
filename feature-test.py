import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# 1. Download Federal Bank data
# -----------------------------
ticker = "FEDERALBNK.NS"

df = yf.download(
    ticker,
    period="1y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

# -----------------------------
# 2. Compute daily log returns
# -----------------------------
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
returns = df["log_return"].dropna()

print(returns)

# -----------------------------
# 6. Annualized volatility
# -----------------------------
annual_vol = returns.std() * np.sqrt(252)
print(f"\nAnnualized Volatility: {annual_vol:.2%}")

# -----------------------------
# 3. Histogram + Normal curve
# -----------------------------
plt.figure(figsize=(8,4))
plt.hist(returns, bins=50, density=True, alpha=0.6, label="Log returns")

x = np.linspace(returns.min(), returns.max(), 200)
plt.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
         linewidth=2, label="Normal PDF")

plt.title("Federal Bank Daily Log Returns")
plt.legend()
plt.show()

# -----------------------------
# 4. Q-Q plot
# -----------------------------
stats.probplot(returns, dist="norm", plot=plt)
plt.title("Q-Q Plot: Federal Bank Log Returns")
plt.show()

# -----------------------------
# 5. Normality tests
# -----------------------------
shapiro_stat, shapiro_p = stats.shapiro(returns)
jb_stat, jb_p = stats.jarque_bera(returns)

print("Shapiro-Wilk Test:")
print(f"  Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")

print("\nJarque-Bera Test:")
print(f"  Statistic = {jb_stat:.4f}, p-value = {jb_p:.4f}")



# -----------------------------
# 7. Interpretation
# -----------------------------
if shapiro_p < 0.05 or jb_p < 0.05:
    print("\nConclusion: Returns are NOT normally distributed.")
else:
    print("\nConclusion: Cannot reject normality assumption.")
