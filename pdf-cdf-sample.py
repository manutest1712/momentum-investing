import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu = 160    # Mean height in cm
sigma = 7   # Typical standard deviation for height

# Range: Generate values for the x-axis (from -4 to +4 standard deviations)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)

# b. Probability Density Function (PDF)
pdf_values = norm.pdf(x, mu, sigma)

# c. Cumulative Distribution Function (CDF)
cdf_values = norm.cdf(x, mu, sigma)

# d. Plotting PDF and CDF in the same diagram
fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary Axis: PDF
color_pdf = 'tab:blue'
ax1.set_xlabel('Height (cm)')
ax1.set_ylabel('Probability Density (PDF)', color=color_pdf)
ax1.plot(x, pdf_values, color=color_pdf, lw=2, label='PDF (Likelihood)')
ax1.fill_between(x, pdf_values, color=color_pdf, alpha=0.1)
ax1.tick_params(axis='y', labelcolor=color_pdf)

# Secondary Axis: CDF
ax2 = ax1.twinx()
color_cdf = 'tab:red'
ax2.set_ylabel('Cumulative Probability (CDF)', color=color_cdf)
ax2.plot(x, cdf_values, color=color_cdf, lw=2, linestyle='--', label='CDF (Percentile)')
ax2.tick_params(axis='y', labelcolor=color_cdf)
ax2.set_ylim(0, 1.05)

# Styling and Labels
plt.title(f"Normal Distribution of Women's Height")
ax1.grid(True, linestyle=':', alpha=0.6)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()