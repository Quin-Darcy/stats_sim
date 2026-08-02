import numpy as np
import matplotlib.pyplot as plt

d = np.genfromtxt('num_replicates_vs_ci_width_delta_jitter_sample_size_132_sims_100_cl_95.csv',
                  delimiter=',', names=True)
x, y = d['num_replicates'], d['ci_width_delta']

c = np.median(y * np.sqrt(x))
tol = 0.01
crossing = (c / tol) ** 2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, lw=0.4, alpha=0.35, color='gray', label='measured span')
ax.plot(x, c / np.sqrt(x), lw=1.8, color='crimson',
        label=f'fitted: {c:.3f} / sqrt(n)')
ax.plot(x, 1.0 / np.sqrt(x), lw=1.4, ls='-.', color='tab:blue',
        label='reference: 1 / sqrt(n)')
ax.axhline(tol, ls='--', lw=1, color='black', label=f'tolerance {tol}')
ax.axvline(crossing, ls=':', lw=1, color='black',
           label=f'crossing ≈ {crossing:.0f}')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Number of Replicates used in Bootstrap'); ax.set_ylabel('95th Percentile CI Width Delta')
ax.set_title('CI Width Delta vs Bootstrap Replicates')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
fig.savefig('num_replicates_vs_ci_width_delta.png', dpi=150, bbox_inches='tight')

slope = np.polyfit(np.log(x), np.log(y), 1)[0]
print(f'empirical log-log slope: {slope:.3f}')
