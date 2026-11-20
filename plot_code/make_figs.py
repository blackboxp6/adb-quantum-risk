import pandas as pd, numpy as np, matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9,
    'axes.grid': True,
    'grid.alpha': .3,
    'figure.dpi': 200,
    'legend.fontsize': 8
})

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
O = os.path.join(HERE, '..', 'outputs', 'figures') + os.sep

os.makedirs(O, exist_ok=True)

col = {
    'Monte Carlo': '#7f7f7f',
    'Stratified Monte Carlo': '#2ca02c',
    'Importance Sampling': '#ff7f0e',
    'Sobol': '#1f77b4',
    'QMLAE': '#d62728'
}

mk = {
    'Monte Carlo': 'o',
    'Stratified Monte Carlo': 's',
    'Importance Sampling': '^',
    'Sobol': 'D',
    'QMLAE': '*'
}

# Fig1
d = pd.read_csv(
    os.path.join(DATA, 'fair_classical_qmlae_comparison.csv')
)

T = [
    'Expected principal loss',
    'Both countries pay',
    'Portfolio loss exceeds 80m',
    'Expected stop-loss',
    'Both bonds exhausted'
]

tl = [
    'Expected principal loss',
    'Both countries pay',
    'Portfolio loss > USD 80m',
    'Expected stop-loss',
    'Both bonds exhausted'
]

fig, ax = plt.subplots(2, 3, figsize=(10, 6))

for i, (t, n) in enumerate(zip(T, tl)):
    a = ax.flat[i]

    for m in col:
        g = d[(d.Method == m) & (d.Metric == t)].sort_values('Budget')

        a.loglog(
            g.Budget,
            g['Relative MAE (%)'],
            marker=mk[m],
            color=col[m],
            label=m,
            ms=8 if m == 'QMLAE' else 5
        )

    g = d[
        (d.Method == 'Monte Carlo') &
        (d.Metric == t)
    ].sort_values('Budget')

    b = np.array([1024, 16384])

    y0 = d[
        (d.Method == 'QMLAE') &
        (d.Metric == t) &
        (d.Budget == 1024)
    ]['Relative MAE (%)'].iloc[0]

    a.loglog(
        b,
        y0 * (b / 1024) ** -.5,
        'k:',
        lw=1,
        label=r'slope $-1/2$ reference'
    )

    a.set_title(f'({"abcde"[i]}) {n}', fontsize=9)
    a.set_xlabel('Nominal budget $M$ (samples or shots)')
    a.set_ylabel('Relative MAE (%)')

    a.set_xticks([1024, 4096, 16384])
    a.set_xticklabels(['1,024', '4,096', '16,384'])
    a.minorticks_off()

h, l = ax.flat[0].get_legend_handles_labels()

ax.flat[5].axis('off')

ax.flat[5].legend(
    h,
    l,
    loc='center',
    frameon=False,
    title='100 repetitions per point\nideal-simulation QMLAE'
)

fig.tight_layout()

fig.savefig(O + 'fig1_error_vs_budget.pdf')
fig.savefig(O + 'fig1_error_vs_budget.png', dpi=200)

plt.close()


# Fig2
s = pd.read_csv(
    os.path.join(DATA, 'quantum_circuit_scaling.csv')
)

q2 = pd.read_csv(
    os.path.join(DATA, 'quantum_resource_results.csv')
)

q3 = pd.read_csv(
    os.path.join(DATA, 'three_country_qae_resource_scaling.csv')
)

sc = pd.read_csv(
    os.path.join(DATA, 'scaling_analysis.csv')
)

fig, ax = plt.subplots(2, 2, figsize=(8, 6.2))

a = ax[0, 0]

a.semilogy(
    s.countries,
    s.depth,
    'o-',
    c='#d62728',
    label='State-preparation depth'
)

a.semilogy(
    s.countries,
    s.total_gates,
    's--',
    c='#1f77b4',
    label='Total gates'
)

for x, y in zip(s.countries, s.depth):
    a.annotate(
        f'{y:,}',
        (x, y),
        textcoords='offset points',
        xytext=(-4, 6),
        fontsize=7,
        ha='right'
    )

a.set_xlabel('Number of countries $n$')
a.set_ylabel('Count')
a.set_xticks([2, 3, 4, 5])
a.set_title('(a) Depth and gates (synthetic, generic loading)', fontsize=9)
a.legend()

a = ax[0, 1]

a.semilogy(
    s.countries,
    s.cx_gates,
    'o-',
    c='#d62728',
    label='CX gates (measured)'
)

a.semilogy(
    s.countries,
    s.dimension,
    'k:',
    label=r'basis states $2^{q}$'
)

a.semilogy(
    s.countries,
    s.outcomes,
    'g--',
    label='Classical outcomes $10^n$'
)

a.set_xlabel('Number of countries $n$')
a.set_xticks([2, 3, 4, 5])
a.set_title('(b) CX gates track the register dimension', fontsize=9)
a.legend()

a = ax[1, 0]

a.plot(
    q2['Grover iterations'],
    q2['Circuit depth'],
    'o-',
    c='#1f77b4',
    label='Two-country (7 data qubits)'
)

a.plot(
    q3.qae_power,
    q3.total_depth,
    's-',
    c='#d62728',
    label='Three-country (10 data qubits)'
)

a.set_xlabel('Grover power $k$')
a.set_ylabel('Transpiled circuit depth')
a.set_title('(c) Depth of $Q^k A$ circuits', fontsize=9)
a.legend()

a = ax[1, 1]

x = np.arange(4)

a.bar(
    x - .2,
    sc['Minimum qubits'],
    .4,
    color='#d62728',
    label='Qubits needed'
)

a.bar(
    x + .2,
    sc['Padding (%)'] / 5,
    .4,
    color='#aaa',
    label='Padding (%) / 5'
)

a.set_xticks(x)
a.set_xticklabels([2, 3, 4, 5])
a.set_xlabel('Number of countries $n$')
a.set_title('(d) Qubits and padded-state overhead', fontsize=9)
a.legend()

for i, v in enumerate(sc['Minimum qubits']):
    a.text(i - .2, v + .2, str(v), ha='center', fontsize=8)

fig.tight_layout()

fig.savefig(O + 'fig2_depth_vs_countries.pdf')
fig.savefig(O + 'fig2_depth_vs_countries.png', dpi=200)

plt.close()


# Fig3
n = pd.read_csv(
    os.path.join(DATA, 'two_country_noise_repeated_summary.csv')
)

r = pd.read_csv(
    os.path.join(DATA, 'two_country_noise_repeated_raw.csv')
)

est = pd.read_csv(
    os.path.join(DATA, 'two_country_noise_repeated_estimates.csv')
)

x = n['Noise rate'].values
xp = np.where(x == 0, 3e-5, x)
ex = 0.012046495708135681

fig, ax = plt.subplots(2, 2, figsize=(8, 6.4))


def fmt(a):
    a.set_xscale('log')
    a.set_xticks(xp)
    a.set_xticklabels(
        ['0', '$10^{-4}$', '$10^{-3}$', '$5{\\times}10^{-3}$', '$10^{-2}$'],
        fontsize=7
    )
    a.minorticks_off()
    a.set_xlabel('Simulated noise rate $p$')


a = ax[0, 0]

a.errorbar(
    xp,
    n['Mean relative error (%)'],
    n['SD relative error (%)'],
    fmt='o-',
    c='#d62728',
    capsize=3
)

fmt(a)
a.set_ylabel('Relative error (%)')
a.set_title('(a) Relative error (10 repetitions)', fontsize=9)

a = ax[0, 1]

for p_, xx in zip(x, xp):
    e = est[est['Noise rate'] == p_]

    a.plot(
        [xx] * len(e),
        e['Estimated probability'] * 100,
        '.',
        c='#1f77b4',
        alpha=.5
    )

a.plot(
    xp,
    n['Mean estimated probability'] * 100,
    'o-',
    c='#d62728',
    label='Mean estimate'
)

a.axhline(
    ex * 100,
    c='k',
    ls='--',
    label='Exact 1.2046%'
)

fmt(a)
a.set_ylabel('Estimated probability (%)')
a.set_title('(b) Estimates vs exact value', fontsize=9)
a.legend()

a = ax[1, 0]

a.errorbar(
    xp,
    n['Mean invalid-state rate'] * 100,
    n['SD invalid-state rate'] * 100,
    fmt='s-',
    c='#2ca02c',
    capsize=3
)

fmt(a)
a.set_ylabel('Invalid padded-state outcomes (%)')
a.set_title('(c) Leakage into padded states', fontsize=9)

a = ax[1, 1]

g = r.groupby(
    ['Noise rate', 'Grover power']
)['Observed probability'].mean().unstack()

k = np.array([0, 1, 2, 4])
th = np.arcsin(np.sqrt(ex))

a.plot(
    k,
    np.sin((2 * k + 1) * th) ** 2,
    'k--',
    lw=2,
    label='Ideal'
)

for p_, c in zip(
    x[1:],
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
):
    a.plot(
        k,
        g.loc[p_].values,
        'o-',
        c=c,
        label=f'$p={p_:g}$'
    )

a.set_xlabel('Grover power $k$')
a.set_ylabel('Marked-outcome probability')
a.set_title('(d) Grover signal vs depth', fontsize=9)
a.legend(fontsize=7)

fig.tight_layout()

fig.savefig(O + 'fig3_error_vs_noise.pdf')
fig.savefig(O + 'fig3_error_vs_noise.png', dpi=200)

plt.close()


# Fig4
rc = pd.read_csv(
    os.path.join(DATA, 'resource_cost_comparison.csv')
)

rt = rc[rc.Method != 'QMLAE'].groupby(
    ['Method', 'Budget']
)['Mean runtime (seconds)'].mean().unstack()

fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.4))

B = [1024, 4096, 8192, 16384]

a = ax[0, 0]

for m in rt.index:
    a.loglog(
        B,
        rt.loc[m] * 1e3,
        marker=mk[m],
        c=col[m],
        label=m
    )

a.set_xlabel('Budget $M$ (samples)')
a.set_ylabel('Mean runtime per estimate (ms)')
a.set_title('(a) Classical runtime (measured)', fontsize=9)
a.legend(fontsize=7)
a.minorticks_off()
a.set_xticks(B)
a.set_xticklabels(['1,024', '4,096', '8,192', '16,384'], fontsize=7)

a = ax[0, 1]

qq = rc[
    (rc.Method == 'QMLAE') &
    (rc.Metric == 'Portfolio loss exceeds 80m')
].sort_values('Budget')

a.loglog(
    B,
    B,
    'o-',
    c='#7f7f7f',
    label='Classical samples / QMLAE shots'
)

a.loglog(
    B,
    qq['Grover applications'],
    '*-',
    c='#d62728',
    ms=9,
    label='Grover (oracle) applications'
)

a.loglog(
    B,
    [b / 4 * (1 + 3 + 5 + 9) for b in B],
    's--',
    c='#1f77b4',
    label='State-prep. applications $A$, $A^\\dagger$'
)

a.set_xlabel('Nominal budget $M$')
a.set_ylabel('Count')
a.set_title(
    r'(b) Query counts, schedule $k \in \{0,1,2,4\}$',
    fontsize=9
)
a.legend(fontsize=7)
a.minorticks_off()
a.set_xticks(B)
a.set_xticklabels(['1,024', '4,096', '8,192', '16,384'], fontsize=7)

a = ax[1, 0]

a.bar(
    q2['Grover iterations'].astype(str),
    q2['Total CX gates across shots'] / 1e6,
    color='#d62728'
)

a.set_xlabel('Grover power $k$ (4,096 shots each)')
a.set_ylabel('CX gates executed (millions)')
a.set_title('(c) Two-country CX executions per power', fontsize=9)

for i, v in enumerate(q2['Total CX gates across shots'] / 1e6):
    a.text(i, v + .1, f'{v:.2f}', ha='center', fontsize=8)

a = ax[1, 1]

v = [
    q2['Total CX gates across shots'].sum() / 1e6,
    225.83296
]

a.bar(
    ['Two-country\n(16,384 shots)', 'Three-country\n(20,480 shots)'],
    v,
    color=['#1f77b4', '#d62728']
)

a.set_yscale('log')
a.set_ylabel('CX gates executed (millions)')
a.set_title('(d) Total CX executions, one estimate', fontsize=9)

for i, y in enumerate(v):
    a.text(i, y * 1.15, f'{y:.1f}M', ha='center', fontsize=8)

a.set_ylim(5, 600)

fig.tight_layout()

fig.savefig(O + 'fig4_cost_comparison.pdf')
fig.savefig(O + 'fig4_cost_comparison.png', dpi=200)

plt.close()

print('ok', sum(q2['Total CX gates across shots']))