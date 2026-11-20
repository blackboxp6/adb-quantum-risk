import pandas as pd, numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(HERE, '..', 'outputs', 'tables')

os.makedirs(OUT, exist_ok=True)

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
    'Portfolio loss $>$ USD 80m',
    'Expected stop-loss',
    'Both bonds exhausted'
]

M = [
    'Monte Carlo',
    'Stratified Monte Carlo',
    'Importance Sampling',
    'Sobol',
    'QMLAE'
]

s = (
    r'\begin{tabular}{llrrrrr}'
    r'\toprule Target & $M$ & MC & Strat. & IS & Sobol & QMLAE\\'
    r'\midrule'
    + '\n'
)

for t, n in zip(T, tl):
    for j, b in enumerate([1024, 4096, 8192, 16384]):
        v = [
            d[
                (d.Method == m) &
                (d.Metric == t) &
                (d.Budget == b)
            ]['Relative MAE (%)'].iloc[0]
            for m in M
        ]

        bi = int(np.argmin(v))

        c = [
            (
                f'\\textbf{{{x:.3g}}}'
                if i == bi
                else f'{x:.3g}'
            )
            for i, x in enumerate(v)
        ]

        s += (
            (n if j == 0 else '')
            + f' & {b:,} & '
            + ' & '.join(c)
            + r'\\'
            + '\n'
        )

    s += (
        r'\midrule' + '\n'
        if t != T[-1]
        else ''
    )

s += r'\bottomrule\end{tabular}'

open(OUT + '/tab_relmae.tex', 'w').write(s)


# RMSE at 16384
s = (
    r'\begin{tabular}{lrrrrr}'
    r'\toprule Target & MC & Strat. & IS & Sobol & QMLAE\\'
    r'\midrule'
    + '\n'
)

for t, n in zip(T, tl):
    v = [
        d[
            (d.Method == m) &
            (d.Metric == t) &
            (d.Budget == 16384)
        ]['Relative RMSE (%)'].iloc[0]
        for m in M
    ]

    bi = int(np.argmin(v))

    s += (
        n
        + ' & '
        + ' & '.join([
            (
                f'\\textbf{{{x:.3g}}}'
                if i == bi
                else f'{x:.3g}'
            )
            for i, x in enumerate(v)
        ])
        + r'\\'
        + '\n'
    )

s += r'\bottomrule\end{tabular}'

open(OUT + '/tab_rmse.tex', 'w').write(s)