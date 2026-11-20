# ADB Quantum Risk Evaluation — Results Draft

This report summarizes numerical benchmark results from the classical sampling and quantum amplitude estimation experiments.

## 1. Estimation Accuracy

The following table summarizes the estimation accuracy at the largest available budget.

| Method                 | Metric                     |   Budget |   Relative MAE (%) |   Relative RMSE (%) |              MAE |             RMSE |
|:-----------------------|:---------------------------|---------:|-------------------:|--------------------:|-----------------:|-----------------:|
| Importance Sampling    | Both bonds exhausted       |    16384 |         21.5611    |           27.8442   |      6.82032e-06 |      8.80781e-06 |
| Monte Carlo            | Both bonds exhausted       |    16384 |        116.334     |          129.378    |      3.67995e-05 |      4.09256e-05 |
| QMLAE                  | Both bonds exhausted       |    16384 |         20.6982    |           26.0253   |      6.54735e-06 |      8.23245e-06 |
| Sobol                  | Both bonds exhausted       |    16384 |        127.77      |          147.791    |      4.0417e-05  |      4.67499e-05 |
| Stratified Monte Carlo | Both bonds exhausted       |    16384 |         57.8165    |           71.7259   |      1.82888e-05 |      2.26887e-05 |
| Importance Sampling    | Both countries pay         |    16384 |          2.21884   |            2.69333  |      0.000979336 |      0.00118876  |
| Monte Carlo            | Both countries pay         |    16384 |          3.09637   |            3.78492  |      0.00136665  |      0.00167056  |
| QMLAE                  | Both countries pay         |    16384 |          0.803132  |            1.05416  |      0.000354481 |      0.000465276 |
| Sobol                  | Both countries pay         |    16384 |          0.758377  |            0.925881 |      0.000334727 |      0.000408659 |
| Stratified Monte Carlo | Both countries pay         |    16384 |          1.53546   |            1.92846  |      0.00067771  |      0.000851168 |
| Importance Sampling    | Expected principal loss    |    16384 |          0.66458   |            0.838485 |  66879.3         |  84380.1         |
| Monte Carlo            | Expected principal loss    |    16384 |          1.42163   |            1.77279  | 143064           | 178403           |
| QMLAE                  | Expected principal loss    |    16384 |          0.712474  |            0.930773 |  71699.1         |  93667.3         |
| Sobol                  | Expected principal loss    |    16384 |          0.0919827 |            0.115727 |   9256.58        |  11646.1         |
| Stratified Monte Carlo | Expected principal loss    |    16384 |          0.49968   |            0.628184 |  50284.7         |  63216.6         |
| Importance Sampling    | Expected stop-loss         |    16384 |          2.41759   |            3.00785  |   6826.08        |   8492.66        |
| Monte Carlo            | Expected stop-loss         |    16384 |          8.08652   |            9.83591  |  22832.3         |  27771.7         |
| QMLAE                  | Expected stop-loss         |    16384 |          2.05736   |            2.5133   |   5808.95        |   7096.3         |
| Sobol                  | Expected stop-loss         |    16384 |          2.10823   |            2.64091  |   5952.59        |   7456.61        |
| Stratified Monte Carlo | Expected stop-loss         |    16384 |          3.48502   |            4.25824  |   9839.96        |  12023.1         |
| Importance Sampling    | Portfolio loss exceeds 80m |    16384 |          2.4018    |            3.00062  |      0.000289333 |      0.00036147  |
| Monte Carlo            | Portfolio loss exceeds 80m |    16384 |          5.77707   |            7.47333  |      0.000695934 |      0.000900275 |
| QMLAE                  | Portfolio loss exceeds 80m |    16384 |          0.981236  |            1.24418  |      0.000118205 |      0.00014988  |
| Sobol                  | Portfolio loss exceeds 80m |    16384 |          2.1368    |            2.57305  |      0.000257409 |      0.000309962 |
| Stratified Monte Carlo | Portfolio loss exceeds 80m |    16384 |          3.23663   |            3.91101  |      0.000389901 |      0.00047114  |


## 2. Circuit Scalability

|   countries |   outcomes |   qubits |   depth |   total_gates |   cx_gates |   u_gates |
|------------:|-----------:|---------:|--------:|--------------:|-----------:|----------:|
|           2 |        100 |        7 |     241 |           247 |        120 |       127 |
|           3 |       1000 |       10 |    2027 |          2036 |       1013 |      1023 |
|           4 |      10000 |       14 |   32739 |         32752 |      16369 |     16383 |
|           5 |     100000 |       17 |  262109 |        262125 |     131054 |    131071 |


These scaling results are based on synthetic probability distributions and generic state preparation. They should not be interpreted as a fully calibrated multi-country ADB model.

## 3. Quantum Noise Sensitivity

|   Noise rate |   Mean relative error (%) |   SD relative error (%) |   Mean invalid-state rate |
|-------------:|--------------------------:|------------------------:|--------------------------:|
|       0      |                   1.38136 |                1.09427  |                0          |
|       0.0001 |                  20.7511  |                0.226485 |                0.00393677 |
|       0.001  |                  57.5095  |                0.150088 |                0.0263733  |
|       0.005  |                  33.2842  |                0.328359 |                0.0894531  |
|       0.01   |                  22.2358  |                0.260459 |                0.122253   |


The noise experiment uses a simulated noise model. The noise parameters do not represent measured error rates from an actual quantum processor.

## 4. Quantum Resource Estimation

|   Grover iterations |   Shots |   Circuit depth |   U gates per circuit |   CX gates per circuit |   Total CX gates across shots |   Total gates across shots |   Total oracle calls across shots |   Total state preparations across shots |
|--------------------:|--------:|----------------:|----------------------:|-----------------------:|------------------------------:|---------------------------:|----------------------------------:|----------------------------------------:|
|                   0 |    4096 |             241 |                   127 |                    120 |              491520           |                1.01171e+06 |                                 0 |                                    4096 |
|                   1 |    4096 |            1077 |                   641 |                    608 |                   2.49037e+06 |                5.1159e+06  |                              4096 |                                   12288 |
|                   2 |    4096 |            1913 |                  1155 |                   1096 |                   4.48922e+06 |                9.2201e+06  |                              8192 |                                   20480 |
|                   4 |    4096 |            3585 |                  2183 |                   2072 |                   8.48691e+06 |                1.74285e+07 |                             16384 |                                   36864 |


## 5. Quantum Circuit Components

| Component              |   Depth |   U gates |   CX gates |   Total gates |
|:-----------------------|--------:|----------:|-----------:|--------------:|
| State preparation A    |     241 |       127 |        120 |           247 |
| Inverse preparation A† |     241 |       127 |        120 |           247 |
| Phase oracle           |     244 |       127 |        126 |           253 |
| Zero reflection S0     |     130 |       148 |        122 |           270 |


## 6. Limitations

- Quantum results are simulator-based.
- No physical quantum hardware experiment was conducted.
- The classical benchmarks use a precomputed probability distribution.
- Equal nominal sample or shot budgets do not represent equal computational costs.
- The ideal statistical QMLAE runtime is not quantum hardware runtime.
- State preparation and oracle construction are significant resource considerations.
- Noise parameters are simulated and are not hardware measurements.
- The scaling experiment uses synthetic distributions.
- No quantum advantage is claimed.
