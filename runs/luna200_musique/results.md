> Cost columns are **estimated** at `claude-sonnet-5` rates from token counts measured on `gpt-5.6-luna`. Not a billed figure.

| system | EM | 95% CI | F1 | $/query | **$/correct** | calls | retr. unique | retr. billed | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 0.275 | 0.215–0.335 | 0.381 | 0.0043 | **0.0155** | 1.0 | 570 | 570 | 2035 |
| reranked | 0.290 | 0.225–0.355 | 0.389 | 0.0045 | **0.0156** | 1.0 | 579 | 579 | 7062 |
| iterative | 0.505 | 0.435–0.570 | 0.631 | 0.0138 | **0.0273** | 2.9 | 1402 | 1623 | 6962 |
| arag | 0.600 | 0.530–0.670 | 0.743 | 0.0233 | **0.0389** | 5.0 | 1561 | 2727 | 8704 |

- `reranked` beats `naive` in 73.7% of paired bootstrap resamples (n=200)
- `iterative` beats `naive` in 100.0% of paired bootstrap resamples (n=200)
- `arag` beats `naive` in 100.0% of paired bootstrap resamples (n=200)
