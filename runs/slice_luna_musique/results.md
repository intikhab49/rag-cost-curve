| system | EM | 95% CI | F1 | $/query | **$/correct** | calls | retr. unique | retr. billed | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 0.600 | 0.300–0.900 | 0.600 | 0.0034 | **0.0057** | 1.0 | 541 | 541 | 1880 |
| reranked | 0.700 | 0.400–1.000 | 0.700 | 0.0027 | **0.0039** | 1.0 | 595 | 595 | 8194 |
| iterative | 0.800 | 0.500–1.000 | 0.800 | 0.0071 | **0.0088** | 2.5 | 1044 | 1171 | 3368 |
| arag | 0.900 | 0.700–1.000 | 0.900 | 0.0098 | **0.0109** | 3.3 | 749 | 1052 | 5628 |

- `reranked` beats `naive` in 65.4% of paired bootstrap resamples (n=10)
- `iterative` beats `naive` in 88.5% of paired bootstrap resamples (n=10)
- `arag` beats `naive` in 97.0% of paired bootstrap resamples (n=10)
