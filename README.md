# RDE26-SOP

RDE26-SOP is the Rank 1 entry for Competition 1-BC-SOPs in the IEEE WCCI 2026 / IEEE CEC 2026 Competition on Numerical Optimization Considering Accuracy and Speed.

## Competition

| Item | Description |
| --- | --- |
| Track | Competition 1-BC-SOPs |
| Problem class | Bound-constrained single-objective optimization |
| Benchmark | 29 CEC 2017 real-parameter numerical optimization problems |
| Runs | 25 independent runs per problem |
| Budget | 10000 x D function evaluations |
| Official rank | Rank 1 |
| Official source | [P-N-Suganthan/2026-CEC](https://github.com/P-N-Suganthan/2026-CEC) |

## Authors

Sichen Tao, Hanyu Hu, Ruihan Zhao, Qingke Zhang, Yifei Yang, Jian Wang, Masatoshi Kawai, and Hiroyuki Takizawa.

## Contents

| Path | Description |
| --- | --- |
| `benchmark/` | Public benchmark materials |
| `code/rde26-sop/main.py` | Track-specific run entry |
| `code/rde26-sop/src/` | Final algorithm source and required benchmark input data |

## Requirements

- Python 3
- C++14 compiler

## Run

```bash
python3 code/rde26-sop/main.py
```

Generated outputs are written under `code/rde26-sop/outputs/`.

## Reference

K. Qiao, X. Ban, P. Chen, K. V. Price, P. N. Suganthan, J. Liang, C. Yue, and G. Wu, "Performance comparison of CEC 2026 competition entries on numerical optimization considering accuracy and speed," IEEE WCCI/CEC 2026 competition slides.
