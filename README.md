# 🏆 RDE26-SOP

![IEEE CEC 2026](https://img.shields.io/badge/IEEE%20CEC%202026-Champion-gold)
![Official Rank](https://img.shields.io/badge/Official%20Rank-1-blue)
![Track](https://img.shields.io/badge/Track-BC--SOPs-green)
![Language](https://img.shields.io/badge/Language-C%2B%2B%20%2B%20Python-lightgrey)

**RDE26-SOP is the official Champion and Rank 1 winner of Competition 1-BC-SOPs in the IEEE WCCI 2026 / IEEE CEC 2026 Competition on Numerical Optimization Considering Accuracy and Speed.**

This repository provides the final open-source algorithm package for the bound-constrained single-objective optimization track.

## 🏆 Champion Result

| Item | Description |
| --- | --- |
| Achievement | Champion / Rank 1 |
| Competition | IEEE WCCI 2026 / IEEE CEC 2026 Competition on Numerical Optimization Considering Accuracy and Speed |
| Track | Competition 1-BC-SOPs |
| Problem class | Bound-constrained single-objective optimization |
| Benchmark | 29 CEC 2017 real-parameter numerical optimization problems |
| Runs | 25 independent runs per problem |
| Budget | 10000 x D function evaluations |
| Official source | [P-N-Suganthan/2026-CEC](https://github.com/P-N-Suganthan/2026-CEC) |

## ✨ Highlights

- 🏆 Official Champion of the CEC 2026 BC-SOPs track.
- ⚡ Designed for the competition setting that jointly evaluates accuracy and speed.
- 📦 Clean final algorithm package with a track-specific `main.py` entry.
- 🔬 Part of the RDE26 series that won all four CEC 2026 numerical optimization tracks.

## Team

Sichen Tao, Hanyu Hu, Ruihan Zhao, Qingke Zhang, Yifei Yang, Jian Wang, Masatoshi Kawai, and Hiroyuki Takizawa.

Tohoku University-centered RDE26 research team.

## Repository Layout

| Path | Description |
| --- | --- |
| `benchmark/` | Public benchmark materials |
| `code/rde26-sop/main.py` | Track-specific run entry |
| `code/rde26-sop/src/` | Final algorithm source and required benchmark input data |

## Requirements

- Python 3
- C++14 compiler
- No third-party Python package is required

## Quick Start

```bash
python3 code/rde26-sop/main.py
```

Generated outputs are written under `code/rde26-sop/outputs/`.

## RDEx Research Line

RDE26-SOP continues the RDEx research line from the CEC 2025 competition series to the 2026 accuracy-and-speed competition setting.

- RDEx-SOP 2025: [arXiv:2603.27089](https://arxiv.org/abs/2603.27089)
- RDEx competition repository: [IEEE-CEC-2025-Competition-RDEx](https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx)

## Citation

K. Qiao, X. Ban, P. Chen, K. V. Price, P. N. Suganthan, J. Liang, C. Yue, and G. Wu, "Performance comparison of CEC 2026 competition entries on numerical optimization considering accuracy and speed," IEEE WCCI/CEC 2026 competition slides.
