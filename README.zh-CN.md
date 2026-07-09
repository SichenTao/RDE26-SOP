# 🏆 RDE26-SOP

![IEEE CEC 2026](https://img.shields.io/badge/IEEE%20CEC%202026-Champion-gold)
![Official Rank](https://img.shields.io/badge/Official%20Rank-1-blue)
![Track](https://img.shields.io/badge/Track-BC--SOPs-green)
![Language](https://img.shields.io/badge/Language-C%2B%2B%20%2B%20Python-lightgrey)

<p align="center">
  <a href="README.md">English</a> |
  <strong>中文</strong> |
  <a href="README.ja.md">日本語</a>
</p>

**RDE26-SOP 是 IEEE WCCI 2026 / IEEE CEC 2026 数值优化精度与速度竞赛 Competition 1-BC-SOPs 赛道的官方冠军和第 1 名算法。**

本仓库提供有界约束单目标优化赛道的最终开源算法包。

## 🏆 冠军结果

| 项目 | 说明 |
| --- | --- |
| 成绩 | 冠军 / 第 1 名 |
| 竞赛 | IEEE WCCI 2026 / IEEE CEC 2026 数值优化精度与速度竞赛 |
| 赛道 | Competition 1-BC-SOPs |
| 问题类别 | 有界约束单目标优化 |
| 基准 | 29 个 CEC 2017 实参数值优化问题 |
| 运行次数 | 每个问题 25 次独立运行 |
| 预算 | 10000 x D 次函数评价 |
| 官方来源 | [P-N-Suganthan/2026-CEC](https://github.com/P-N-Suganthan/2026-CEC) |

## ✨ 主要特点

- 🏆 CEC 2026 BC-SOPs 赛道官方冠军。
- ⚡ 面向同时评价求解精度与运行速度的竞赛设置。
- 📦 提供清晰的最终算法包，并包含该赛道专用的 `main.py` 入口。
- 🔬 属于 RDE26 系列；该系列获得 CEC 2026 数值优化四个赛道全部冠军。

## 🧭 RDE26 系列导航

| 结果 | 赛道 | 仓库 |
| --- | --- | --- |
| 🏆 冠军 / 第 1 名 (sum/RS: 277418/75) | Competition 1-BC-SOPs | [RDE26-SOP](https://github.com/SichenTao/RDE26-SOP) |
| 🏆 冠军 / 第 1 名 (sum/RS: 192754/55) | Competition 2-CSOPs | [RDE26-CSOP](https://github.com/SichenTao/RDE26-CSOP) |
| 🏆 冠军 / 第 1 名 (sum/RS: 87261.5/16) | Competition 3-BC-MOPs | [RDE26-MOP](https://github.com/SichenTao/RDE26-MOP) |
| 🏆 冠军 / 第 1 名 (sum/RS: 188840/38) | Competition 4-CMOPs | [RDE26-CMOP](https://github.com/SichenTao/RDE26-CMOP) |

## 团队

Sichen Tao, Hanyu Hu, Ruihan Zhao, Qingke Zhang, Yifei Yang, Jian Wang, Masatoshi Kawai, and Hiroyuki Takizawa.

以东北大学为中心的 RDE26 研究团队。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `benchmark/` | 公开基准材料 |
| `code/rde26-sop/main.py` | 该赛道专用运行入口 |
| `code/rde26-sop/src/` | 最终算法源代码和必要的基准输入数据 |

## 运行需求

- Python 3
- C++14 编译器
- 不需要第三方 Python 包

## 快速开始

```bash
python3 code/rde26-sop/main.py
```

生成结果会写入 `code/rde26-sop/outputs/`。

## 🔗 RDE 研究线

RDE26 系列将 RDE 研究线从 IEEE CEC 2024 单目标赛道亚军方案和 CEC 2025 RDEx U-score 第 1 系列，推进到 2026 四赛道冠军系列。

### RDE 2024 参赛方案

| 结果 | 赛道 | 仓库 | 论文 |
| --- | --- | --- | --- |
| 🥈 亚军奖 | IEEE CEC 2024 BC-SOPs | [RDE](https://github.com/SichenTao/IEEE-WCCI-CEC-2024-Competition-RDE) | [arXiv:2404.16280](https://arxiv.org/abs/2404.16280) |

### RDEx 2025 导航

| 结果 | 赛道 | 仓库 | 论文 |
| --- | --- | --- | --- |
| 🥇 U-score 第 1 名 (total: 81229.5) | CEC 2025 BC-SOPs | [RDEx_SOP](https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series/tree/main/RDEx_SOP) | [arXiv:2603.27089](https://arxiv.org/abs/2603.27089) |
| 🥇 U-score 第 1 名 (total: 53680.5) | CEC 2025 BC-CSOPs | [RDEx_CSOP](https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series/tree/main/RDEx_CSOP) | [arXiv:2603.27090](https://arxiv.org/abs/2603.27090) |
| 🥇 U-score 第 1 名 (total: 36343.5) | CEC 2025 BC-MOPs | [RDEx_MOP](https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series/tree/main/RDEx_MOP) | [arXiv:2603.27092](https://arxiv.org/abs/2603.27092) |
| 🥇 U-score 第 1 名 (total: 58456.0) | CEC 2025 BC-CMOPs | [RDEx_CMOP](https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series/tree/main/RDEx_CMOP) | [arXiv:2604.03708](https://arxiv.org/abs/2604.03708) |

### RDE 研究线论文引用

- 2024 RDE: Sichen Tao, Ruihan Zhao, Kaiyu Wang, and Shangce Gao, "An Efficient Reconstructed Differential Evolution Variant by Some of the Current State-of-the-art Strategies for Solving Single Objective Bound Constrained Problems," arXiv:2404.16280, 2024. DOI: [10.48550/arXiv.2404.16280](https://doi.org/10.48550/arXiv.2404.16280).
- 2025 RDEx-SOP: Sichen Tao, Yifei Yang, Ruihan Zhao, Kaiyu Wang, Sicheng Liu, and Shangce Gao, "RDEx-SOP: Exploitation-Biased Reconstructed Differential Evolution for Fixed-Budget Bound-Constrained Single-Objective Optimization," arXiv:2603.27089, 2026. DOI: [10.48550/arXiv.2603.27089](https://doi.org/10.48550/arXiv.2603.27089).
- 2025 RDEx-CSOP: Sichen Tao, Yifei Yang, Ruihan Zhao, Kaiyu Wang, Sicheng Liu, and Shangce Gao, "RDEx-CSOP: Feasibility-Aware Reconstructed Differential Evolution with Adaptive epsilon-Constraint Ranking," arXiv:2603.27090, 2026. DOI: [10.48550/arXiv.2603.27090](https://doi.org/10.48550/arXiv.2603.27090).
- 2025 RDEx-MOP: Sichen Tao, Yifei Yang, Ruihan Zhao, Kaiyu Wang, Sicheng Liu, and Shangce Gao, "RDEx-MOP: Indicator-Guided Reconstructed Differential Evolution for Fixed-Budget Multiobjective Optimization," arXiv:2603.27092, 2026. DOI: [10.48550/arXiv.2603.27092](https://doi.org/10.48550/arXiv.2603.27092).
- 2025 RDEx-CMOP: Sichen Tao, Yifei Yang, Ruihan Zhao, Kaiyu Wang, Sicheng Liu, and Shangce Gao, "RDEx-CMOP: Feasibility-Aware Indicator-Guided Differential Evolution for Fixed-Budget Constrained Multiobjective Optimization," arXiv:2604.03708, 2026. DOI: [10.48550/arXiv.2604.03708](https://doi.org/10.48550/arXiv.2604.03708).

<details>
<summary>BibTeX</summary>

```bibtex
@misc{tao2024efficient,
  title = {An Efficient Reconstructed Differential Evolution Variant by Some of the Current State-of-the-art Strategies for Solving Single Objective Bound Constrained Problems},
  author = {Tao, Sichen and Zhao, Ruihan and Wang, Kaiyu and Gao, Shangce},
  year = {2024},
  eprint = {2404.16280},
  archivePrefix = {arXiv},
  primaryClass = {cs.NE},
  doi = {10.48550/arXiv.2404.16280}
}

@misc{tao2026rdexsop,
  title = {RDEx-SOP: Exploitation-Biased Reconstructed Differential Evolution for Fixed-Budget Bound-Constrained Single-Objective Optimization},
  author = {Tao, Sichen and Yang, Yifei and Zhao, Ruihan and Wang, Kaiyu and Liu, Sicheng and Gao, Shangce},
  year = {2026},
  eprint = {2603.27089},
  archivePrefix = {arXiv},
  primaryClass = {cs.NE},
  doi = {10.48550/arXiv.2603.27089}
}

@misc{tao2026rdexcsop,
  title = {RDEx-CSOP: Feasibility-Aware Reconstructed Differential Evolution with Adaptive epsilon-Constraint Ranking},
  author = {Tao, Sichen and Yang, Yifei and Zhao, Ruihan and Wang, Kaiyu and Liu, Sicheng and Gao, Shangce},
  year = {2026},
  eprint = {2603.27090},
  archivePrefix = {arXiv},
  primaryClass = {cs.NE},
  doi = {10.48550/arXiv.2603.27090}
}

@misc{tao2026rdexmop,
  title = {RDEx-MOP: Indicator-Guided Reconstructed Differential Evolution for Fixed-Budget Multiobjective Optimization},
  author = {Tao, Sichen and Yang, Yifei and Zhao, Ruihan and Wang, Kaiyu and Liu, Sicheng and Gao, Shangce},
  year = {2026},
  eprint = {2603.27092},
  archivePrefix = {arXiv},
  primaryClass = {cs.NE},
  doi = {10.48550/arXiv.2603.27092}
}

@misc{tao2026rdexcmop,
  title = {RDEx-CMOP: Feasibility-Aware Indicator-Guided Differential Evolution for Fixed-Budget Constrained Multiobjective Optimization},
  author = {Tao, Sichen and Yang, Yifei and Zhao, Ruihan and Wang, Kaiyu and Liu, Sicheng and Gao, Shangce},
  year = {2026},
  eprint = {2604.03708},
  archivePrefix = {arXiv},
  primaryClass = {cs.NE},
  doi = {10.48550/arXiv.2604.03708}
}
```

</details>

## 引用

K. Qiao, X. Ban, P. Chen, K. V. Price, P. N. Suganthan, J. Liang, C. Yue, and G. Wu, "Performance comparison of CEC 2026 competition entries on numerical optimization considering accuracy and speed," IEEE WCCI/CEC 2026 competition slides.
