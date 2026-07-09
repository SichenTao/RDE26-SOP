# 🏆 RDE26-SOP

![IEEE CEC 2026](https://img.shields.io/badge/IEEE%20CEC%202026-Champion-gold)
![Official Rank](https://img.shields.io/badge/Official%20Rank-1-blue)
![Track](https://img.shields.io/badge/Track-BC--SOPs-green)
![Language](https://img.shields.io/badge/Language-C%2B%2B%20%2B%20Python-lightgrey)

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">中文</a> |
  <strong>日本語</strong>
</p>

**RDE26-SOP は、IEEE WCCI 2026 / IEEE CEC 2026 の「精度と速度を考慮した数値最適化コンペティション」における Competition 1-BC-SOPs トラックの公式優勝・第 1 位のアルゴリズムです。**

本リポジトリは、有界制約単目的最適化トラック向けの最終版オープンソースアルゴリズムパッケージを提供します。

## 🏆 優勝結果

| 項目 | 説明 |
| --- | --- |
| 成績 | 優勝 / 第 1 位 |
| コンペティション | IEEE WCCI 2026 / IEEE CEC 2026 Competition on Numerical Optimization Considering Accuracy and Speed |
| トラック | Competition 1-BC-SOPs |
| 問題クラス | 有界制約単目的最適化 |
| ベンチマーク | CEC 2017 実数値数値最適化問題 29 問 |
| 実行回数 | 各問題につき 25 回の独立実行 |
| 評価予算 | 10000 x D 回の関数評価 |
| 公式ソース | [P-N-Suganthan/2026-CEC](https://github.com/P-N-Suganthan/2026-CEC) |

## ✨ 特長

- 🏆 CEC 2026 BC-SOPs トラックの公式優勝アルゴリズムです。
- ⚡ 精度と速度を同時に評価するコンペティション設定を対象に設計されています。
- 📦 トラック専用の `main.py` エントリを含む、整理済みの最終アルゴリズムパッケージです。
- 🔬 CEC 2026 の数値最適化 4 トラックすべてで優勝した RDE26 シリーズの一部です。

## 🧭 RDE26 シリーズナビゲーション

| 結果 | トラック | リポジトリ |
| --- | --- | --- |
| 🏆 優勝 / 第 1 位 (sum/RS: 277418/75) | Competition 1-BC-SOPs | [RDE26-SOP](https://github.com/SichenTao/RDE26-SOP) |
| 🏆 優勝 / 第 1 位 (sum/RS: 192754/55) | Competition 2-CSOPs | [RDE26-CSOP](https://github.com/SichenTao/RDE26-CSOP) |
| 🏆 優勝 / 第 1 位 (sum/RS: 87261.5/16) | Competition 3-BC-MOPs | [RDE26-MOP](https://github.com/SichenTao/RDE26-MOP) |
| 🏆 優勝 / 第 1 位 (sum/RS: 188840/38) | Competition 4-CMOPs | [RDE26-CMOP](https://github.com/SichenTao/RDE26-CMOP) |

## チーム

Sichen Tao, Hanyu Hu, Ruihan Zhao, Qingke Zhang, Yifei Yang, Jian Wang, Masatoshi Kawai, and Hiroyuki Takizawa.

東北大学を中心とする RDE26 研究チームです。

## リポジトリ構成

| パス | 説明 |
| --- | --- |
| `benchmark/` | 公開ベンチマーク資料 |
| `code/rde26-sop/main.py` | トラック専用の実行エントリ |
| `code/rde26-sop/src/` | 最終アルゴリズムのソースコードと必要なベンチマーク入力データ |

## 要件

- Python 3
- C++14 コンパイラ
- サードパーティ製 Python パッケージは不要です

## クイックスタート

```bash
python3 code/rde26-sop/main.py
```

生成された出力は `code/rde26-sop/outputs/` に書き込まれます。

## 🔗 RDE 研究ライン

RDE26 シリーズは、IEEE CEC 2024 の単目的トラックで Runner-Up となった RDE エントリと、CEC 2025 の RDEx Rank 1 U-score シリーズを発展させ、2026 年の 4 トラック優勝シリーズへとつなげたものです。

### RDE 2024 エントリ

| 結果 | トラック | リポジトリ | 論文 |
| --- | --- | --- | --- |
| 🥈 Runner-Up Award | IEEE CEC 2024 BC-SOPs | [RDE](https://github.com/SichenTao/IEEE-WCCI-CEC-2024-Competition-RDE) | [arXiv:2404.16280](https://arxiv.org/abs/2404.16280) |

### RDEx 2025 ナビゲーション

| 結果 | トラック | リポジトリ | 論文 |
| --- | --- | --- | --- |
| 🥇 Rank 1 U-score (total: 81229.5) | CEC 2025 BC-SOPs | [RDEx_SOP](https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series/tree/main/RDEx_SOP) | [arXiv:2603.27089](https://arxiv.org/abs/2603.27089) |
| 🥇 Rank 1 U-score (total: 53680.5) | CEC 2025 BC-CSOPs | [RDEx_CSOP](https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series/tree/main/RDEx_CSOP) | [arXiv:2603.27090](https://arxiv.org/abs/2603.27090) |
| 🥇 Rank 1 U-score (total: 36343.5) | CEC 2025 BC-MOPs | [RDEx_MOP](https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series/tree/main/RDEx_MOP) | [arXiv:2603.27092](https://arxiv.org/abs/2603.27092) |
| 🥇 Rank 1 U-score (total: 58456.0) | CEC 2025 BC-CMOPs | [RDEx_CMOP](https://github.com/SichenTao/IEEE-CEC-2025-Competition-RDEx-Series/tree/main/RDEx_CMOP) | [arXiv:2604.03708](https://arxiv.org/abs/2604.03708) |

### RDE 研究ラインの論文引用

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
