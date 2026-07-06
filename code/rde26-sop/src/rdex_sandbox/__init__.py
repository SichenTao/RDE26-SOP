"""
rdex_sop_cpp - RDEx 单目标优化器（平台版，C++ 扩展）

本模块基于 references/RDEx_SOP/RDEx.cpp，将 RDEx-SOP 算法移植到
PyEvo 平台，采用与 RDEx-CSOP 相同风格的 C++/Python 接口：

- C++ 侧：完整 RDEx-SOP 算法，评价通过 problem_bridge 回调到 Python Problem
- Python 侧：本目录仅提供 C++ 扩展版的 Python 封装（RDExSOPOptimizer）

典型用法::

    from evolutionary_computation.problems import Sphere
    from evolutionary_computation.optimizers.cec2017_sop import RDExSOPOptimizer, RDExSOPPythonOptimizer

    problem = Sphere(dimension=30)
    opt = RDExSOPOptimizer(population_size=300, max_evaluations=300000)      # C++ 扩展版（rdex_sop_cpp）
    opt_py = RDExSOPPythonOptimizer(population_size=300, max_evaluations=300000)  # Python 复现版（rdex_sop）
    best_x, best_f = opt.optimize(problem)
"""

__all__ = [
    "RDExSOPOptimizer",
    "RDExSOPPureCppOptimizer",
    "RDExSOPPureCppParallelOptimizer",
    "RDExSOPPureCppAutoOptimizer",
    "RDExSOPPureCppRBFInitialOptimizer",
    "RDExS2XoverW0Ed60Rm2p0C02D1Optimizer",
    "RDExS2TrFullGbestSt3Ft5Optimizer",
]

try:
    from .rdex_sop_optimizer import RDExSOPOptimizer
    from .rdex_sop_cpp_pure_optimizer import (
        RDExSOPPureCppOptimizer,
        RDExSOPPureCppParallelOptimizer,
        RDExSOPPureCppAutoOptimizer,
        RDExSOPPureCppRBFInitialOptimizer,
    )
    from .rdex_s2_xover_w0_ed60_rm2p0_c02_d1 import (
        RDExS2XoverW0Ed60Rm2p0C02D1Optimizer,
    )
    from .rdex_s2_trfull_gbest_st3_ft5 import (
        RDExS2TrFullGbestSt3Ft5Optimizer,
    )
except ImportError:  # pragma: no cover - 未编译扩展时的兜底提示
    class RDExSOPOptimizer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RDEx-SOP C++ 模块未编译。\n"
                "请先执行以下步骤编译扩展：\n"
                "  cd src/evolutionary_computation/optimizers/cec2017_sop/rdex_sop_cpp\n"
                "  bash build.sh\n"
                "或：\n"
                "  python setup.py build_ext --inplace\n"
                "\n"
                "如仅需 Python 复现版（无需编译），请使用 `rdex_sop`（兼容别名：`rdex_sop_repro` / `rdex_sop_python`；目录：cec2017_sop/rdex_sop）。\n"
            )

    class RDExSOPPureCppOptimizer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RDEx-SOP C++ 模块未编译。\n"
                "请先执行以下步骤编译扩展：\n"
                "  cd src/evolutionary_computation/optimizers/cec2017_sop/rdex_sop_cpp\n"
                "  bash build.sh\n"
                "或：\n"
                "  python setup.py build_ext --inplace\n"
                "\n"
                "如仅需 Python 复现版（无需编译），请使用 `rdex_sop`（目录：cec2017_sop/rdex_sop）。\n"
            )

    class RDExSOPPureCppParallelOptimizer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RDEx-SOP C++ 模块未编译。\n"
                "请先执行以下步骤编译扩展：\n"
                "  cd src/evolutionary_computation/optimizers/cec2017_sop/rdex_sop_cpp\n"
                "  bash build.sh\n"
            )

    class RDExSOPPureCppAutoOptimizer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RDEx-SOP C++ 模块未编译。\n"
                "请先执行以下步骤编译扩展：\n"
                "  cd src/evolutionary_computation/optimizers/cec2017_sop/rdex_sop_cpp\n"
                "  bash build.sh\n"
            )

    class RDExSOPPureCppRBFInitialOptimizer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RDEx-SOP C++ 模块未编译。\n"
                "请先执行以下步骤编译扩展：\n"
                "  cd src/evolutionary_computation/optimizers/cec2017_sop/rdex_sop_cpp\n"
                "  bash build.sh\n"
            )

    class RDExS2XoverW0Ed60Rm2p0C02D1Optimizer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RDEx-SOP C++ 模块未编译。\n"
                "请先执行：python setup.py build_ext --inplace\n"
            )

    class RDExS2TrFullGbestSt3Ft5Optimizer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RDEx-SOP C++ 模块未编译。\n"
                "请先执行：python setup.py build_ext --inplace\n"
            )

# The fixed hhyexp iteration candidate must remain importable even if one of
# the legacy wrapper imports above is unavailable in this sandbox copy.
try:
    from .rdex_s2_xover_w0_ed60_rm2p0_c02_d1 import (  # type: ignore[assignment]
        RDExS2XoverW0Ed60Rm2p0C02D1Optimizer,
    )
except ImportError:
    pass

try:
    from .rdex_s2_trfull_gbest_st3_ft5 import (  # type: ignore[assignment]
        RDExS2TrFullGbestSt3Ft5Optimizer,
    )
except ImportError:
    pass
