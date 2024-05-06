from _limap import _base, _ceresbase, _optimize
import numpy as np

def _init_bundle_adjustment_engine(cfg, imagecols, max_num_iterations=100):
    if type(cfg) == dict:
        ba_config = _optimize.HybridBAConfig(cfg)
    else:
        ba_config = cfg
    ba_config.solver_options.logging_type = _ceresbase.LoggingType.SILENT
    ba_config.solver_options.max_num_iterations = max_num_iterations
    ba_engine = _optimize.HybridBAEngine(ba_config)
    ba_engine.InitImagecols(imagecols)
    return ba_engine

def _solve_bundle_adjustment(ba_engine):
    # setup and solve
    ba_engine.SetUp() # 设置问题结构
    ba_engine.Solve() # 进行优化求解
    return ba_engine

def solve_point_bundle_adjustment(cfg, imagecols, pointtracks, max_num_iterations=100):
    ba_engine = _init_bundle_adjustment_engine(cfg, imagecols, max_num_iterations=max_num_iterations)
    ba_engine.InitPointTracks(pointtracks)
    ba_engine = _solve_bundle_adjustment(ba_engine)
    return ba_engine

# NOTICE 只用线特征优化
def solve_line_bundle_adjustment(cfg, imagecols, linetracks, max_num_iterations=100):
    # 1. 初始化BA优化引擎 - 传递配置参数、图像集合以及最大迭代次数。
    ba_engine = _init_bundle_adjustment_engine(cfg, imagecols, max_num_iterations=max_num_iterations)
    # 2. 初始化line_tracks - 将线段轨迹初始化到BA引擎中，作为BA中将要优化的数据结构
    ba_engine.InitLineTracks(linetracks)
    # 3. BA优化
    ba_engine = _solve_bundle_adjustment(ba_engine)
    return ba_engine

# NOTICE 点线混合优化接口
def solve_hybrid_bundle_adjustment(cfg, imagecols, pointtracks, linetracks, max_num_iterations=100):
    ba_engine = _init_bundle_adjustment_engine(cfg, imagecols, max_num_iterations=max_num_iterations)
    ba_engine.InitPointTracks(pointtracks)
    ba_engine.InitLineTracks(linetracks)
    ba_engine = _solve_bundle_adjustment(ba_engine)
    return ba_engine

