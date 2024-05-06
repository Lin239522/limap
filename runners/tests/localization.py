import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import limap
import limap.base as _base
import limap.estimators as _estimators
import logging
import argparse
from pathlib import Path
from hloc.utils.read_write_model import *

formatter = logging.Formatter(
    fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
# 配置日志
logger = logging.getLogger("TestLoc")
logger.setLevel(logging.INFO)          # 这里可以设置日志等级
logger.addHandler(handler)
logger.propagate = False

# 参数解析函数
# 数据文件路径、输出目录、RANSAC算法的选择和阈值设置。
def parse_args():
    arg_parser = argparse.ArgumentParser(description='minimal test for visual localization with points and lines')
    arg_parser.add_argument('--data', type=Path, default='runners/tests/data/localization/localization_test_data_stairs_1.npy',
                            help='Path to test data file, default: %(default)s')
    arg_parser.add_argument('--outputs', type=Path, default='outputs/test_outputs/localization',
                            help='Path to the output directory, default: %(default)s')
    arg_parser.add_argument('--ransac_method', choices=['ransac', 'solver', 'hybrid'], default='hybrid',
                            help='RANSAC method')
    arg_parser.add_argument('--thres', type=float, default=5.0,
                            help='Threshold for RANSAC/Solver first RANSAC, default: %(default)s')
    arg_parser.add_argument('--thres_point', type=float, default=5.0,
                            help='Threshold for points in hybrid RANSAC, default: %(default)s')
    arg_parser.add_argument('--thres_line', type=float, default=5.0,
                            help='Threshold for lines in hybrid RANSAC, default: %(default)s')
    arg_parser.add_argument('--line2d_matcher', type=str, default='sold2',
                            help='2D matcher for lines, default: %(default)s')
    arg_parser.add_argument('--line_cost_func', type=str, default='PerpendicularDist',
                            help='Line Cost function for scoring and optimization, default: %(default)s')

    args, unknown = arg_parser.parse_known_args()
    return args

def main():
    args = parse_args()

    # step 1 加载数据文件并解析
    data = np.load(args.data, allow_pickle=True).item()
    # 配置信息
    cfg = data['cfg']
    cfg['2d_matcher'] = args.line2d_matcher
    cfg['line_cost_func'] = args.line_cost_func
    cfg['ransac']['method'] = args.ransac_method
    cfg['ransac']['thres'] = args.thres
    cfg['ransac']['thres_point'] = args.thres_point
    cfg['ransac']['thres_line'] = args.thres_line

    # 线和点的数据
    l3ds = data['l3ds']
    l2ds = data['l2ds']
    l3d_ids = data['l3d_ids']

    p3ds = data['p3ds']
    p2ds = data['p2ds']
    # 相机模型
    cam = data['camera']

    # NOTICE 基于点和线的数据，计算相机的绝对姿态
    # 该函数采用 2D-3D 点和线对应关系、相机模型，并返回估计的绝对相机位姿。
    final_pose, ransac_stats = _estimators.pl_estimate_absolute_pose(                   
                cfg, l3ds, l3d_ids, l2ds, p3ds, p2ds, cam, silent=True, logger=logger) 

    # step 3 用RANSC统计信息并进行日志记录
    # Let's Check some RANSAC status
    log = "RANSAC stats: \n"
    log += f"num_iterations_total: {ransac_stats.num_iterations_total}\n"
    log += f"best_num_inliers: {ransac_stats.best_num_inliers}\n"
    log += f"best_model_score: {ransac_stats.best_model_score}\n"
    log += f"inlier_ratios (Points, Lines): {ransac_stats.inlier_ratios}\n"
    logger.info(log)

    # step 4 计算结果姿态的误差
    # 相机位姿真值
    R_gt, t_gt = data['pose_gt'].R(), data['pose_gt'].tvec

    log = "Results: \n"
    log += f"Result(P+L) Pose (qvec, tvec): {final_pose.qvec}, {final_pose.tvec}\n"
    log += f"HLoc(Point) Pose (qvec, tvec): {data['pose_point'].qvec}, {data['pose_point'].tvec}\n"
    log += f"GT Pose (qvec, tvec): {data['pose_gt'].qvec}, {data['pose_gt'].tvec}\n\n"

    # 根据绝对位姿估计结果，读取由点线估计的位姿值；并与真值对比，计算平移和旋转的误差
    R, t = final_pose.R(), final_pose.tvec
    e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
    e_R = np.rad2deg(np.abs(np.arccos(cos)))
    log += f'Result(P+L) Pose errors: {e_t:.3f}m, {e_R:.3f}deg\n'

    # 根据原始数据，读取由点估计的位姿值；并与真值对比，计算平移和旋转的误差
    R, t = data['pose_point'].R(), data['pose_point'].tvec
    e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
    e_R = np.rad2deg(np.abs(np.arccos(cos)))
    log += f'HLoc(Point) Pose errors: {e_t:.3f}m, {e_R:.3f}deg'

    logger.info(log)

    # step 5 用过滤后的内点和线，来更新视觉表示，并进行保存 
    # IDEA 这个好像可以用
    inlier_indices = ransac_stats.inlier_indices
    p2ds = np.array(p2ds)[inlier_indices[0]]
    p3ds = np.array(p3ds)[inlier_indices[0]]
    l2ds = np.array(l2ds)[inlier_indices[1]]
    l3d_ids = np.array(l3d_ids)[inlier_indices[1]]

    # 初始化相机视图
    camview_point = _base.CameraView(cam, data['pose_point'])
    camview_line = _base.CameraView(cam, final_pose)

    # 创建输出目录
    args.outputs.mkdir(parents=True, exist_ok=True)

    img = data['image'].copy()
    for l2d, l3d_id in zip(l2ds, l3d_ids):
        l3d = l3ds[l3d_id]
        l2d_proj = l3d.projection(camview_point)
        img = cv2.line(img, l2d.start.astype(int), l2d.end.astype(int), color=[255, 0, 0])
        img = cv2.line(img, l2d_proj.start.astype(int), l2d_proj.end.astype(int), color=[0, 0, 255])
    for p2d, p3d in zip(p2ds, p3ds):
        img = cv2.circle(img, p2d.astype(int), radius=1, color=[255, 0, 0])
        img = cv2.circle(img, camview_point.projection(p3d).astype(int), radius=1, color=[0, 255, 0])
    cv2.imwrite((args.outputs / "pose_point.png").as_posix(), img)


    # 复制原始图像
    img = data['image'].copy()
    # 遍历每个线特征l2d，在图像中绘制线段
    # 遍历每个3D线段l3d，使用相应的相机视图进行投影，得到2D投影后的线段l2d_proj，并在图像上绘制
    for l2d, l3d_id in zip(l2ds, l3d_ids):
        l3d = l3ds[l3d_id]
        l2d_proj = l3d.projection(camview_line)
        img = cv2.line(img, l2d.start.astype(int), l2d.end.astype(int), color=[255, 0, 0])
        img = cv2.line(img, l2d_proj.start.astype(int), l2d_proj.end.astype(int), color=[0, 0, 255])
    for p2d, p3d in zip(p2ds, p3ds):
        img = cv2.circle(img, p2d.astype(int), radius=1, color=[255, 0, 0])
        img = cv2.circle(img, camview_line.projection(p3d).astype(int), radius=1, color=[0, 255, 0])
    cv2.imwrite((args.outputs / "pose_p+l.png").as_posix(), img)

if __name__ == '__main__':
    main()

