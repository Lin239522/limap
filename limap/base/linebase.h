#ifndef LIMAP_BASE_LINEBASE_H_
#define LIMAP_BASE_LINEBASE_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <set>

namespace py = pybind11;

#include "util/types.h"
#include "base/camera_view.h"

namespace limap {

class Line2d {
public:
    // 设置2D线段的基本属性 - 起点向量，终点向量，评分
    Line2d() {}
    Line2d(const Eigen::MatrixXd& seg2d); // start, end ： 线段的起点和终点
    Line2d(V2D start, V2D end, double score=-1);
    V2D start, end;
    double score = -1; // 分数 - 线段的置信度或质量

    // 计算2D线段的几何属性
    double length() const {return (start - end).norm();}        // 线段的长度
    V2D midpoint() const {return 0.5 * (start + end);}          // 线段的中点
    V2D direction() const {return (end - start).normalized();}  // 线段的方向：起点到终点的单位向量，表示线段的方向
    V2D perp_direction() const {V2D dir = direction(); return V2D(dir[1], -dir[0]); } // 线段垂直方向的单位向量
    V3D coords() const; // get homogeneous coordinate           // 线段的齐次坐标表示

    // 计算2D点和2D线段间的关系
    V2D point_projection(const V2D& p) const;                   // 计算点P在线段上的投影
    double point_distance(const V2D& p) const;                  // ! 计算2D点P到2D线段的距离
    Eigen::MatrixXd as_array() const;                           // 返回包含线段数据的 Eigen::MatrixXd
};

class Line3d {
public:
    // 设置3D线段的基本属性 - 起点向量，终点向量，评分, 不确定性， 深度向量
    Line3d() {}
    Line3d(const Eigen::MatrixXd& seg3d);
    Line3d(V3D start, V3D end, double score=-1, double depth_start=-1, double depth_end=-1, double uncertainty=-1);
    V3D start, end;
    double score = -1;                  // 线段的评分，用于表示线段的置信度或质量
    double uncertainty = -1.0;          // 不确定性，用于描述线段的测量不确定性
    V2D depths; // [depth_start, depth_end] for the source perspective image 包含线段起点和终点深度的向量，通常用于从源图像的视角描述深度信息。

    // 计算3D线段的相关属性
    void set_uncertainty(const double val) { uncertainty = val; } // 设置线段的不确定性。
    double length() const {return (start - end).norm();}          // 线段长度
    V3D midpoint() const {return 0.5 * (start + end);}            // 线段中点
    V3D direction() const {return (end - start).normalized();}    // 线段方向：起点到终点的单位向量，表示线段的方向

    // 计算3D点和3D线段间的关系
    V3D point_projection(const V3D& p) const;       // 计算3D点P在3D线段上的投影
    double point_distance(const V3D& p) const;      // 计算3D点P到3D线段的距离
    Eigen::MatrixXd as_array() const;               // 返回一个包含线段数据的矩阵
    Line2d projection(const CameraView& view) const;// ! 根据给定的相机视图计算线段的二维投影。
    double sensitivity(const CameraView& view) const; // in angle, 0 for perfect view, 90 for collapsing 计算线段在给定相机视角下的视觉敏感度。
    double computeUncertainty(const CameraView& view, const double var2d=5.0) const; // 计算线段的不确定性-根据给定的相机视角和二维方差计算线段的不确定性。
};

std::vector<Line2d> GetLine2dVectorFromArray(const Eigen::MatrixXd& segs2d);
std::vector<Line3d> GetLine3dVectorFromArray(const std::vector<Eigen::MatrixXd>& segs3d);

Line2d projection_line3d(const Line3d& line3d, const CameraView& view);
Line3d unprojection_line2d(const Line2d& line2d, const CameraView& view, const std::pair<double, double>& depths);

} // namespace limap

#endif

