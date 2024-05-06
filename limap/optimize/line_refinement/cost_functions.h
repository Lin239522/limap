#ifndef LIMAP_OPTIMIZE_LINE_REFINEMENT_COST_FUNCTIONS_H_
#define LIMAP_OPTIMIZE_LINE_REFINEMENT_COST_FUNCTIONS_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_limap/helpers.h"

#include <colmap/base/camera_models.h>
#include <ceres/ceres.h>

#include "base/camera_models.h"
#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/infinite_line.h"
#include "util/types.h"

#include "ceresbase/line_transforms.h"
#include "ceresbase/line_projection.h"
#include "ceresbase/line_dists.h"

#ifdef INTERPOLATION_ENABLED
    #include "optimize/line_refinement/pixel_cost_functions.h"
#endif // INTERPOLATION_ENABLED

namespace py = pybind11;

namespace limap {

namespace optimize {

namespace line_refinement {

////////////////////////////////////////////////////////////
// VP Constraints 
////////////////////////////////////////////////////////////
template <typename CameraModel>
struct VPConstraintsFunctor {
public:
    VPConstraintsFunctor(const V3D& VP, const double* params = NULL, const double* qvec = NULL):
        VP_(VP), params_(params), qvec_(qvec) {}
    
    static ceres::CostFunction* Create(const V3D& VP, const double* params = NULL, const double* qvec = NULL) {
        if (!params && !qvec)
            return new ceres::AutoDiffCostFunction<VPConstraintsFunctor, 1, 4, 2, CameraModel::kNumParams, 4>(new VPConstraintsFunctor(VP, NULL, NULL));
        else
            return new ceres::AutoDiffCostFunction<VPConstraintsFunctor, 1, 4, 2>(new VPConstraintsFunctor(VP, params, qvec));
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, T* residuals) const {
        CHECK_NOTNULL(params_);
        CHECK_NOTNULL(qvec_);

        const int num_params = CameraModel::kNumParams;
        T params[num_params];
        for (size_t i = 0; i < num_params; ++i) {
            params[i] = T(params_[i]);
        }
        T qvec[4] = {T(qvec_[0]), T(qvec_[1]), T(qvec_[2]), T(qvec_[3])};
        return (*this)(uvec, wvec, params, qvec, residuals);
    }

    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, const T* const params, const T* const qvec, T* residuals) const {
        T kvec[4];
        ParamsToKvec(CameraModel::model_id, params, kvec);
        T dir3d[3], m[3];
        MinimalPluckerToPlucker<T>(uvec, wvec, dir3d, m);
        T dir3d_rotated[3];
        ceres::QuaternionRotatePoint(qvec, dir3d, dir3d_rotated);

        const V3D& vp = VP_;
        T vpvec[3] = {T(vp[0]), T(vp[1]), T(vp[2])};
        T direc[3];
        GetDirectionFromVP<T>(vpvec, kvec, direc);
        residuals[0] = CeresComputeDist3D_sine(dir3d_rotated, direc);
        return true;
    }

protected:
    V3D VP_;
    const double* params_;
    const double* qvec_;
};

////////////////////////////////////////////////////////////
// Geometric Refinement
////////////////////////////////////////////////////////////

template <typename T>
void Ceres_PerpendicularDist2D(const T coor[3], const T p1[2], const T p2[2], T* res) {
    T direc_norm = ceres::sqrt(coor[0] * coor[0] + coor[1] * coor[1] + EPS);
    T dist1 = (p1[0] * coor[0] + p1[1] * coor[1] + coor[2]) / direc_norm;
    T dist2 = (p2[0] * coor[0] + p2[1] * coor[1] + coor[2]) / direc_norm;
    res[0] = dist1; res[1] = dist2;
}

template <typename T>
void Ceres_CosineWeightedPerpendicularDist2D_1D(const T coor[3], const T p1[2], const T p2[2], T* res, const double alpha=10.0) {
    T direc_norm = ceres::sqrt(coor[0] * coor[0] + coor[1] * coor[1] + EPS);
    // 2D direction of the projection
    T dir2d[2];
    dir2d[0] = -coor[1] / direc_norm;
    dir2d[1] = coor[0] / direc_norm;
    // 2D direction of the 2D line segment
    T direc[2];
    direc[0] = p2[0] - p1[0];
    direc[1] = p2[1] - p1[1];
    // compute weight
    T cosine = CeresComputeDist2D_cosine(dir2d, direc);
    const T alpha_t = T(alpha);
    T weight = ceres::exp(alpha_t * (T(1.0) - cosine));
    // compute raw distance and multiply it by the weight
    Ceres_PerpendicularDist2D(coor, p1, p2, res);
    res[0] *= weight; res[1] *= weight;
}

// NOTICE 线段的成本函数costfunction 用于优化和调整3D线段的位置和方向，使其更准确地与其在多个图像中的观测相匹配
template <typename CameraModel>
struct GeometricRefinementFunctor {
public:
    GeometricRefinementFunctor(const Line2d& line2d, const double* params = NULL, const double* qvec = NULL, const double* tvec = NULL, const double alpha = 10.0):
        line2d_(line2d), params_(params), qvec_(qvec), tvec_(tvec), alpha_(alpha) {}
    
    // 工厂函数 - 创建 GeometricRefinementFunctor 实例，初始化一个自动微分的成本函数
    static ceres::CostFunction* Create(const Line2d& line2d, const double* params = NULL, const double* qvec = NULL, const double* tvec = NULL, const double alpha = 10.0) {
        // OS A 相机内参、相机位姿都为NULL时（默认） - 优化相机内外参及3D线段位置
        if (!params && !qvec && !tvec) 
            // 2(残差维度), 4(3D线段旋转), 2(3D线段平移), CameraModel::kNumParams(相机内参), 4(相机姿态), 3（相机位置）
            // TODO cout 
            return new ceres::AutoDiffCostFunction<GeometricRefinementFunctor, 2, 4, 2, CameraModel::kNumParams, 4, 3>(new GeometricRefinementFunctor(line2d, NULL, NULL, NULL, alpha));
        // OS B 当提供了相机内参、相机位姿中的任意一个时，认为该信息已经足够准确，只专注于优化uvec 和 wvec
        else 
            // 2(残差维度), 4(3D线段旋转), 2(3D线段平移)
            return new ceres::AutoDiffCostFunction<GeometricRefinementFunctor, 2, 4, 2>(new GeometricRefinementFunctor(line2d, params, qvec, tvec, alpha));
    }

    // OS B 当提供了相机内参、相机位姿中的任意一个时，认为该信息已经足够准确，只专注于优化uvec 和 wvec
    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec, T* residuals) const {
        CHECK_NOTNULL(params_);
        CHECK_NOTNULL(qvec_);
        CHECK_NOTNULL(tvec_);

        const int num_params = CameraModel::kNumParams;
        T params[num_params];
        for (size_t i = 0; i < num_params; ++i) {
            params[i] = T(params_[i]);
        }
        T qvec[4] = {T(qvec_[0]), T(qvec_[1]), T(qvec_[2]), T(qvec_[3])};
        T tvec[3] = {T(tvec_[0]), T(tvec_[1]), T(tvec_[2])};
        return (*this)(uvec, wvec, params, qvec, tvec, residuals);
    }

    // OS A 相机内参、相机位姿都为NULL时（默认） - 优化相机内外参及3D线段位置
    template <typename T>
    bool operator()(const T* const uvec, const T* const wvec,                        // 3D线段坐标 - 最小普吕克坐标
                    const T* const params, const T* const qvec, const T* const tvec, // 相机内外参
                    T* residuals) const {                                            // 重投影误差
        // 获取相机内参 [fx,fy,cx,cy]
        T kvec[4];
        ParamsToKvec(CameraModel::model_id, params, kvec);
        // 转换后的普吕克坐标，分别表示线段的方向和一点的位置
        T dvec[3], mvec[3];
        MinimalPluckerToPlucker<T>(uvec, wvec, dvec, mvec);      // ! 将简化的普吕克向量，转化为标准普吕克向量
        //  直线在像素平面上的位置（2D坐标）
        T coor[3];
        Line_WorldToPixel<T>(kvec, qvec, tvec, dvec, mvec, coor);// !

        // 读取2D线段在图像上的起点和终点
        const Line2d& line = line2d_;
        T p1[2] = {T(line.start(0)), T(line.start(1))};
        T p2[2] = {T(line.end(0)), T(line.end(1))};

        // 计算线段在图像平面上的投影与实际观测线段之间的垂直距离，同时考虑了方向余弦的权重，这有助于减少方向偏差对误差的影响。
        // coor:3D直线投影回像素平面的直线
        // p1,p2: 像素平面上2D线段的起点和终点
        // residuals: 重投影误差
        // alpha: 各个角度的权重 - for weighting angle
        Ceres_CosineWeightedPerpendicularDist2D_1D(coor, p1, p2, residuals, alpha_);// !
        return true;
    }

// 成员变量
protected:
    Line2d line2d_;         // 2D线段 - 3D线段在图像中观测的二维投影
    const double* params_;  // 相机内参
    const double* qvec_;    // 相机姿态
    const double* tvec_;    // 相机位置
    double alpha_;          // 各个角度的权重 - for weighting angle
};

} // namespace line_refinement

} // namespace optimize

} // namespace limap

#endif

