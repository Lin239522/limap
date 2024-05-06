#include "optimize/hybrid_bundle_adjustment/hybrid_bundle_adjustment.h"
#include "optimize/line_refinement/cost_functions.h"
#include "optimize/hybrid_bundle_adjustment/cost_functions.h"
#include "base/camera_models.h"
#include "ceresbase/parameterization.h"

#include <colmap/util/logging.h>
#include <colmap/util/threading.h>
#include <colmap/util/misc.h>
#include <colmap/optim/bundle_adjustment.h>
#include <colmap/base/cost_functions.h>

namespace limap {

namespace optimize {

namespace hybrid_bundle_adjustment {

void HybridBAEngine::InitPointTracks(const std::vector<PointTrack>& point_tracks) {
    point_tracks_.clear();
    points_.clear();
    size_t num_tracks = point_tracks.size();
    for (size_t track_id = 0; track_id < num_tracks; ++track_id) {
        point_tracks_.insert(std::make_pair(track_id, point_tracks[track_id]));
        points_.insert(std::make_pair(track_id, point_tracks[track_id].p));
    }
}

void HybridBAEngine::InitPointTracks(const std::map<int, PointTrack>& point_tracks) {
    point_tracks_.clear();
    points_.clear();
    point_tracks_ = point_tracks;
    for (auto it = point_tracks.begin(); it != point_tracks.end(); ++it) {
        points_.insert(std::make_pair(it->first, it->second.p));
    }
}

void HybridBAEngine::InitLineTracks(const std::vector<LineTrack>& line_tracks) {
    line_tracks_.clear();
    lines_.clear();
    size_t num_tracks = line_tracks.size();
    for (size_t track_id = 0; track_id < num_tracks; ++track_id) {
        line_tracks_.insert(std::make_pair(track_id, line_tracks[track_id]));
        lines_.insert(std::make_pair(track_id, MinimalInfiniteLine3d(line_tracks[track_id].line)));
    }
}

void HybridBAEngine::InitLineTracks(const std::map<int, LineTrack>& line_tracks) {
    line_tracks_.clear();
    lines_.clear();
    line_tracks_ = line_tracks;
    for (auto it = line_tracks.begin(); it != line_tracks.end(); ++it) {
        lines_.insert(std::make_pair(it->first, MinimalInfiniteLine3d(it->second.line)));
    }
}

void HybridBAEngine::ParameterizeCameras() {
    for (const int& img_id: imagecols_.get_img_ids()) {
        double* params_data = imagecols_.params_data(img_id);
        double* qvec_data = imagecols_.qvec_data(img_id);
        double* tvec_data = imagecols_.tvec_data(img_id);

        if (!problem_->HasParameterBlock(params_data))
                continue;
        if (config_.constant_intrinsics) {
            problem_->SetParameterBlockConstant(params_data);
        }
        else if (config_.constant_principal_point) {
            int cam_id = imagecols_.camimage(img_id).cam_id;
            std::vector<int> const_idxs;
            const std::vector<size_t>& principal_point_idxs = imagecols_.cam(cam_id).PrincipalPointIdxs();
            const_idxs.insert(const_idxs.end(), principal_point_idxs.begin(), principal_point_idxs.end());
            SetSubsetManifold(imagecols_.cam(cam_id).params().size(), const_idxs, problem_.get(), params_data); 
        }

        if (config_.constant_pose) {
            if (!problem_->HasParameterBlock(qvec_data))
                continue;
            problem_->SetParameterBlockConstant(qvec_data);
            if (!problem_->HasParameterBlock(tvec_data))
                continue;
            problem_->SetParameterBlockConstant(tvec_data);
        }
        else {
            if (!problem_->HasParameterBlock(qvec_data))
                continue;
            SetQuaternionManifold(problem_.get(), qvec_data);
        }
    }
}

void HybridBAEngine::ParameterizePoints() {
    for (auto it = points_.begin(); it != points_.end(); ++it) {
        double* point_data = it->second.data();
        if (!problem_->HasParameterBlock(point_data))
            continue;
        if (config_.constant_point)
            problem_->SetParameterBlockConstant(point_data);
    }
}

void HybridBAEngine::ParameterizeLines() {
    for (auto it = line_tracks_.begin(); it != line_tracks_.end(); ++it) {
        int track_id = it->first;
        size_t n_images = it->second.count_images();
        double* uvec_data = lines_[track_id].uvec.data();
        double* wvec_data = lines_[track_id].wvec.data();
        if (!problem_->HasParameterBlock(uvec_data) || !problem_->HasParameterBlock(wvec_data))
            continue;
        if (config_.constant_line || n_images < config_.min_num_images) {
            problem_->SetParameterBlockConstant(uvec_data);
            problem_->SetParameterBlockConstant(wvec_data);
        }
        else {
            SetQuaternionManifold(problem_.get(), uvec_data);
            SetSphereManifold<2>(problem_.get(), wvec_data);
        }
    }
}
// NOTICE 设置每个3D点的残差 - 将该3D点对应的所有2D点的重投影误差加入优化问题中
void HybridBAEngine::AddPointGeometricResiduals(const int track_id) {
    // 如果点的权重 <= 0，则此函数不进行任何操作，直接返回
    if (config_.lw_point <= 0)
        return;
    // step 1 [获取点轨迹] 根据track_id，获取点轨迹 （PointTrack代表了一个3D点在多个图像中的观测信息）
    const PointTrack& track = point_tracks_.at(track_id);  // 根据ID来访问点
    // step 2 [获取损失函数] 从配置中获取损失函数（防止离群点对整体估计的影响）
    ceres::LossFunction* loss_function = config_.point_geometric_loss_function.get();
    // step 3 [遍历所有图像] 遍历该3D点对应的图像们，将对应的2D点重投影误差加入残差项
    for (size_t i = 0; i < track.count_images(); ++i) { 
        // step 3.1 获取图像信息
        int img_id = track.image_id_list[i];                        // 图像ID
        int model_id = imagecols_.camview(img_id).cam.ModelId();    // 当前图像对应的相机模型ID
        V2D p2d = track.p2d_list[i];                                // 图像中点的二维投影

        // step 3.2 选择并创建成本函数cost_function（PointGeometricRefinementFunctor 计算重投影误差）
        ceres::CostFunction* cost_function = nullptr;
        switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = PointGeometricRefinementFunctor<CameraModel>::Create(p2d, NULL, NULL, NULL); \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
        }
        // step 3.3 根据权重缩放损失函数：新损失函数scaled_loss_function = 原始损失函数 * 点的权重
        ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, config_.lw_point, ceres::DO_NOT_TAKE_OWNERSHIP);
        // step 3.4 [添加残差块] 将[成本函数cost_function]和[损失函数scaled_loss_function]添加到BA问题里，连接到相应的参数块（点的3D坐标、相机参数、相机pose）
        ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, scaled_loss_function, 
                                                                     points_.at(track_id).data(),
                                                                     imagecols_.params_data(img_id), imagecols_.qvec_data(img_id), imagecols_.tvec_data(img_id));

    }
}

// NOTICE 设置每个3D线的残差，添加到优化问题中
void HybridBAEngine::AddLineGeometricResiduals(const int track_id) {
    // step 1 [获取线轨迹] 根据track_id，获取线轨迹 （LineTrack代表了一条3D线段在多个2D图像中的观测信息）
    const LineTrack& track = line_tracks_.at(track_id);
    // step 2 [获取损失函数] 从配置中获取用于线段几何残差的损失函数
    ceres::LossFunction* loss_function = config_.line_geometric_loss_function.get();

    // step 3 [计算线权重] 为该3D线段对应的每一条2D线段设置权重
    auto idmap = track.GetIdMap();  // std::map<int, std::vector<int>> - <图像ID,该图像中的2D线段ID>
    std::vector<double> weights;
    ComputeLineWeights(track, weights); // 为该3D线段对应的每一条2D线段设置权重，2D线段越长，表示越接近于正视拍摄，权重越大

    // add to problem for each supporting image (for each supporting line)
    // step 4 [遍历所有图像] 遍历该3D线段对应的图像们，将对应的2D线重投影误差加入残差项
    auto& minimal_line = lines_.at(track_id);
    std::vector<int> image_ids = track.GetSortedImageIds();  // 调整3D线段对应图像的优化顺序
    for (auto it1 = image_ids.begin(); it1 != image_ids.end(); ++it1) {
        // step 4.1 获取图像信息
        int img_id = *it1;                                       // 当前图像ID
        int model_id = imagecols_.camview(img_id).cam.ModelId(); // 当前图像对应的相机模型ID
        const auto& ids = idmap.at(img_id);                      // 当前图像中所有2D线段的索引
        // step 4.2 [遍历图像中的每条2D线段] 
        for (auto it2 = ids.begin(); it2 != ids.end(); ++it2) {
            const Line2d& line = track.line2d_list[*it2];         // 当前2D线段
            double weight = weights[*it2];                        // 获取当前2D图像对应的权重
            // step 4.2.1 选择并创建成本函数cost_function（PointGeometricRefinementFunctor 计算重投影误差）
            ceres::CostFunction* cost_function = nullptr;
            switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
    case CameraModel::kModelId:        \
        cost_function = line_refinement::GeometricRefinementFunctor<CameraModel>::Create(line, NULL, NULL, NULL, config_.geometric_alpha); \
        break;
            LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
            }

            // step 4.2.2 根据权重缩放损失函数：新损失函数scaled_loss_function = 原始损失函数 * 该2D线段权重
            ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
            // step 4.2.3 [添加残差块] 将[成本函数cost_function]和[损失函数scaled_loss_function]添加到BA问题里，连接到相应的参数块（点的3D坐标、相机参数、相机pose）
            ceres::ResidualBlockId block_id = problem_->AddResidualBlock(cost_function, scaled_loss_function, 
                                                                         minimal_line.uvec.data(), minimal_line.wvec.data(),  // ? 线是怎么表示的？一个旋转四元数+一个2D向量？
                                                                         imagecols_.params_data(img_id), imagecols_.qvec_data(img_id), imagecols_.tvec_data(img_id));
        }
    }
}

// NOTICE 对应python文件中的ba_engine.SetUp() # 设置问题结构
void HybridBAEngine::SetUp() {
    // step 1 setup problem 初始化问题实例
    problem_.reset(new ceres::Problem(config_.problem_options));

    // step 2 add residuals 设置残差项
        // step 2.1: 【点残差】point geometric residual 
        // 如果配置不是将点、内参、和姿态都设置为常量，则遍历点轨迹（point_tracks_），为每个3D点添加几何残差。
    if (!config_.constant_point || !config_.constant_intrinsics || !config_.constant_pose) { // 配置中point、pose、intrinsics中有一个为FALSE(需要优化)
        for (auto it = point_tracks_.begin(); it != point_tracks_.end(); ++it) {
            int point3d_id = it->first;
            AddPointGeometricResiduals(point3d_id);
        }
    }
        // step 2.2: 【线残差】line geometric residual
        // 如果配置不是将线、内参、和姿态都设置为常量，则遍历线轨迹（line_tracks_），为每个线添加几何残差。
    if (!config_.constant_line || !config_.constant_intrinsics || !config_.constant_pose) { // 配置中line、pose、intrinsics中有一个为FALSE(需要优化)
        for (auto it = line_tracks_.begin(); it != line_tracks_.end(); ++it) {
            int line3d_id = it->first;                  // std::map<int, LineTrack> line_tracks_;
            AddLineGeometricResiduals(line3d_id);
        }
    }

    // step 3 parameterization 参数化相机、点云、线条
    ParameterizeCameras();
    ParameterizePoints();
    ParameterizeLines();
}

//  对应python文件中的ba_engine.Solve() # 进行优化求解
// NOTICE
bool HybridBAEngine::Solve() {
    if (problem_->NumResiduals() == 0)
        return false;
    ceres::Solver::Options solver_options = config_.solver_options;
    
    // Empirical choice.
    const size_t kMaxNumImagesDirectDenseSolver = 50;
    const size_t kMaxNumImagesDirectSparseSolver = 900;
    const size_t num_images = imagecols_.NumImages();
    if (num_images <= kMaxNumImagesDirectDenseSolver) {
        solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
        solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {  // Indirect sparse (preconditioned CG) solver.
        solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
    }

    solver_options.num_threads =
        colmap::GetEffectiveNumThreads(solver_options.num_threads);
    #if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads =
        colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
    #endif  // CERES_VERSION_MAJOR

    std::string solver_error;
    CHECK(solver_options.IsValid(&solver_error)) << solver_error;

    ceres::Solve(solver_options, problem_.get(), &summary_);
    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (config_.print_summary) {
        colmap::PrintHeading2("Optimization report");
        colmap::PrintSolverSummary(summary_); // We need to replace this with our own Printer!!!
    }
    return true;
}

std::map<int, V3D> HybridBAEngine::GetOutputPoints() const {
    std::map<int, V3D> outputs;
    for (auto it = point_tracks_.begin(); it != point_tracks_.end(); ++it) {
        int track_id = it->first;
        outputs.insert(std::make_pair(track_id, points_.at(track_id)));
    }
    return outputs;
}

std::map<int, PointTrack> HybridBAEngine::GetOutputPointTracks() const {
    std::map<int, PointTrack> outputs;
    for (auto it = point_tracks_.begin(); it != point_tracks_.end(); ++it) {
        int track_id = it->first;
        PointTrack track = it->second;
        track.p = points_.at(track_id);
        outputs.insert(std::make_pair(track_id, track));
    }
    return outputs;
}

std::map<int, Line3d> HybridBAEngine::GetOutputLines(const int num_outliers) const {
    std::map<int, Line3d> outputs;
    std::map<int, LineTrack> output_line_tracks =  GetOutputLineTracks(num_outliers);
    for (auto it = output_line_tracks.begin(); it != output_line_tracks.end(); ++it) {
        outputs.insert(std::make_pair(it->first, it->second.line));
    }
    return outputs;
}

std::map<int, LineTrack> HybridBAEngine::GetOutputLineTracks(const int num_outliers) const {
    std::map<int, LineTrack> outputs;
    for (auto it = line_tracks_.begin(); it != line_tracks_.end(); ++it) {
        int track_id = it->first;
        LineTrack track = it->second;
        Line3d line = GetLineSegmentFromInfiniteLine3d(lines_.at(track_id).GetInfiniteLine(), track.line3d_list, num_outliers);
        track.line = line;
        outputs.insert(std::make_pair(track_id, track));
    }
    return outputs;
}

} // namespace hybrid_bundle_adjustment 

} // namespace optimize 

} // namespace limap

