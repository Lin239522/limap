#ifndef LIMAP_BASE_LINETRACK_H_
#define LIMAP_BASE_LINETRACK_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <set>
#include <map>

namespace py = pybind11;

#include "util/types.h"
#include "base/camera_view.h"
#include "base/linebase.h"
#include "base/infinite_line.h"

namespace limap {

class LineTrack {
public:
    // 构造函数
    LineTrack() {}                      // 默认构造函数
    LineTrack(const LineTrack& track);  // 拷贝构造函数
    LineTrack(const Line3d& line_, const std::vector<int>& image_id_list_, const std::vector<int>& line_id_list_, const std::vector<Line2d>& line2d_list_): 
        line(line_), image_id_list(image_id_list_), line_id_list(line_id_list_), line2d_list(line2d_list_) {}  // 带参数的构造函数，直接初始化所有成员变量。
    // python接口相关函数
    py::dict as_dict() const;  // LineTrack->python字典
    LineTrack(py::dict dict);  // python字典->LineTrack

    // properties 成员变量
    Line3d line;                        // 3D线段
    std::vector<int> image_id_list;     // 包含此3D线段的所有图像的ID
    std::vector<int> line_id_list;      // 每个图像中对应的2D线段的ID
    std::vector<Line2d> line2d_list;    // 各个图像中2D线段坐标的向量列表
    
    // auxiliary information (may not be initialized)
    std::vector<int> node_id_list;
    std::vector<Line3d> line3d_list;
    std::vector<double> score_list;

    // active status for recursive merging
    bool active = true;
    
    // HACK 结构性语义信息
    bool is_horizontal = false; // 水平线段
    bool is_plumb = false;      // 铅直线段

    size_t count_lines() const {return line2d_list.size();} // 该3D线段在所有图像中的2D表示的数量
    std::vector<int> GetSortedImageIds() const;             // 返回一个包含所有图像ID的有序列表，通常用于确定优化处理的顺序
    std::map<int, int> GetIndexMapforSorted() const;
    std::vector<int> GetIndexesforSorted() const;
    size_t count_images() const {return GetSortedImageIds().size(); } // 线段在多少个不同的图像中被观察到。
    std::vector<Line2d> projection(const std::vector<CameraView>& views) const; // 计算线段在给定一组相机视图下的投影
    std::map<int, std::vector<int>> GetIdMap() const; // (img_id, {index})      // 返回一个映射，键为图像ID，值为该图像中线段的索引列表，用于管理和访问线段数据

    void Resize(const size_t& n_lines);
    bool HasImage(const int& image_id) const;
    void Read(const std::string& filename);
    void Write(const std::string& filename) const;
};

////////////////////////////////////////////////////////////
// sampling for optimization
////////////////////////////////////////////////////////////
void ComputeLineWeights(const LineTrack& track,
                        std::vector<double>& weights); // weights.size() == track.count_lines()

void ComputeLineWeightsNormalized(const LineTrack& track,
                                  std::vector<double>& weights); // weights.size() == track.count_lines()

void ComputeHeatmapSamples(const LineTrack& track, 
                           std::vector<std::vector<InfiniteLine2d>>& heatmap_samples, // samples for each line
                           const std::pair<double, double> sample_range,
                           const int n_samples); 

void ComputeFConsistencySamples(const LineTrack& track,
                                const std::map<int, CameraView>& views, // {img_id, view}
                                std::vector<std::tuple<int, InfiniteLine2d, std::vector<int>>>& fconsis_samples, // [ref_image_id, sample, {tgt_image_id(s)}]
                                const std::pair<double, double> sample_range,
                                const int n_samples);

} // namespace limap

#endif

