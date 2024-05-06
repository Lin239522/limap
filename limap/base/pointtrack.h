#ifndef LIMAP_BASE_POINTTRACK_H_
#define LIMAP_BASE_POINTTRACK_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <cmath>
#include <set>
#include <map>

namespace py = pybind11;

#include "util/types.h"
#include "_limap/helpers.h"

namespace limap {

template <typename PTYPE>
struct Feature2dWith3dIndex {
    Feature2dWith3dIndex() {}
    Feature2dWith3dIndex(PTYPE p_, int point3D_id_ = -1): p(p_), point3D_id(point3D_id_) {}
    Feature2dWith3dIndex(py::dict dict) {
        ASSIGN_PYDICT_ITEM(dict, p, PTYPE)
        ASSIGN_PYDICT_ITEM(dict, point3D_id, int)
    }
    py::dict as_dict() const {
        py::dict output;
        output["p"] = p;
        output["point3D_id"] = point3D_id;
        return output;
    }
    PTYPE p;
    int point3D_id = -1;
};
typedef Feature2dWith3dIndex<V2D> Point2d;

// 存储和管理一个3D点及其在多个图像中的2D观测信息，为BA提供了必要信息
class PointTrack { 
public:
    // 构造函数
    PointTrack() {}                      // 默认构造函数
    PointTrack(const PointTrack& track); // 拷贝构造函数，用于创建一个新的 PointTrack 对象，其内容是现有 PointTrack 对象的复制
    PointTrack(const V3D& p_, const std::vector<int>& image_id_list_, const std::vector<int>& p2d_id_list_, const std::vector<V2D> p2d_list_): p(p_), image_id_list(image_id_list_), p2d_id_list(p2d_id_list_), p2d_list(p2d_list_) {} // 带参数的构造函数，直接初始化所有成员变量。
    // python接口相关函数
    py::dict as_dict() const;
    PointTrack(py::dict dict);
    
    // 成员变量
    V3D p;                          // 3D点坐标的向量
    std::vector<int> image_id_list; // 包含此3D点的所有图像的ID
    std::vector<int> p2d_id_list;   // 每个图像中对应的2D点的ID
    std::vector<V2D> p2d_list;      // 各个图像中2D点坐标的向量列表

    size_t count_images() const { return image_id_list.size(); } // 该3D点在多少个图像中被观测到
};

} // namespace limap

#endif

