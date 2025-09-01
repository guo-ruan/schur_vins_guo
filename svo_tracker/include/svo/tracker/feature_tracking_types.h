#pragma once  
  
#include <glog/logging.h>  
#include <svo/common/types.h>  
  
namespace svo {  
  
// -----------------------------------------------------------------------------  
/// 特征跟踪器配置选项结构体  
/// 包含 KLT (Kanade-Lucas-Tomasi) 跟踪算法的所有参数设置  
struct FeatureTrackerOptions  
{  
  /// 金字塔 Lucas-Kanade 跟踪的最粗糙层级  
  /// 对于 640x480 分辨率图像设为 4，分辨率翻倍时此数值加 1  
  /// 更高的层级意味着更粗糙的搜索，但计算更快  
  int klt_max_level = 4;  
  
  /// 搜索的最粗糙层级，类似于 klt_max_level  
  /// 对于高分辨率图像，如果不需要提取到最底层特征，可以将此值设为大于 0  
  /// 这样可以跳过最精细的层级以提高性能  
  int klt_min_level = 0;  
  
  /// 每个金字塔层级使用的补丁（patch）大小  
  /// 数组长度应与金字塔层级数匹配，较高层级通常使用较大补丁  
  std::vector<int> klt_patch_sizes = {16, 16, 16, 8, 8};  
  
  /// KLT 算法的最大迭代次数（终止条件之一）  
  /// 防止算法在难以收敛的情况下无限循环  
  int klt_max_iter = 30;  
  
  /// KLT 算法的最小更新阈值（终止条件之一）  
  /// 当更新量的平方小于此值时停止迭代，表示已收敛  
  double klt_min_update_squared = 0.001;  
  
  /// 是否使用首次观测作为 KLT 模板  
  /// 设为 false 则使用最后观测作为模板，但会导致更多特征漂移  
  /// 使用首次观测可以保持更好的跟踪一致性  
  bool klt_template_is_first_observation = true;  
  
  /// 当活跃跟踪数量低于此阈值时，触发新特征检测  
  /// 确保系统始终有足够的特征进行跟踪  
  size_t min_tracks_to_detect_new_features = 50;  
  
  /// 检测新特征前是否重置跟踪器  
  /// 设为 true 意味着所有活跃轨迹始终保持相同年龄  
  /// 这有助于保持跟踪的一致性  
  bool reset_before_detection = true;  
};  
  
// -----------------------------------------------------------------------------  
/// 特征引用类  
/// 提供对特定帧束中特定帧的特定特征的轻量级引用  
/// 避免直接存储特征数据，而是通过索引进行间接访问  
class FeatureRef  
{  
public:  
  /// 禁用默认构造函数，强制提供必要的初始化参数  
  FeatureRef() = delete;  
  
  /// 构造函数：创建特征引用  
  /// @param frame_bundle 包含特征的帧束指针  
  /// @param frame_index 帧束中的帧索引  
  /// @param feature_index 帧中的特征索引  
  FeatureRef(  
      const FrameBundlePtr& frame_bundle, size_t frame_index, size_t feature_index);  
  
  /// 获取包含此特征的帧束指针  
  /// @return 帧束的共享指针  
  inline const FrameBundlePtr getFrameBundle() const {  
    return frame_bundle_;  
  }  
  
  /// 获取帧束中的帧索引  
  /// @return 帧在帧束中的位置索引  
  inline size_t getFrameIndex() const {  
    return frame_index_;  
  }  
  
  /// 获取帧中的特征索引  
  /// @return 特征在帧中的位置索引  
  inline size_t getFeatureIndex() const {  
    return feature_index_;  
  }  
  
  /// 获取特征的像素坐标（2D 图像坐标）  
  /// @return 2x1 的 Eigen 块，包含 x, y 像素坐标  
  const Eigen::Block<Keypoints, 2, 1> getPx() const;  
  
  /// 获取特征的归一化方位向量（3D 单位向量）  
  /// @return 3x1 的 Eigen 块，表示从相机中心指向特征的单位向量  
  const Eigen::Block<Bearings, 3, 1> getBearing() const;  
  
  /// 获取包含此特征的帧指针  
  /// @return 帧的共享指针  
  const FramePtr getFrame() const;  
  
private:  
  FrameBundlePtr frame_bundle_;  ///< 帧束指针  
  size_t frame_index_;           ///< 帧索引  
  size_t feature_index_;         ///< 特征索引  
};  
  
/// 特征引用列表的类型别名  
/// 用于存储多个特征引用  
typedef std::vector<FeatureRef> FeatureRefList;  
  
  
// -----------------------------------------------------------------------------  
/// 特征轨迹类  
/// 管理单个特征在时间序列上的完整跟踪轨迹  
/// 每个轨迹包含该特征在不同帧中的所有观测  
class FeatureTrack  
{  
public:  
  /// Eigen 内存对齐宏，确保正确的内存对齐以优化性能  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
  
  /// 构造函数：创建新的特征轨迹  
  /// @param track_id 轨迹的唯一标识符  
  explicit FeatureTrack(int track_id);  
  
  /// 获取轨迹的唯一标识符  
  /// @return 轨迹 ID  
  inline int getTrackId() const {  
    return track_id_;  
  }  
  
  /// 获取完整的特征轨迹（所有观测的引用列表）  
  /// @return 包含所有特征观测引用的常量引用  
  inline const FeatureRefList& getFeatureTrack() const {  
    return feature_track_;  
  }  
  
  /// 获取轨迹中观测的数量  
  /// @return 轨迹长度（观测次数）  
  inline size_t size() const {  
    return feature_track_.size();  
  }  
  
  /// 检查轨迹是否为空（无观测）  
  /// @return 如果轨迹为空返回 true  
  inline bool empty() const {  
    return feature_track_.empty();  
  }  
  
  /// 获取轨迹中的第一个观测（最早观测到的特征）  
  /// 用于访问特征首次被检测到时的信息  
  /// @return 第一个特征观测的常量引用  
  inline const FeatureRef& front() const {  
    CHECK(!empty()) << "Track empty when calling front().";  
    return feature_track_.front();  
  }  
  
  /// 获取轨迹中的最后一个观测（最近观测到的特征）  
  /// 用于访问特征最新的观测信息  
  /// @return 最后一个特征观测的常量引用  
  inline const FeatureRef& back() const {  
    CHECK(!empty()) << "Track empty when calling back().";  
    return feature_track_.back();  
  }  
  
  /// 获取轨迹中指定索引的观测  
  /// @param i 观测在轨迹中的索引位置  
  /// @return 指定位置的特征观测的常量引用  
  inline const FeatureRef& at(size_t i) const {  
    CHECK_LT(i, feature_track_.size()) << "Index too large.";  
    return feature_track_.at(i);  
  }  
  
  /// 在轨迹末尾添加新的观测  
  /// 新观测总是插入到向量的末尾，保持时间顺序  
  /// @param frame_bundle 包含新观测的帧束指针  
  /// @param frame_index 帧束中的帧索引  
  /// @param feature_index 帧中的特征索引  
  inline void pushBack(  
      const FrameBundlePtr& frame_bundle,  
      const size_t frame_index,  
      const size_t feature_index) {  
    feature_track_.emplace_back(FeatureRef(frame_bundle, frame_index, feature_index));  
  }  
  
  /// 计算轨迹的视差（第一个和最后一个观测之间的像素距离）  
  /// 视差越大表示特征跟踪质量越好，用于评估轨迹的可靠性  
  /// @return 首尾观测间的欧几里得距离（像素单位）  
  double getDisparity() const;  
  
private:  
  int track_id_;                 ///< 轨迹的唯一标识符  
  FeatureRefList feature_track_; ///< 存储所有观测引用的向量  
};  
  
/// 特征轨迹向量的类型别名  
/// 使用 Eigen 对齐分配器确保内存对齐，提高性能  
typedef std::vector<FeatureTrack,  
Eigen::aligned_allocator<FeatureTrack> > FeatureTracks;  
  
} // namespace svo