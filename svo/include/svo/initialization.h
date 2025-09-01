// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <svo/common/types.h>           // 包含基础类型定义
#include <svo/common/transformation.h>  // 包含坐标变换相关定义
#include <svo/common/camera_fwd.h>      // 相机相关前向声明

namespace svo {

// 前向声明
class StereoTriangulation;
class FeatureTracker;
using FeatureTrackerUniquePtr = std::unique_ptr<FeatureTracker>;
class AbstractDetector;
struct DetectorOptions;
struct FeatureTrackerOptions;

// 初始化方法枚举
enum class InitializerType {
  kHomography,       ///< 通过前两帧估计平面（使用单应性矩阵）
  kTwoPoint,         ///< 假设已知IMU旋转，估计平移（两点法）
  kFivePoint,        ///< 使用5点RANSAC估计两相机相对位姿
  kOneShot,          ///< 在给定深度的平面上初始化特征点（用于自主起飞）
  kStereo,           ///< 通过已知位姿的两帧三角化初始化
  kArrayGeometric,   ///< 使用17点RANSAC估计多相机阵列的相对位姿
  kArrayOptimization ///< 使用GTSAM优化估计多相机阵列的相对位姿
};

/// 所有初始化器的通用参数配置
/// 标记为(!)的参数更为重要
struct InitializationOptions
{
  /// 初始化方法类型（见上述枚举）
  InitializerType init_type = InitializerType::kHomography;

  /// (!) 初始化第二帧所需的最小视差（特征轨迹长度）
  /// 建议值：50像素。较小的值（如20像素）能加快初始化，但较大视差更有利于生成高质量点云
  double init_min_disparity = 50.0;

  /// 当检查特征视差时，要求至少有init_disparity_pivot_ratio比例的特征满足
  /// init_min_disparity阈值。例如0.5表示中位数检查，0.25表示至少25%的特征满足
  double init_disparity_pivot_ratio = 0.5;

  /// (!) 如果第一帧提取的特征数小于该阈值，则丢弃该帧并检查下一帧
  size_t init_min_features = 100;

  /// (!) 初始化时尝试提取的特征数 = init_min_features * init_min_features_factor
  double init_min_features_factor = 2.5;

  /// (!) 初始化过程中跟踪的特征数低于该阈值时返回失败
  size_t init_min_tracked = 50;

  /// (!) 三角化后，若重投影误差大于reproj_error_thresh的点视为离群点
  /// 只有当内点数超过init_min_inliers时才返回成功
  size_t init_min_inliers = 40;

  /// 重投影误差阈值（像素）与姿态优化器中使用的相同
  double reproj_error_thresh = 2.0;

  // TODO: 这些参数的作用尚不明确（由Art引入）
  double expected_avg_depth = 1.0;        // 预期平均深度
  double init_min_depth_error = 1.0;      // 初始化最小深度误差
};

// 初始化结果枚举
enum class InitResult
{
  kFailure,         // 初始化失败
  kNoKeyframe,      // 未选择关键帧
  kTracking,        // 跟踪失败
  kSuccess          // 初始化成功
};

/// 从前两帧构建地图的抽象初始化类
class AbstractInitialization
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Eigen内存对齐宏

  // 类型别名
  using BearingVectors = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;  // 3D方向向量容器
  using Ptr = std::shared_ptr<AbstractInitialization>;  // 共享指针
  using UniquePtr = std::unique_ptr<AbstractInitialization>;  // 唯一指针
  using InlierMask = Eigen::Matrix<bool, Eigen::Dynamic, 1, Eigen::ColMajor>;  // 内点掩码
  using FeatureMatches = std::vector<std::pair<size_t, size_t>>;  // 特征匹配对

  InitializationOptions options_;       //!< 初始化参数
  FeatureTrackerUniquePtr tracker_;     //!< 特征追踪器
  FrameBundlePtr frames_ref_;           //!< 参考帧集合
  Transformation T_cur_from_ref_;       //!< 两帧间的计算变换
  Quaternion R_ref_world_;              // 参考帧世界坐标系旋转先验
  Quaternion R_cur_world_;              //!< 当前帧世界坐标系绝对旋转先验
  Eigen::Vector3d t_ref_cur_;           //!< 参考到当前帧的平移先验
  bool have_rotation_prior_ = false;    //!< 是否有旋转先验
  bool have_translation_prior_ = false; //!< 是否有平移先验
  bool have_depth_prior_ = false;       //!< 是否有深度先验
  double depth_at_current_frame_ = 1.0; //!< 当前帧特征点深度先验

  // 构造函数
  AbstractInitialization(
      const InitializationOptions& init_options,
      const FeatureTrackerOptions& tracker_options,
      const DetectorOptions& detector_options,
      const CameraBundlePtr& cams);

  virtual ~AbstractInitialization();  // 虚析构函数

  // 跟踪特征并检查视差
  bool trackFeaturesAndCheckDisparity(const FrameBundlePtr& frames);

  // 纯虚函数：添加帧集合进行初始化
  virtual InitResult addFrameBundle(const FrameBundlePtr& frames_cur) = 0;

  virtual void reset();  // 重置初始化器

  // 设置绝对旋转先验
  inline void setAbsoluteOrientationPrior(const Quaternion& R_cam_world)
  {
    R_cur_world_ = R_cam_world;
    have_rotation_prior_ = true;
  }

  // 设置平移先验
  inline void setTranslationPrior(const Eigen::Vector3d& t_ref_cur) {
    t_ref_cur_ = t_ref_cur;
    have_translation_prior_ = true;
  }

  // 设置深度先验
  inline void setDepthPrior(double depth) {
    depth_at_current_frame_ = depth;
    have_depth_prior_ = true;
  }

};

/// 使用Lucas-Kanade追踪器并估计单应性矩阵的初始化类
class HomographyInit : public AbstractInitialization
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using AbstractInitialization::AbstractInitialization;  // 继承构造函数
  virtual ~HomographyInit() = default;  // 默认析构函数

  // 实现添加帧集合的虚函数
  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};

/// 使用KLT追踪特征并结合IMU旋转先验进行2点RANSAC估计的初始化类
class TwoPointInit : public AbstractInitialization
{
public:
  using AbstractInitialization::AbstractInitialization;
  virtual ~TwoPointInit() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};

/// 使用KLT追踪特征并进行5点RANSAC估计相对位姿的初始化类
class FivePointInit : public AbstractInitialization
{
public:
  using AbstractInitialization::AbstractInitialization;
  virtual ~FivePointInit() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};

/// 假设水平地面并给定深度的初始化类（用于自主起飞）
class OneShotInit : public AbstractInitialization
{
public:
  using AbstractInitialization::AbstractInitialization;
  virtual ~OneShotInit() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};

// 双目初始化类
class StereoInit : public AbstractInitialization
{
public:

  StereoInit(
      const InitializationOptions& init_options,
      const FeatureTrackerOptions& tracker_options,
      const DetectorOptions& detector_options,
      const CameraBundlePtr& cams);

  virtual ~StereoInit() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;

  std::unique_ptr<StereoTriangulation> stereo_;  // 双目三角化模块
  std::shared_ptr<AbstractDetector> detector_;   // 特征检测器
};

// 多相机阵列几何初始化类（17点RANSAC）
class ArrayInitGeometric : public AbstractInitialization
{
public:
  using AbstractInitialization::AbstractInitialization;
  virtual ~ArrayInitGeometric() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};

// 多相机阵列优化初始化类（GTSAM优化）
class ArrayInitOptimization : public AbstractInitialization
{
public:
  using AbstractInitialization::AbstractInitialization;
  virtual ~ArrayInitOptimization() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};


namespace initialization_utils {

// 三角化并初始化点云
bool triangulateAndInitializePoints(
    const FramePtr& frame_cur,
    const FramePtr& frame_ref,
    const Transformation& T_cur_ref,
    const double reprojection_threshold,
    const double depth_at_current_frame,
    const size_t min_inliers_threshold,
    AbstractInitialization::FeatureMatches& matches_cur_ref);

// 三角化特征点
void triangulatePoints(
    const Frame& frame_cur,
    const Frame& frame_ref,
    const Transformation& T_cur_ref,
    const double reprojection_threshold,
    AbstractInitialization::FeatureMatches& matches_cur_ref,
    Positions& points_in_cur);

// 重新缩放并初始化点云
void rescaleAndInitializePoints(
    const FramePtr& frame_cur,
    const FramePtr& frame_ref,
    const AbstractInitialization::FeatureMatches& matches_cur_ref,
    const Positions& points_in_cur,
    const Transformation& T_cur_ref,
    const double depth_at_current_frame);

// 显示特征轨迹
void displayFeatureTracks(
    const FramePtr& frame_cur,
    const FramePtr& frame_ref);

// 创建初始化器工厂函数
AbstractInitialization::UniquePtr makeInitializer(
    const InitializationOptions& init_options,
    const FeatureTrackerOptions& tracker_options,
    const DetectorOptions& detector_options,
    const std::shared_ptr<CameraBundle>& camera_array);

// 复制方向向量
void copyBearingVectors(
    const Frame& frame_cur,
    const Frame& frame_ref,
    const AbstractInitialization::FeatureMatches& matches_cur_ref,
    AbstractInitialization::BearingVectors* f_cur,
    AbstractInitialization::BearingVectors* f_ref);

// 计算方向向量夹角误差
inline double angleError(const Eigen::Vector3d& f1, const Eigen::Vector3d& f2)
{
  return std::acos(f1.dot(f2) / (f1.norm()*f2.norm()));
}

} // namespace initialization_utils

} // namespace svo