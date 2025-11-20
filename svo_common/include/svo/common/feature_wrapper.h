/*
 * @Author: guo g78730158@163.com
 * @Date: 2025-11-13 09:42:09
 * @LastEditors: guo g78730158@163.com
 * @LastEditTime: 2025-11-20 16:44:53
 * @FilePath: /schur_vins_guo/svo_common/include/svo/common/feature_wrapper.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once

#include <memory>
#include <svo/common/types.h>

namespace svo {

// forward declaration
class Point;
using PointPtr = std::shared_ptr<Point>;

class Frame;
using FramePtr = std::shared_ptr<Frame>;

struct SeedRef
{
  FramePtr keyframe;
  int seed_id = -1;
  SeedRef(const FramePtr& _keyframe, const int _seed_id)
    : keyframe(_keyframe)
    , seed_id(_seed_id)
  { ; }
  SeedRef() = default;
  ~SeedRef() = default;
};

/** @todo (MWE)
 */
struct FeatureWrapper
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FeatureType& type;                //!< 特征类型引用，指示特征是角点(corner)还是边缘点(edgelet)
  Eigen::Ref<Keypoint> px;          //!< 特征在金字塔0层上的像素坐标引用
  Eigen::Ref<BearingVector> f;      //!< 特征的单位方向向量引用，从相机光心指向特征的方向
  Eigen::Ref<GradientVector> grad;  //!< 边缘特征的主梯度方向引用，已归一化
  Score& score;                     //!< 特征提取的质量分数引用，用于评估特征的稳定性和显著性
  Level& level;                     //!< 特征提取时所在的图像金字塔层级引用
  PointPtr& landmark;               //!< 与该特征关联的地图点指针引用，nullptr表示尚未三角化
  SeedRef& seed_ref;                //!< 特征的种子点引用，用于深度滤波器中的深度估计
  int& track_id;                    //!< 特征的跟踪ID引用，用于在多帧间识别同一特征点

  FeatureWrapper(
      FeatureType& _type,
      Eigen::Ref<Keypoint> _px,
      Eigen::Ref<BearingVector> _f,
      Eigen::Ref<GradientVector> _grad,
      Score& _score,
      Level& _pyramid_level,
      PointPtr& _landmark,
      SeedRef& _seed_ref,
      int& _track_id)
    : type(_type)
    , px(_px)
    , f(_f)
    , grad(_grad)
    , score(_score)
    , level(_pyramid_level)
    , landmark(_landmark)
    , seed_ref(_seed_ref)
    , track_id(_track_id)
  { ; }

  FeatureWrapper() = delete;
  ~FeatureWrapper() = default;

  //! @todo (MWE) do copy and copy-asignment operators make sense?
  FeatureWrapper(const FeatureWrapper& other) = default;
  FeatureWrapper& operator=(const FeatureWrapper& other) = default;
};

} // namespace svo
