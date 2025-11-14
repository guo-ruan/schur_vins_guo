// Modification Note:
// This file may have been modified by the authors of SchurVINS.
// (All authors of SchurVINS are with PICO department of ByteDance Corporation)
#include "svo_ros/svo_node_base.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ros/ros.h>
#include <svo/common/logging.h>
#include <vikit/params_helper.h>

namespace svo_ros
{

    void SvoNodeBase::initThirdParty(int argc, char **argv)
    {
        google::InitGoogleLogging(argv[0]);
        google::ParseCommandLineFlags(&argc, &argv, true);
        google::InstallFailureSignalHandler();
        google::SetStderrLogging(google::GLOG_WARNING);
        // google::SetLogDestination(google::GLOG_INFO, "logs/");

        ros::init(argc, argv, "svo");
    }

    /**
     * @brief SvoNodeBase构造函数
     * @details 初始化ROS节点和SVO接口，设置系统工作模式并订阅必要的数据
     */
    SvoNodeBase::SvoNodeBase()
        // 初始化ROS节点句柄，node_handle_为公共命名空间，private_node_handle_为私有命名空间("~")
        : node_handle_(), private_node_handle_("~"),
          // 根据ROS参数服务器中的"pipeline_is_stereo"参数决定使用单目还是双目模式
          type_(vk::param<bool>(private_node_handle_, "pipeline_is_stereo", false) ? svo::PipelineType::kStereo : svo::PipelineType::kMono),
          // 初始化SVO接口，传入工作模式、公共节点句柄和私有节点句柄
          svo_interface_(type_, node_handle_, private_node_handle_)
    {
        // 检查SVO接口是否配置了IMU处理器
        if (svo_interface_.imu_handler_)
        {
            // 如果配置了IMU处理器，则订阅IMU数据
            svo_interface_.subscribeImu();
        }
        // 订阅图像数据，根据工作模式(单目/双目)会自动选择对应的回调函数
        svo_interface_.subscribeImage();
        // 订阅远程键盘输入，用于系统控制(如重置、暂停等)
        svo_interface_.subscribeRemoteKey();
    }

    void SvoNodeBase::run()
    {
        ros::spin();
        SVO_INFO_STREAM("SVO quit");
        svo_interface_.quit_ = true;
        SVO_INFO_STREAM("SVO terminated.\n");
    }

} // namespace svo_ros
