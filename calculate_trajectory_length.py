#!/usr/bin/env python3
import os
import sys
import numpy as np

def load_trajectory(traj_path):
    """加载轨迹文件（TUM格式）"""
    print(f"加载轨迹文件: {traj_path}")
    trajectory = []
    try:
        with open(traj_path, 'r') as f:
            for line in f:
                if line.strip() and not line.strip().startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        # TUM格式: timestamp tx ty tz qx qy qz qw
                        timestamp = float(parts[0])
                        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                        trajectory.append([timestamp, tx, ty, tz])
        return np.array(trajectory)
    except Exception as e:
        print(f"加载轨迹文件出错: {e}")
        sys.exit(1)

def calculate_length(trajectory):
    """计算轨迹总长度"""
    if len(trajectory) < 2:
        print("轨迹点太少，无法计算长度")
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(trajectory)):
        # 计算相邻两点之间的欧几里得距离
        prev_pos = trajectory[i-1, 1:4]  # tx, ty, tz
        curr_pos = trajectory[i, 1:4]
        distance = np.linalg.norm(curr_pos - prev_pos)
        total_length += distance
    
    return total_length

def calculate_statistics(trajectory):
    """计算轨迹统计信息"""
    if len(trajectory) < 2:
        return None
    
    # 计算总时间
    start_time = trajectory[0, 0]
    end_time = trajectory[-1, 0]
    total_time = end_time - start_time
    
    # 计算平均速度
    total_length = calculate_length(trajectory)
    avg_speed = total_length / total_time if total_time > 0 else 0
    
    # 计算瞬时速度（相邻点之间）
    velocities = []
    for i in range(1, len(trajectory)):
        time_diff = trajectory[i, 0] - trajectory[i-1, 0]
        if time_diff > 0:
            prev_pos = trajectory[i-1, 1:4]
            curr_pos = trajectory[i, 1:4]
            distance = np.linalg.norm(curr_pos - prev_pos)
            velocity = distance / time_diff
            velocities.append(velocity)
    
    if velocities:
        max_velocity = max(velocities)
        avg_instant_velocity = np.mean(velocities)
    else:
        max_velocity = 0
        avg_instant_velocity = 0
    
    return {
        'total_length': total_length,
        'total_time': total_time,
        'avg_speed': avg_speed,
        'max_velocity': max_velocity,
        'avg_instant_velocity': avg_instant_velocity,
        'num_poses': len(trajectory)
    }

def main():
    if len(sys.argv) < 2:
        print("用法: python calculate_trajectory_length.py <轨迹文件路径>")
        sys.exit(1)
    
    traj_path = sys.argv[1]
    if not os.path.exists(traj_path):
        print(f"文件不存在: {traj_path}")
        sys.exit(1)
    
    # 加载轨迹
    trajectory = load_trajectory(traj_path)
    print(f"加载了 {len(trajectory)} 个轨迹点")
    
    # 计算统计信息
    stats = calculate_statistics(trajectory)
    if stats:
        print("\n轨迹统计信息:")
        print(f"总长度: {stats['total_length']:.4f} 米")
        print(f"总时间: {stats['total_time']:.4f} 秒")
        print(f"平均速度: {stats['avg_speed']:.4f} 米/秒")
        print(f"最大瞬时速度: {stats['max_velocity']:.4f} 米/秒")
        print(f"平均瞬时速度: {stats['avg_instant_velocity']:.4f} 米/秒")
    else:
        print("无法计算轨迹统计信息")

if __name__ == '__main__':
    main()
