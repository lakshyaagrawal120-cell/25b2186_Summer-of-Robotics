#!/usr/bin/env python3
"""
Custom obstacle-avoidance navigator (no Nav2 required).

All tuning values are ROS 2 parameters — override at launch:
  ros2 run diff_drive_robot navigation.py --ros-args \
      -p goal_x:=3.0 -p goal_y:=2.0 -p base_speed:=0.8
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math
import numpy as np


class ReliableObstacleNavigator(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance_navigator')

        self.declare_parameter('goal_x',             5.0)
        self.declare_parameter('goal_y',             4.0)
        self.declare_parameter('obstacle_threshold', 1.0)
        self.declare_parameter('clearance_required', 2.0)
        self.declare_parameter('move_distance',      2.5)
        self.declare_parameter('scan_angle_deg',     60.0)
        self.declare_parameter('front_angle_deg',    30.0)
        self.declare_parameter('base_speed',         1.5)
        self.declare_parameter('turn_speed',         3.5)
        self.declare_parameter('goal_tolerance',     0.3)
        self.declare_parameter('timer_period',       0.05)
        self.declare_parameter('cmd_vel_topic',  '/cmd_vel')
        self.declare_parameter('scan_topic',     '/scan')
        self.declare_parameter('odom_topic',     '/odom')

        self.goal = [
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
        ]
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        self.clearance_required = self.get_parameter('clearance_required').value
        self.move_distance      = self.get_parameter('move_distance').value
        self.scan_angle         = math.radians(self.get_parameter('scan_angle_deg').value)
        self.front_angle_range  = math.radians(self.get_parameter('front_angle_deg').value)
        self.base_speed         = self.get_parameter('base_speed').value
        self.turn_speed         = self.get_parameter('turn_speed').value
        self.goal_tolerance     = self.get_parameter('goal_tolerance').value

        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        scan_topic    = self.get_parameter('scan_topic').value
        odom_topic    = self.get_parameter('odom_topic').value

        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10)

        self.state      = 'GOAL_SEEK'
        self.robot_pos  = [0.0, 0.0, 0.0]   # x, y, yaw
        self.start_pos  = [0.0, 0.0]
        self.target_yaw = 0.0
        self.laser_ranges: list = []
        self.laser_angles: list = []

        timer_period = self.get_parameter('timer_period').value
        self.create_timer(timer_period, self.navigate)

        self.get_logger().info(
            f'Navigator ready. Goal: ({self.goal[0]}, {self.goal[1]})')

    # ------------------------------------------------------------------
    # Callbacks — do not modify
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_pos[2] = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y ** 2 + q.z ** 2))

    def scan_callback(self, msg):
        self.laser_ranges = msg.ranges
        if len(self.laser_angles) != len(msg.ranges):
            self.laser_angles = [
                msg.angle_min + i * msg.angle_increment
                for i in range(len(msg.ranges))]

    def distance_moved(self):
        return math.hypot(
            self.robot_pos[0] - self.start_pos[0],
            self.robot_pos[1] - self.start_pos[1])

    # ------------------------------------------------------------------
    # TODO 1 — Front obstacle distance
    # ------------------------------------------------------------------
    def get_front_obstacle_distance(self):
        if not self.laser_ranges:
            return float('inf')
        
        min_dist = float('inf')
        half_angle = self.front_angle_range / 2.0
        
        for r, angle in zip(self.laser_ranges, self.laser_angles):
            # Normalize scan angles to [-pi, pi]
            norm_angle = (angle + math.pi) % (2 * math.pi) - math.pi
            # Check if the ray is within the front wedge
            if -half_angle <= norm_angle <= half_angle:
                if 0.1 < r < min_dist and not math.isinf(r) and not math.isnan(r):
                    min_dist = r
                    
        return min_dist

    # ------------------------------------------------------------------
    # TODO 2 — Clear direction search
    # ------------------------------------------------------------------
    def find_clear_direction(self):
        if not self.laser_ranges:
            goal_yaw = math.atan2(self.goal[1] - self.robot_pos[1], self.goal[0] - self.robot_pos[0])
            return False, goal_yaw

        best_heading = 0.0
        max_clearance = -1.0
        half_scan = self.scan_angle / 2.0

        # Scan candidate headings from -90 to +90 degrees relative to robot (10-degree increments)
        for candidate_deg in range(-90, 91, 10):
            candidate_rad = math.radians(candidate_deg)
            min_r_in_sector = float('inf')

            for r, angle in zip(self.laser_ranges, self.laser_angles):
                norm_angle = (angle + math.pi) % (2 * math.pi) - math.pi
                diff = (norm_angle - candidate_rad + math.pi) % (2 * math.pi) - math.pi
                
                # If ray falls into the candidate sector, evaluate clearance
                if -half_scan <= diff <= half_scan:
                    if 0.1 < r < float('inf') and not math.isnan(r):
                        if r < min_r_in_sector:
                            min_r_in_sector = r

            # Update best heading if this sector is clearer than previous ones
            if min_r_in_sector > max_clearance:
                max_clearance = min_r_in_sector
                best_heading = candidate_rad

        goal_yaw = math.atan2(self.goal[1] - self.robot_pos[1], self.goal[0] - self.robot_pos[0])
        
        if max_clearance > self.clearance_required:
            absolute_yaw = (self.robot_pos[2] + best_heading + math.pi) % (2 * math.pi) - math.pi
            return True, absolute_yaw
        else:
            return False, goal_yaw

    # ------------------------------------------------------------------
    # TODO 3 — Navigation FSM
    # ------------------------------------------------------------------
    def navigate(self):
        if not self.laser_ranges:
            return

        cmd = Twist()
        
        def normalize_angle(angle):
            return (angle + math.pi) % (2 * math.pi) - math.pi
            
        goal_yaw = math.atan2(self.goal[1] - self.robot_pos[1], self.goal[0] - self.robot_pos[0])
        dist_to_goal = math.hypot(self.goal[0] - self.robot_pos[0], self.goal[1] - self.robot_pos[1])
        
        if self.state == 'GOAL_SEEK':
            if dist_to_goal < self.goal_tolerance:
                self.cmd_vel_pub.publish(Twist())
                self.get_logger().info('Goal reached!')
                return # Stop processing once goal is met
                
            if self.get_front_obstacle_distance() < self.obstacle_threshold:
                self.state = 'FIND_CLEAR'
                _, self.target_yaw = self.find_clear_direction()
            else:
                err_yaw = normalize_angle(goal_yaw - self.robot_pos[2])
                cmd.linear.x = max(0.0, self.base_speed * math.cos(err_yaw)) # Prevent backwards driving
                cmd.angular.z = max(-self.turn_speed, min(self.turn_speed, 2.0 * err_yaw))
                
        elif self.state == 'FIND_CLEAR':
            err_yaw = normalize_angle(self.target_yaw - self.robot_pos[2])
            if abs(err_yaw) < math.radians(5):
                self.state = 'MOVE_CLEAR'
                self.start_pos = [self.robot_pos[0], self.robot_pos[1]]
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = self.turn_speed if err_yaw > 0 else -self.turn_speed
                
        elif self.state == 'MOVE_CLEAR':
            if self.get_front_obstacle_distance() < self.obstacle_threshold:
                self.state = 'FIND_CLEAR'
                _, self.target_yaw = self.find_clear_direction()
            elif self.distance_moved() >= self.move_distance:
                self.state = 'REALIGN'
            else:
                err_yaw = normalize_angle(self.target_yaw - self.robot_pos[2])
                cmd.linear.x = self.base_speed
                cmd.angular.z = max(-self.turn_speed, min(self.turn_speed, 2.0 * err_yaw))
                
        elif self.state == 'REALIGN':
            err_yaw = normalize_angle(goal_yaw - self.robot_pos[2])
            if abs(err_yaw) < math.radians(5):
                self.state = 'GOAL_SEEK'
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = self.turn_speed if err_yaw > 0 else -self.turn_speed

        # Guard against NaN before publishing
        if math.isnan(cmd.linear.x): cmd.linear.x = 0.0
        if math.isnan(cmd.angular.z): cmd.angular.z = 0.0
        
        # Final clamp of all twist values
        cmd.linear.x = max(-self.base_speed, min(self.base_speed, cmd.linear.x))
        cmd.angular.z = max(-self.turn_speed, min(self.turn_speed, cmd.angular.z))
        
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ReliableObstacleNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()