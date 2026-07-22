#!/usr/bin/env python3
"""
obstacle_tracker.py — Moving obstacle detection from consecutive LaserScans.

Compares range values between the current scan and a scan N frames ago.
Rays that have shortened faster than `min_speed` m/s indicate an approaching
object.  Detected points are clustered, transformed to the map frame via TF,
and published as RViz markers and a JSON state topic.

Algorithm
─────────
  1. Keep a rolling buffer of the last `history_len` LaserScan messages.
  2. On each new scan, compare current range[i] with range[i] from
     `lookback` frames ago.
  3. A ray is "closing" if  Δrange / Δtime  <  -min_speed  (range shrinking).
  4. Convert closing ray endpoints from robot frame → map frame via TF.
  5. Cluster nearby points within `cluster_radius` metres (single-linkage).
  6. Publish:
       /obstacle_tracker/markers  — MarkerArray (RViz spheres, red)
       /obstacle_tracker/state    — std_msgs/String JSON per-cluster summary

Parameters
──────────
  robot_ns        namespace prefix      (default: '')
  min_speed       m/s closing threshold (default: 0.08)
  history_len     scan buffer size      (default: 10)
  lookback        frames to compare     (default: 5)
  cluster_radius  grouping distance m   (default: 0.4)
  marker_lifetime seconds to show mark  (default: 0.5)
  base_frame      robot frame           (default: base_link)
  map_frame       world frame           (default: map)

Usage
─────
  ros2 run diff_drive_robot obstacle_tracker.py
  ros2 run diff_drive_robot obstacle_tracker.py --ros-args \
      -p min_speed:=0.05
  ros2 topic echo /obstacle_tracker/state
"""

import json
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class ObstacleTracker(Node):
    def __init__(self):
        super().__init__('obstacle_tracker')

        self.declare_parameter('robot_ns',        '')
        self.declare_parameter('min_speed',        0.08)
        self.declare_parameter('history_len',      10)
        self.declare_parameter('lookback',         5)
        self.declare_parameter('cluster_radius',   0.4)
        self.declare_parameter('marker_lifetime',  0.5)
        self.declare_parameter('base_frame',       'base_link')
        self.declare_parameter('map_frame',        'map')

        ns                 = self.get_parameter('robot_ns').value
        self._min_speed    = self.get_parameter('min_speed').value
        history_len        = self.get_parameter('history_len').value
        self._lookback     = self.get_parameter('lookback').value
        self._cluster_r    = self.get_parameter('cluster_radius').value
        self._marker_life  = self.get_parameter('marker_lifetime').value
        self._base_frame   = self.get_parameter('base_frame').value
        self._map_frame    = self.get_parameter('map_frame').value

        if ns:
            self._base_frame = f'{ns}/{self._base_frame}'

        pre = f'/{ns}' if ns else ''

        self._buf: deque = deque(maxlen=history_len)

        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        self._marker_pub = self.create_publisher(
            MarkerArray, f'{pre}/obstacle_tracker/markers', 10)
        self._state_pub = self.create_publisher(
            String, f'{pre}/obstacle_tracker/state', 10)

        self.create_subscription(LaserScan, f'{pre}/scan', self._scan_cb, 10)

        self.get_logger().info(
            f'ObstacleTracker  ns={ns or "/"}  '
            f'min_speed={self._min_speed} m/s  '
            f'lookback={self._lookback} frames')

    # ── TODO 1 — Closing-ray detection and TF transform ───────────────────────

    def _scan_cb(self, msg: LaserScan):
        # 1. Append msg to self._buf. Return early if fewer than lookback + 1 scans.
        self._buf.append(msg)
        if len(self._buf) < self._lookback + 1:
            return

        # 2. Retrieve the scan from self._lookback frames ago as prev.
        prev = self._buf[-self._lookback - 1]

        # 3. Compute dt from the header timestamps. Return if dt <= 0.
        t_now = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        t_prev = prev.header.stamp.sec + (prev.header.stamp.nanosec * 1e-9)
        dt = t_now - t_prev
        
        if dt <= 0:
            return

        robot_pts = []
        min_len = min(len(msg.ranges), len(prev.ranges))
        
        # 4. For each ray i (up to the shorter of the two scans):
        for i in range(min_len):
            r_now = msg.ranges[i]
            r_prev = prev.ranges[i]

            # Skip rays outside the valid range of either scan.
            if (math.isinf(r_now) or math.isnan(r_now) or not (msg.range_min <= r_now <= msg.range_max)):
                continue
            if (math.isinf(r_prev) or math.isnan(r_prev) or not (prev.range_min <= r_prev <= prev.range_max)):
                continue

            # Compute closing_speed
            closing_speed = (r_prev - r_now) / dt
            
            # If closing_speed > self._min_speed, convert to robot-frame Cartesian
            if closing_speed > self._min_speed:
                angle = msg.angle_min + (i * msg.angle_increment)
                x = r_now * math.cos(angle)
                y = r_now * math.sin(angle)
                robot_pts.append((x, y))

        # 5. If no closing rays, call self._publish([], stamp) and return.
        stamp = msg.header.stamp
        if not robot_pts:
            self._publish([], stamp)
            return

        # 6. Look up the TF from self._map_frame to self._base_frame (timeout 0.1 s).
        try:
            trans_stamped = self._tf_buf.lookup_transform(
                self._map_frame,
                self._base_frame,
                rclpy.time.Time(),
                Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f'TF Error: {e}', throttle_duration_sec=2.0)
            return

        # 7. Apply the 2-D rigid transform
        trans = trans_stamped.transform.translation
        rot = trans_stamped.transform.rotation
        
        # Extract yaw from quaternion
        siny_cosp = 2.0 * (rot.w * rot.z + rot.x * rot.y)
        cosy_cosp = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        map_pts = []
        for rx, ry in robot_pts:
            mx = rx * math.cos(yaw) - ry * math.sin(yaw) + trans.x
            my = rx * math.sin(yaw) + ry * math.cos(yaw) + trans.y
            map_pts.append((mx, my))

        # 8. Pass the map-frame points to self._cluster(), then publish
        clusters = self._cluster(map_pts)
        self._publish(clusters, stamp)


    # ── TODO 2 — Single-linkage clustering ───────────────────────────────────

    def _cluster(self, pts: list[tuple[float, float]]) -> list[dict]:
        if not pts:
            return []

        visited = set()
        clusters = []

        def get_distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        # Flood-fill/BFS approach to find all connected components
        for i, p in enumerate(pts):
            if i in visited:
                continue

            queue = [i]
            visited.add(i)
            cluster_pts = []

            while queue:
                curr_idx = queue.pop(0)
                curr_p = pts[curr_idx]
                cluster_pts.append(curr_p)

                for j, other_p in enumerate(pts):
                    if j not in visited:
                        if get_distance(curr_p, other_p) <= self._cluster_r:
                            visited.add(j)
                            queue.append(j)

            # Calculate the centroid of the cluster
            if cluster_pts:
                centroid_x = sum(pt[0] for pt in cluster_pts) / len(cluster_pts)
                centroid_y = sum(pt[1] for pt in cluster_pts) / len(cluster_pts)
                clusters.append({
                    'x': centroid_x,
                    'y': centroid_y,
                    'count': len(cluster_pts)
                })

        return clusters

    # ── Publish — do not modify ───────────────────────────────────────────────

    def _publish(self, clusters: list[dict], stamp):
        markers = MarkerArray()

        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        del_marker.header.frame_id = self._map_frame
        del_marker.header.stamp    = stamp
        markers.markers.append(del_marker)

        for i, c in enumerate(clusters):
            m = Marker()
            m.header.frame_id = self._map_frame
            m.header.stamp    = stamp
            m.ns              = 'moving_obstacles'
            m.id              = i
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD
            m.pose.position.x = c['x']
            m.pose.position.y = c['y']
            m.pose.position.z = 0.3
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.35
            m.color.r = 1.0
            m.color.g = 0.2
            m.color.b = 0.0
            m.color.a = 0.85
            m.lifetime.sec     = int(self._marker_life)
            m.lifetime.nanosec = int((self._marker_life % 1) * 1e9)
            markers.markers.append(m)

        self._marker_pub.publish(markers)

        state_msg = String()
        state_msg.data = json.dumps({
            'moving_obstacles': [
                {'x': round(c['x'], 2), 'y': round(c['y'], 2), 'points': c['count']}
                for c in clusters
            ]
        })
        self._state_pub.publish(state_msg)

        if clusters:
            self.get_logger().info(
                f'Moving obstacles: {len(clusters)} cluster(s) — '
                + ', '.join(f'({c["x"]:.2f},{c["y"]:.2f})' for c in clusters))


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()