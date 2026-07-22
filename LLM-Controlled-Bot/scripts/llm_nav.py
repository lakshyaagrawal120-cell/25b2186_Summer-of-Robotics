#!/usr/bin/env python3
"""
LLM Navigation node — plain-English commands → Nav2 goal.

Pipeline:
  text topic/input → ollama LLM → NavigateToPose action

Usage
─────
  ros2 run diff_drive_robot llm_nav.py

Deps:
  ollama must be running: `ollama serve`
"""

import json
import math
import os
import re
import threading
import time
import urllib.request
import urllib.error

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


# ── helpers — do not modify ───────────────────────────────────────────────────

def _load_locations(share_dir: str) -> dict:
    candidates = [
        os.path.join(share_dir, 'config', 'locations.yaml'),
        os.path.join(os.path.expanduser('~'), 'rosnav', 'locations.yaml'),
    ]
    try:
        import yaml
    except ImportError:
        return {}
    for p in candidates:
        if os.path.isfile(p):
            with open(p) as f:
                data = yaml.safe_load(f) or {}
            return data.get('locations', {})
    return {}


def _yaw_to_quat(yaw_deg: float):
    yaw = math.radians(yaw_deg)
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


# ── TODO 1 — System prompt ────────────────────────────────────────────────────

_SYSTEM = """You are an AI navigation assistant for a robot. Your ONLY purpose is to convert natural language commands into a strictly formatted JSON object. 
DO NOT include any markdown, conversational text, or explanations. OUTPUT ONLY VALID JSON.

Known locations: {locations}

Rules:
- If the user wants to go to a known location, output: {{"action": "go", "location": "<name>"}}
- If the user gives x, y coordinates, output: {{"action": "go", "x": <float>, "y": <float>, "yaw": <float>}}
- If the user says stop/halt, output: {{"action": "stop"}}
- If the command is unrelated or unrecognised, output: {{"action": "unknown", "reason": "<reason>"}}

Examples:
Command: "Go to the kitchen"
{{"action": "go", "location": "kitchen"}}

Command: "Drive to x 2.5 and y 3.0"
{{"action": "go", "x": 2.5, "y": 3.0, "yaw": 0.0}}

Command: "Stop right there"
{{"action": "stop"}}

Command: "{command}"
"""

# ── TODO 2 — Ollama API call ──────────────────────────────────────────────────

def call_ollama(model: str, prompt: str, base_url: str = 'http://localhost:11434') -> str:
    """
    Send a completion request to a locally running Ollama instance and return
    the model's response string.
    """
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",       # Forces strict JSON token generation
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    
    # Timeout removed as requested
    with urllib.request.urlopen(req) as response:
        res_body = json.loads(response.read().decode('utf-8'))
        return res_body.get('response', '')


# ── TODO 3 — JSON extraction ──────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """
    LLMs often wrap their JSON in prose or markdown. This function defensively
    extracts the first {...} block from the raw output string and parses it.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── ROS node ──────────────────────────────────────────────────────────────────

class LLMNavigator(Node):
    def __init__(self):
        super().__init__('llm_navigator')

        self.declare_parameter('ollama_model', 'tinyllama')
        self.declare_parameter('ollama_url',   'http://localhost:11434')
        self.declare_parameter('nav_action',   'navigate_to_pose')
        self.declare_parameter('frame_id',     'map')

        g = self.get_parameter
        self._ollama_model = g('ollama_model').value
        self._ollama_url   = g('ollama_url').value
        self._frame_id     = g('frame_id').value

        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('diff_drive_robot')
        except Exception:
            share = os.path.join(
                os.path.expanduser('~'), 'rosnav', 'src', 'diff_drive_robot-main')
        self._locations = _load_locations(share)
        self.get_logger().info(f'Loaded locations: {list(self._locations.keys())}')

        self._nav_client = ActionClient(self, NavigateToPose, g('nav_action').value)

        self.create_subscription(String, '/llm_nav/command', self._text_cmd_cb, 10)

        self._current_pose: tuple[float, float] | None = None
        self._goal_xy: tuple[float, float] | None = None
        self._nav_start_time: float | None = None
        self._recovery_count = 0
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10)

        self._busy = False
        self._busy_lock = threading.Lock()

        self.get_logger().info(f'LLM nav ready.  ollama={self._ollama_model}')
        self.get_logger().info('Type a command in this terminal or publish to /llm_nav/command')

    # ── LLM parse — do not modify ─────────────────────────────────────────────

    def _parse_command(self, text: str) -> dict | None:
        location_list = ', '.join(self._locations.keys()) if self._locations else 'none'
        prompt = _SYSTEM.format(locations=location_list, command=text)
        try:
            raw = call_ollama(self._ollama_model, prompt, self._ollama_url)
        except (urllib.error.URLError, TimeoutError) as e:
            self.get_logger().error(f'ollama error: {e}')
            return None
        parsed = _extract_json(raw)
        if parsed is None:
            self.get_logger().error(f'LLM returned unparseable: {raw[:200]}')
        return parsed

    # ── TODO 4 — Goal resolution ──────────────────────────────────────────────

    def _resolve_goal(self, parsed: dict) -> tuple[float, float, float] | None:
        action = parsed.get('action')
        
        if action == 'stop':
            self.get_logger().info('Stop command received. Goal execution aborted.')
            return None
            
        elif action == 'unknown':
            reason = parsed.get('reason', 'No reason provided by LLM.')
            self.get_logger().warn(f'LLM could not interpret command: {reason}')
            return None
            
        elif action == 'go':
            if 'location' in parsed:
                loc_name = parsed['location']
                if loc_name in self._locations:
                    loc = self._locations[loc_name]
                    x = float(loc[0])
                    y = float(loc[1])
                    yaw = float(loc[2]) if len(loc) > 2 else 0.0
                    return (x, y, yaw)
                else:
                    self.get_logger().warn(f'Location "{loc_name}" not found in locations.yaml.')
                    return None
                    
            elif 'x' in parsed and 'y' in parsed:
                return (
                    float(parsed['x']), 
                    float(parsed['y']), 
                    float(parsed.get('yaw', 0.0))
                )
        
        self.get_logger().warn(f'LLM generated valid JSON, but missing/invalid "action" key. Got: {parsed}')
        return None

    # ── TODO 5 — Goal dispatch ────────────────────────────────────────────────

    def _send_goal(self, x: float, y: float, yaw_deg: float):
        threading.Thread(
            target=self._send_goal_thread, 
            args=(x, y, yaw_deg), 
            daemon=True
        ).start()

    def _send_goal_thread(self, x: float, y: float, yaw_deg: float):
        if not self._nav_client.wait_for_server(timeout_sec=60.0):
            self.get_logger().error('Nav2 action server not available after 60s!')
            with self._busy_lock:
                self._busy = False
            return

        self._goal_xy = (x, y)
        self._nav_start_time = time.time()
        self._recovery_count = 0

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self._frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)

        # Convert yaw to quaternion (x=0, y=0, z, w)
        z, w = _yaw_to_quat(yaw_deg)
        goal_msg.pose.pose.orientation.z = z
        goal_msg.pose.pose.orientation.w = w

        future = self._nav_client.send_goal_async(goal_msg, feedback_callback=self._feedback_cb)
        future.add_done_callback(self._goal_accepted_cb)

    # ── TODO 6 — Result handling ──────────────────────────────────────────────

    def _result_cb(self, future):
        status = future.result().status
        duration = time.time() - self._nav_start_time

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Nav2 succeeded in {duration:.1f}s with {self._recovery_count} recoveries.')
            
            if self._goal_xy and self._current_pose:
                acc = math.hypot(
                    self._goal_xy[0] - self._current_pose[0], 
                    self._goal_xy[1] - self._current_pose[1]
                )
                self.get_logger().info(f'Navigation accuracy: {acc:.2f}m from target.')
        else:
            self.get_logger().warn(f'Nav2 failed with GoalStatus code: {status}')

        with self._busy_lock:
            self._busy = False


    # ── callbacks — do not modify ─────────────────────────────────────────────

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        self._current_pose = (p.x, p.y)

    def _feedback_cb(self, fb):
        dist = fb.feedback.distance_remaining
        self._recovery_count = fb.feedback.number_of_recoveries
        if dist > 0.0:
            self.get_logger().info(
                f'  distance remaining: {dist:.2f}m', throttle_duration_sec=3.0)

    def _goal_accepted_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Goal rejected by Nav2.')
            with self._busy_lock:
                self._busy = False
            return
        handle.get_result_async().add_done_callback(self._result_cb)

    def _text_cmd_cb(self, msg: String):
        self._process(msg.data.strip())

    def handle_typed(self, text: str):
        with self._busy_lock:
            busy = self._busy
        if busy:
            print('Still navigating — wait or type "stop".', flush=True)
            return
        self._process(text)

    def _process(self, text: str):
        self.get_logger().info(f'Command: "{text}"')
        print(f'   Asking {self._ollama_model}…', flush=True)
        parsed = self._parse_command(text)
        if parsed is None:
            return
        self.get_logger().info(f'LLM parsed: {parsed}')
        goal = self._resolve_goal(parsed)
        if goal:
            with self._busy_lock:
                self._busy = True
            self._send_goal(*goal)


# ── main — do not modify ──────────────────────────────────────────────────────

def _ui_loop(node: LLMNavigator):
    print('\n─────────────────────────────────────────', flush=True)
    print(' LLM Navigator  |  ctrl-C to quit', flush=True)
    print(' Type a command → send as text', flush=True)
    print('─────────────────────────────────────────\n', flush=True)
    while rclpy.ok():
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line:
            node.handle_typed(line)


def main(args=None):
    rclpy.init(args=args)
    node = LLMNavigator()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        _ui_loop(node)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()