#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
import numpy as np

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.publisher_ = self.create_publisher(Twist, '/commands/velocity', 10)
        self.subscription = self.create_subscription(Point, '/human_target_info', self.listener_callback, 10)
        
        # --- SMOOTHING CONFIG ---
        self.smooth_factor = 0.08  # Lower = Smoother (Fixed the jiggles)
        self.curr_v = 0.0
        self.curr_w = 0.0
        
        # --- STATE MACHINE ---
        self.mode = "IDLE" 
        self.step_counter = 0
        self.avoid_dir = 1.0 
        self.stop_zone = 1.5

    def listener_callback(self, msg):
        target_v, target_w = 0.0, 0.0
        state = msg.z 
        dist = msg.y

        # Mode Transitions
        if state == 0.0:
            self.mode = "IDLE"
        elif 2.0 <= state <= 4.0 and self.mode == "FOLLOW":
            self.mode = "AVOID_TURN"
            self.step_counter = 35 # Blind turn time
            self.avoid_dir = -1.0 if state == 2.0 else 1.0 
        elif state == 1.0 and self.mode == "IDLE":
            self.mode = "FOLLOW"

        # Behavior Execution
        if self.mode == "AVOID_TURN":
            target_w = 0.7 * self.avoid_dir
            self.step_counter -= 1
            if self.step_counter <= 0:
                self.mode = "AVOID_DRIVE"
                self.step_counter = 55 # Blind drive time (~0.5m)
        elif self.mode == "AVOID_DRIVE":
            target_v = 0.15
            self.step_counter -= 1
            if self.step_counter <= 0:
                self.mode = "SEARCH"
        elif self.mode == "SEARCH" or (state == 9.0 and self.mode == "FOLLOW"):
            target_w = 0.5 * (-self.avoid_dir) # Turn back to find human
            if state == 1.0: self.mode = "FOLLOW"
        elif self.mode == "FOLLOW":
            if abs(msg.x) > 40: # Deadzone to stop jitter
                target_w = -0.0035 * msg.x
            if dist > self.stop_zone:
                target_v = min(0.3 * (dist - self.stop_zone), 0.4)

        # Apply Smoothing (The Jiggle-Killer)
        self.curr_v += (target_v - self.curr_v) * self.smooth_factor
        self.curr_w += (target_w - self.curr_w) * self.smooth_factor

        # Final publish
        cmd = Twist()
        cmd.linear.x = float(self.curr_v) if self.mode != "IDLE" else 0.0
        cmd.angular.z = float(self.curr_w) if self.mode != "IDLE" else 0.0
        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ControlNode())
    rclpy.shutdown()