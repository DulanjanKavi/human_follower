#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
import numpy as np

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.get_logger().info(">>> FINAL SMOOTH CONTROLLER INITIALIZED.")
        
        self.publisher_ = self.create_publisher(Twist, '/commands/velocity', 10)
        self.subscription = self.create_subscription(Point, '/human_target_info', self.listener_callback, 10)
        
        # --- TUNING PARAMETERS ---
        self.stop_zone = 1.5   
        self.smooth_factor = 0.07  # Very smooth damping
        self.deadband_lin = 0.01   # Ignore tiny tremors
        self.deadband_ang = 0.02
        
        # --- MANEUVER STATE ---
        self.mode = "IDLE" 
        self.step_counter = 0
        self.avoid_dir = 1.0 
        
        self.curr_lin_x = 0.0
        self.curr_ang_z = 0.0

    def listener_callback(self, msg):
        t_lin, t_ang = 0.0, 0.0
        state = msg.z 

        # 1. STATE LOGIC
        if state == 0.0:
            self.mode = "IDLE"
        elif state >= 2.0 and state <= 4.0 and self.mode == "FOLLOW":
            # Start the 3-phase Blind Maneuver
            self.mode = "AVOID_TURN"
            self.step_counter = 35 
            self.avoid_dir = -1.0 if state == 2.0 else 1.0 
        elif state == 1.0 and self.mode == "IDLE":
            self.mode = "FOLLOW"

        # 2. EXECUTION MACHINE
        if self.mode == "IDLE":
            t_lin, t_ang = 0.0, 0.0

        elif self.mode == "AVOID_TURN":
            t_lin, t_ang = 0.0, (0.7 * self.avoid_dir)
            self.step_counter -= 1
            if self.step_counter <= 0:
                self.mode = "AVOID_DRIVE"
                self.step_counter = 60 # Drive 0.5m

        elif self.mode == "AVOID_DRIVE":
            t_lin, t_ang = 0.15, 0.0
            self.step_counter -= 1
            if self.step_counter <= 0:
                self.mode = "SEARCH"

        elif self.mode == "SEARCH" or state == 9.0:
            t_lin, t_ang = 0.0, (0.5 * -self.avoid_dir)
            if state == 1.0: self.mode = "FOLLOW"

        elif self.mode == "FOLLOW":
            # Human Follow Logic
            if abs(msg.x) > 40: # Ignore center-jitter below 40 pixels
                t_ang = -0.0035 * msg.x
            
            if msg.y > self.stop_zone:
                t_lin = min(0.25 * (msg.y - self.stop_zone), 0.4)
            else:
                t_lin = 0.0

        # 3. THE "JIGGLE-KILLER" (Exponential Smoothing)
        new_lin = self.curr_lin_x + (t_lin - self.curr_lin_x) * self.smooth_factor
        new_ang = self.curr_ang_z + (t_ang - self.curr_ang_z) * self.smooth_factor

        # Deadband Check: Only send if change is significant
        if abs(new_lin) < self.deadband_lin: new_lin = 0.0
        if abs(new_ang) < self.deadband_ang: new_ang = 0.0

        self.curr_lin_x, self.curr_ang_z = new_lin, new_ang

        cmd = Twist()
        cmd.linear.x = float(self.curr_lin_x)
        cmd.angular.z = float(self.curr_ang_z)
        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()