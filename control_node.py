import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.get_logger().info(">>> [DEBUG] Control Node Started.")
        
        self.subscription = self.create_subscription(
            Point, '/human_target_info', self.listener_callback, 10)
            
        self.publisher_ = self.create_publisher(Twist, '/commands/velocity', 10)
        
        # --- TUNING ---
        self.target_distance = 1.5  # 1.5m Safety Zone
        self.k_turn = -0.005        # Turn sensitivity
        self.k_dist = 0.5           # Speed sensitivity
        self.max_speed = 0.3        # Limit speed to 0.3 m/s
        self.dist_tolerance = 0.1   # Dead-zone (meters)

    def listener_callback(self, msg):
        cmd = Twist()
        turn_error = msg.x
        current_distance = msg.y
        is_following = (msg.z == 1.0)

        if is_following:
            # 1. Rotation Logic (Center human in frame)
            cmd.angular.z = self.k_turn * turn_error

            # 2. Distance Logic (Move to reach 1.5m)
            if current_distance > 0:
                distance_error = current_distance - self.target_distance
                
                if abs(distance_error) < self.dist_tolerance:
                    cmd.linear.x = 0.0 # Inside safety zone
                else:
                    speed = self.k_dist * distance_error
                    # Safety clamps
                    if speed > self.max_speed: speed = self.max_speed
                    if speed < -0.1: speed = -0.1 # Limit reverse speed
                    cmd.linear.x = speed
            else:
                cmd.linear.x = 0.0 # Stop if distance is unknown
        else:
            # Stop if PAUSED
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
