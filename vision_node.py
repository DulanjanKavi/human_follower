import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info(">>> [DEBUG] Initializing Vision Node with YOLO...")

        # Subscribers (Adjust topic names if 'ros2 topic list' shows different paths)
        self.subscription_rgb = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.rgb_callback, 10)
        
        self.subscription_depth = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        # Publisher to send info to the control node
        self.target_pub = self.create_publisher(Point, '/human_target_info', 10)
        
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n-pose.pt") # Ensure this file is in ~/turtlebot2_ws/
        
        self.follow_mode = False 
        self.current_depth_image = None

    def depth_callback(self, msg):
        try:
            self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth error: {e}")

    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            return

        # Run YOLO Inference
        results = self.model(cv_image, classes=[0], verbose=False)
        annotated_frame = results[0].plot() 
        
        target_msg = Point()
        height, width, _ = cv_image.shape
        center_x_image = width // 2

        # Draw Center Line for alignment
        cv2.line(annotated_frame, (center_x_image, 0), (center_x_image, height), (255, 0, 0), 2)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # Get keypoints for the first detected person
            person = results[0].keypoints.xy.cpu().numpy()[0]
            
            # Keypoint indices: 0=Nose, 9=Left Wrist, 10=Right Wrist
            nose_x, nose_y = person[0]
            l_wrist_y = person[9][1]   
            r_wrist_y = person[10][1]  

            # --- GESTURE LOGIC ---
            # Start: Right Hand Up (y is smaller at the top)
            if r_wrist_y > 0 and r_wrist_y < nose_y:
                if not self.follow_mode:
                    self.follow_mode = True
                    self.get_logger().info("SIGNAL: START")
            
            # Stop: Left Hand Up
            elif l_wrist_y > 0 and l_wrist_y < nose_y:
                if self.follow_mode:
                    self.follow_mode = False
                    self.get_logger().info("SIGNAL: STOP")

            # --- DEPTH CALCULATION ---
            measured_distance = 0.0
            if nose_x > 0 and self.current_depth_image is not None:
                ix, iy = int(nose_x), int(nose_y)
                if 0 <= iy < self.current_depth_image.shape[0] and 0 <= ix < self.current_depth_image.shape[1]:
                    depth_mm = self.current_depth_image[iy, ix]
                    measured_distance = depth_mm / 1000.0 # Meters
                    cv2.putText(annotated_frame, f"Dist: {measured_distance:.2f}m", (ix+10, iy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- PREPARE DATA ---
            target_msg.x = float(nose_x - center_x_image) # Horizontal Error
            target_msg.y = float(measured_distance)      # Distance
            target_msg.z = 1.0 if self.follow_mode else 0.0

        self.target_pub.publish(target_msg)
        
        # HUD View
        status_text = "MODE: FOLLOWING" if self.follow_mode else "MODE: PAUSED"
        color = (0, 255, 0) if self.follow_mode else (0, 0, 255)
        cv2.putText(annotated_frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("YOLO Human Following", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
