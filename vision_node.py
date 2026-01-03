#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
from ultralytics import YOLO

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()
        self.get_logger().info("Loading YOLOv8 model... please wait.")
        self.model = YOLO("/home/dulanjan/turtlebot2_ws/yolov8n-pose.pt")
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription_rgb = self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_callback, qos_profile)
        self.subscription_depth = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile)
        self.target_pub = self.create_publisher(Point, '/human_target_info', 10)
        
        self.follow_mode = False 
        self.locked_person_id = None 
        self.current_depth_image = None
        self.get_logger().info("Vision Node is Ready. Waiting for camera stream...")

    def depth_callback(self, msg):
        try:
            self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except: pass

    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: 
            return

        results = self.model.track(cv_image, persist=True, classes=[0], verbose=False)
        annotated_frame = cv_image.copy()
        h, w, _ = cv_image.shape
        target_msg = Point() 
        target_msg.z = 0.0 # Default: Paused

        if self.current_depth_image is None: 
            cv2.imshow("Vision View", annotated_frame)
            cv2.waitKey(1)
            return
            
        obs_depth = self.current_depth_image.copy()
        best_idx = None

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints_list = results[0].keypoints.xy.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for i, person in enumerate(keypoints_list):
                nose_y, rw_y, lw_y = person[0][1], person[10][1], person[9][1]

                if not self.follow_mode:
                    if 0 < rw_y < nose_y:
                        self.follow_mode = True
                        self.locked_person_id = i
                        self.get_logger().info("Target Locked by Gesture!")
                elif self.follow_mode and i == self.locked_person_id:
                    if 0 < lw_y < nose_y:
                        self.follow_mode = False
                        self.locked_person_id = None
                        self.get_logger().info("Target Released.")
                    best_idx = i
                
                # Mask humans from depth map for obstacle scanning
                bx = boxes[i].astype(int)
                obs_depth[max(0,bx[1]):min(h,bx[3]), max(0,bx[0]):min(w,bx[2])] = 0

            if self.follow_mode and best_idx is not None:
                p = keypoints_list[best_idx]
                cx, cy = int((p[5][0]+p[6][0])/2), int((p[5][1]+p[6][1])/2)
                
                # Distance measurement
                roi = self.current_depth_image[max(0,cy-5):min(h,cy+5), max(0,cx-5):min(w,cx+5)]
                valid = roi[roi > 0]
                h_dist = float(np.median(valid)) / 1000.0 if valid.size > 0 else 0.0

                target_msg.x = float(cx - (w/2))
                target_msg.y = h_dist
                target_msg.z = 1.0 
                
                # RESTORED DISTANCE DISPLAY
                cv2.rectangle(annotated_frame, (int(p[5][0]-50), int(p[0][1]-50)), (int(p[6][0]+50), int(p[16][1])), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(annotated_frame, f"TARGET LOCKED: {h_dist:.2f}m", (20, 90), 0, 0.8, (0, 255, 0), 2)

        # Obstacle Detection
        if self.follow_mode:
            danger_area = obs_depth[int(h*0.5):, :]
            mid = w // 2
            left_valid = danger_area[:, :mid][(danger_area[:, :mid] > 0) & (danger_area[:, :mid] < 500)]
            right_valid = danger_area[:, mid:][(danger_area[:, mid:] > 0) & (danger_area[:, mid:] < 500)]
            
            if best_idx is None:
                target_msg.z = 9.0 # Search mode if human lost
            elif left_valid.size > 200 and right_valid.size > 200:
                target_msg.z = 4.0 
                cv2.putText(annotated_frame, "OBSTACLE: BLOCKED", (50, 150), 2, 1.0, (0,0,255), 3)
            elif left_valid.size > 200:
                target_msg.z = 2.0 
                cv2.putText(annotated_frame, "OBSTACLE: LEFT", (50, 150), 2, 0.8, (0,165,255), 2)
            elif right_valid.size > 200:
                target_msg.z = 3.0 
                cv2.putText(annotated_frame, "OBSTACLE: RIGHT", (50, 150), 2, 0.8, (0,165,255), 2)

        status_txt = "FOLLOWING" if self.follow_mode else "PAUSED"
        color = (0, 255, 0) if self.follow_mode else (0, 0, 255)
        cv2.putText(annotated_frame, status_txt, (20, 50), 1, 2.0, color, 2)
        cv2.imshow("Vision View", annotated_frame)
        cv2.waitKey(1)
        self.target_pub.publish(target_msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VisionNode())
    rclpy.shutdown()