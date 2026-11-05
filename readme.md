# Human-Following Robot with Safety Zone & Gesture Control

An autonomous human-following system implemented in **ROS 2 Humble** using a **TurtleBot2 (Kobuki base)** and an **Intel RealSense RGB-D camera**. The robot utilizes **YOLOv8 Pose** estimation for skeleton tracking, depth-based spatial positioning, gesture recognition for target locking/unlocking, real-time depth-map obstacle masking, and exponentially smoothed kinematic velocity control.

---

## 📸 Key Features

- **Pose-Based Gesture Control:**
  - **Raise Right Hand (Above Nose):** Locks target and initiates `FOLLOW` mode.
  - **Raise Left Hand (Above Nose):** Unlocks target and returns to `IDLE` mode.
- **YOLOv8 Target Tracking & Identification:** Maintains focus on a specific locked human target using skeleton keypoint estimation ($p[5], p[6]$ shoulders).
- **Depth Masking & Obstacle Avoidance:** Automatically masks human bounding boxes out of the depth map to inspect surrounding space for obstacles ($\text{depth} < 0.5\text{m}$), triggering evasive maneuver state machines (`AVOID_TURN`, `AVOID_DRIVE`, `SEARCH`).
- **Jiggle-Killer Exponential Smoothing:** Damps abrupt control variations ($\alpha = 0.07$) with linear and angular deadbands to prevent actuator chatter.
- **Safety Zone Maintenance:** Configured with a default stop zone at $1.5\text{m}$.

---

## 🛠 System Architecture

```
                               +-----------------------------+
                               |     RealSense RGB-D Camera   |
                               +--------------+--------------+
                                              |
                   /camera/camera/color/image_raw  |  /camera/camera/aligned_depth_to_color/image_raw
                                              v
                                   +-------------------+
                                   |    Vision Node    |
                                   | (YOLOv8-pose + CV)|
                                   +---------+---------+
                                             |
                                  /human_target_info (geometry_msgs/Point)
                                             | [x: pixel error, y: distance, z: state]
                                             v
                                   +-------------------+
                                   |   Control Node    |
                                   | (Smooth Controller|
                                   |  & State Machine) |
                                   +---------+---------+
                                             |
                                   /commands/velocity (geometry_msgs/Twist)
                                             v
                                  +--------------------+
                                  |    Kobuki Base     |
                                  +--------------------+
```

### ROS 2 Interface Specifications

| Topic Name | Message Type | Publisher | Subscriber | Description |
|---|---|---|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/msg/Image` | RealSense | `vision_node` | RGB camera stream |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | RealSense | `vision_node` | Aligned depth frame (16UC1) |
| `/human_target_info` | `geometry_msgs/msg/Point` | `vision_node` | `control_node` | $x$: Pixel deviation from center<br>$y$: Target distance ($m$)<br>$z$: Target state code |
| `/commands/velocity` | `geometry_msgs/msg/Twist` | `control_node` | `kobuki_node` | Linear ($x$) and angular ($z$) velocity commands |

#### Target State Codes (`msg.z`):
- **`0.0`**: IDLE / Paused
- **`1.0`**: Active Target Following
- **`2.0`**: Obstacle Detected on Left (Trigger Right Evasive Turn)
- **`3.0`**: Obstacle Detected on Right (Trigger Left Evasive Turn)
- **`4.0`**: Critical Obstacle Ahead (Full Stop)
- **`9.0`**: Target Lost / Search Mode

---

## 🚀 Prerequisites & Installation

### Hardware Requirements
- Kobuki TurtleBot 2 Mobile Base
- Intel RealSense RGB-D Camera (or compatible Asus Xtion RGB-D camera)
- Ubuntu 22.04 LTS Machine with ROS 2 Humble

### Dependencies
Install ROS 2 dependencies and Python packages:

```bash
sudo apt update
sudo apt install ros-humble-desktop ros-humble-cv-bridge ros-humble-realsense2-camera
pip install ultralytics opencv-python numpy
```

Ensure the YOLOv8 pose model is available:
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt -O ~/turtlebot2_ws/yolov8n-pose.pt
```

---

## 💻 Build & Execution Instructions

### 1. Build Package
```bash
cd ~/turtlebot2_ws
source /opt/ros/humble/setup.bash
source ~/ros2_humble/install/setup.bash
colcon build --packages-select human_follower --symlink-install
source install/setup.bash
```

### 2. Launch Kobuki Base
Open Terminal 1:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_humble/install/setup.bash
source ~/turtlebot2_ws/install/setup.bash
ros2 launch kobuki_node kobuki_node-launch.py
```

### 3. Launch RealSense Camera
Open Terminal 2:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_humble/install/setup.bash
source ~/turtlebot2_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

### 4. Run Vision Node
Open Terminal 3:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_humble/install/setup.bash
source ~/turtlebot2_ws/install/setup.bash
export PYTHONUNBUFFERED=1
ros2 run human_follower vision_node
```

### 5. Run Control Node
Open Terminal 4:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_humble/install/setup.bash
source ~/turtlebot2_ws/install/setup.bash
ros2 run human_follower control_node
```

---

## ⚙️ Control Tuning Parameters

Parameters in `control_node.py` can be adjusted according to environment requirements:

```python
self.stop_zone = 1.5        # Safe distance to maintain from target (meters)
self.smooth_factor = 0.07   # Exponential smoothing factor (alpha)
self.deadband_lin = 0.01    # Minimum linear speed threshold (m/s)
self.deadband_ang = 0.02    # Minimum angular speed threshold (rad/s)
```
