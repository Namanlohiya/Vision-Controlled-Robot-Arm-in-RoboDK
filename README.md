Vision-Controlled Robot Arm in RoboDK 🤖📸

Welcome to the Vision-Controlled Robot Arm project! This innovative system allows you to control a simulated robot arm in RoboDK using hand gestures captured via a webcam. By leveraging computer vision and robotics APIs, this project creates an intuitive, real-time interface for commanding a robot to move between predefined targets based on finger gestures. 🚀

🎯 Project Overview
This project integrates computer vision with robotics simulation to control a robot arm in RoboDK. Using a webcam, the system detects hand gestures (specific finger combinations) and translates them into commands to move the robot to designated targets (Target 1, Target 2, or Target 3). The project showcases the power of combining OpenCV, MediaPipe, and the RoboDK API to create a futuristic, gesture-based control system.
Key Features

Gesture-Based Control: Move the robot by raising specific fingers:
Index finger up ✌️: Moves to Target 1.
Index + middle fingers up ✌️✌️: Moves to Target 2.
Index + middle + ring fingers up ✌️✌️✌️: Moves to Target 3.


Real-Time Feedback: Displays FPS, finger states, and target information on the video feed for easy debugging.
Robust Error Handling: Ensures reliable connection to RoboDK with checks for timeouts and invalid robot items.
Seamless Integration: Combines computer vision with robotics for a smooth, interactive experience.

🛠️ Technologies Used
The project is built with Python and relies on the following libraries:

RoboDK API (Robolink): Connects to the RoboDK simulation environment to control the robot and manage targets.
OpenCV (cv2): Handles webcam video capture and image processing.
MediaPipe: Provides accurate hand landmark detection for gesture recognition.
Math: Calculates distances between hand landmarks for gesture logic.
Time: Tracks frame rate (FPS) for performance monitoring.

📂 Project Structure
The project consists of two main Python scripts:

Prog4.py: A simple script to initialize and verify the RoboDK API connection.
Prog2_vision.py: The core script that integrates computer vision with robot control, handling gesture detection and robot movements.

File Details

Prog4.py:
Initializes the RoboDK API using Robolink.
Verifies the connection to RoboDK with error handling for failed connections.


Prog2_vision.py:
Connects to RoboDK and retrieves the robot and target items (Target 1, Target 2, Target 3).
Uses OpenCV to capture webcam video and MediaPipe to detect hand landmarks.
Implements gesture recognition logic to identify raised fingers and map them to robot movements.
Displays real-time feedback (FPS, finger states, and target names) on the video feed.
Includes a loop to continuously process video frames and control the robot until the user exits (by pressing the ESC key).



🚀 How It Works

RoboDK Setup:

The script establishes a connection to RoboDK using the Robolink class.
It retrieves the robot and target items (Target 1, Target 2, Target 3) from the RoboDK station.
Error handling ensures a valid robot and connection, with clear error messages for debugging.


Gesture Detection:

The webcam captures video frames, which are processed using OpenCV.
MediaPipe’s hand tracking model identifies hand landmarks (e.g., fingertips, palm points).
Custom functions (findPos, lmDistance, fingersState) calculate distances between landmarks to determine which fingers are raised.
Specific finger combinations (e.g., [0,1,0,0,0] for index finger up) trigger corresponding robot movements.


Robot Control:

Based on the detected gesture, the robot moves to the appropriate target using the MoveJ command.
After each movement, the script triggers a Program_Done instruction in RoboDK to signal completion.
The video feed overlays real-time information, including FPS, finger states, and the active target.


User Interaction:

The system runs in a loop, continuously processing video frames and updating the robot’s position.
Press the ESC key (key code 27) to exit the program and release the webcam.



🛠️ Setup Instructions
To run this project, follow these steps:
Prerequisites

RoboDK: Install RoboDK and ensure it’s running with a station containing a robot and three targets (named "Target 1", "Target 2", "Target 3").
Python: Install Python 3.x.
Webcam: Ensure a webcam is connected and accessible.
Required Libraries:pip install robodk opencv-python mediapipe



Steps

Clone or Download the Project:
Save Prog4.py and Prog2_vision.py in your project directory.


Set Up RoboDK:
Open RoboDK and create/load a station with a robot and three targets.
Ensure the target names match exactly: "Target 1", "Target 2", "Target 3".


Run the Script:
Execute Prog2_vision.py:python Prog2_vision.py


The webcam will activate, and the video feed will display with real-time overlays.


Control the Robot:
Raise your index finger to move to Target 1, index + middle for Target 2, or index + middle + ring for Target 3.
Monitor the video feed for FPS and gesture feedback.
Press ESC to stop the program.



⚠️ Challenges and Solutions

Challenge: Inconsistent gesture detection in varying lighting conditions.
Solution: Fine-tuned distance thresholds in the lmDistance and fingersState functions to improve robustness.


Challenge: Ensuring reliable RoboDK API connection.
Solution: Added error handling for connection failures and invalid robot items using try-except blocks.


Challenge: Balancing real-time performance with accurate gesture recognition.
Solution: Optimized frame processing and used MediaPipe’s efficient hand-tracking model to maintain smooth performance.



🌟 Future Improvements

Enhanced Gestures: Add more complex gestures (e.g., swipe motions) for additional robot actions.
Multi-Robot Control: Extend the system to control multiple robots simultaneously.
UI Enhancements: Create a graphical interface for configuring targets and gestures.
Robustness: Improve gesture detection for diverse lighting conditions and hand orientations.
