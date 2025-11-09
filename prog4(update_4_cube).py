# Type help("robodk.robolink") or help("robodk.robomath") for more information
# Press F5 to run the script
# Documentation: https://robodk.com/doc/en/RoboDK-API.html
# Reference:     https://robodk.com/doc/en/PythonAPI/robodk.html

from robolink import Robolink, ITEM_TYPE_ROBOT, ITEM_TYPE_TOOL, ITEM_TYPE_FRAME, ITEM_TYPE_OBJECT
from robodk import *
import time
import threading
import cv2 as cv
import mediapipe as mp
import math

# ===== ROBODK SETUP =====
RDK = Robolink()

if not RDK.Connect():
    raise ConnectionError("Failed to connect to RoboDK")
print("Successfully connected to RoboDK!")

def get_item(name, item_type=None, required=True):
    item = RDK.Item(name, item_type) if item_type else RDK.Item(name)
    if required and not item.Valid():
        raise ValueError(f"Required item not found: {name}")
    return item

def find_tool(robot):
    common_tool_names = ['Vacuum Gripper', 'Gripper', 'Tool', 'End Effector']
    for tool_name in common_tool_names:
        tool = RDK.Item(tool_name, ITEM_TYPE_TOOL)
        if tool.Valid():
            print(f"Found tool: {tool_name}")
            return tool
    
    tool = robot.getLink(ITEM_TYPE_TOOL)
    if tool.Valid():
        print(f"Found tool via robot link: {tool.Name()}")
        return tool
    
    raise ValueError("No tool found attached to the robot")

try:
    robot = get_item('', ITEM_TYPE_ROBOT)
    print(f"Robot found: {robot.Name()}")
    tool = find_tool(robot)
    frame = get_item('Frame 2', ITEM_TYPE_FRAME)
    home = get_item('Home')
    print("All essential items found successfully")
except ValueError as e:
    print(f"Configuration error: {e}")
    exit()

robot.setFrame(frame)
robot.setTool(tool)

# Get cubes
cubes = []
original_cube_poses = []
cube_names_to_try = ['Cube 50mm', 'Cube 50mm 2', 'Cube 50mm 3', 'Cube 50mm 4', 'Cube', 'Cube 2', 'Cube 3', 'Cube 4']

for name in cube_names_to_try:
    cube = RDK.Item(name, ITEM_TYPE_OBJECT)
    if cube.Valid() and cube not in cubes:
        cubes.append(cube)
        original_cube_poses.append(cube.Pose())
        print(f"Found cube: {cube.Name()}")

if not cubes:
    raise ValueError("No cubes found in station")

print(f"Found {len(cubes)} cubes for operation")

# Find existing targets
def find_targets_for_cubes(num_cubes):
    approach_picks, pick_targets, approach_places, place_targets = [], [], [], []
    
    for i in range(num_cubes):
        cube_num = i + 1
        
        # Find targets
        approach_pick = RDK.Item('Approach Pick' if cube_num == 1 else f'Approach Pick {cube_num}')
        pick_target = RDK.Item('Pick' if cube_num == 1 else f'Pick {cube_num}')
        approach_place = RDK.Item('Approach Place' if cube_num == 1 else f'Approach Place {cube_num}')
        place_target = RDK.Item('Place' if cube_num == 1 else f'Place {cube_num}')
        
        approach_picks.append(approach_pick)
        pick_targets.append(pick_target)
        approach_places.append(approach_place)
        place_targets.append(place_target)
        
        print(f"Cube {cube_num} - Pick: {pick_target.Name() if pick_target.Valid() else 'MISSING'}")
    
    return approach_picks, pick_targets, approach_places, place_targets

approach_picks, pick_targets, approach_places, place_targets = find_targets_for_cubes(len(cubes))

# ===== RELIABLE ROBOT OPERATIONS =====
current_operation = None
operation_lock = threading.Lock()

def safe_robot_move(move_func, target, operation_name):
    if not target or not target.Valid():
        print(f"Target invalid for {operation_name}")
        return False
    try:
        print(f"Moving to {operation_name}...")
        move_func(target)
        time.sleep(0.5)  # Small pause for stability
        return True
    except Exception as e:
        print(f"Movement error in {operation_name}: {e}")
        return False

def pick_sequence(cube_idx):
    global current_operation
    if not (0 <= cube_idx < len(cubes)):
        print(f"Invalid cube index: {cube_idx}")
        return
        
    with operation_lock:
        if current_operation:
            print(f"Operation {current_operation} in progress, skipping pick")
            return
        current_operation = f"pick_{cube_idx}"
    
    try:
        print(f"=== STARTING PICK SEQUENCE FOR CUBE {cube_idx + 1} ===")
        
        # Step 1: Move to approach pick
        if approach_picks[cube_idx] and approach_picks[cube_idx].Valid():
            if not safe_robot_move(robot.MoveL, approach_picks[cube_idx], "approach pick"):
                return
        
        # Step 2: Move to pick position
        if pick_targets[cube_idx] and pick_targets[cube_idx].Valid():
            if not safe_robot_move(robot.MoveL, pick_targets[cube_idx], "pick position"):
                return
            
            # Step 3: Attach cube
            print("Attaching cube...")
            attached = tool.AttachClosest()
            if attached:
                print("Cube attached successfully")
            else:
                print("Warning: No cube attached")
                # Continue anyway
            
            time.sleep(0.5)  # Wait for attachment
        
        # Step 4: Return to approach
        if approach_picks[cube_idx] and approach_picks[cube_idx].Valid():
            safe_robot_move(robot.MoveL, approach_picks[cube_idx], "return from pick")
        
        robot.RunInstruction('Program_Done')
        print(f"=== PICK SEQUENCE COMPLETED FOR CUBE {cube_idx + 1} ===\n")
        
    except Exception as e:
        print(f"Pick sequence error: {e}")
    finally:
        with operation_lock:
            current_operation = None

def place_sequence(cube_idx):
    global current_operation
    if not (0 <= cube_idx < len(cubes)):
        print(f"Invalid cube index: {cube_idx}")
        return
        
    with operation_lock:
        if current_operation:
            print(f"Operation {current_operation} in progress, skipping place")
            return
        current_operation = f"place_{cube_idx}"
    
    try:
        print(f"=== STARTING PLACE SEQUENCE FOR CUBE {cube_idx + 1} ===")
        
        # Step 1: Move to approach place
        if approach_places[cube_idx] and approach_places[cube_idx].Valid():
            if not safe_robot_move(robot.MoveJ, approach_places[cube_idx], "approach place"):
                return
        
        # Step 2: Move to place position
        if place_targets[cube_idx] and place_targets[cube_idx].Valid():
            if not safe_robot_move(robot.MoveL, place_targets[cube_idx], "place position"):
                return
            
            # Step 3: Detach cube
            print("Detaching cube...")
            tool.DetachAll()
            print("Cube detached successfully")
            time.sleep(0.5)  # Wait for detachment
        
        # Step 4: Return to approach
        if approach_places[cube_idx] and approach_places[cube_idx].Valid():
            safe_robot_move(robot.MoveL, approach_places[cube_idx], "return from place")
        
        robot.RunInstruction('Program_Done')
        print(f"=== PLACE SEQUENCE COMPLETED FOR CUBE {cube_idx + 1} ===\n")
        
    except Exception as e:
        print(f"Place sequence error: {e}")
    finally:
        with operation_lock:
            current_operation = None

def go_home():
    global current_operation
    with operation_lock:
        if current_operation:
            return
        current_operation = "home"
    try:
        print("Moving to home position...")
        safe_robot_move(robot.MoveJ, home, "home position")
        robot.RunInstruction('Program_Done')
        print("Home position reached")
    finally:
        with operation_lock:
            current_operation = None

def reset_all_objects():
    global current_operation
    with operation_lock:
        if current_operation:
            return
        current_operation = "reset"
    try:
        print("Resetting all objects...")
        safe_robot_move(robot.MoveJ, home, "home for reset")
        tool.DetachAll()
        for i, cube in enumerate(cubes):
            cube.setPose(original_cube_poses[i])
            print(f"Reset {cube.Name()}")
        robot.RunInstruction('Program_Done')
        print("All objects reset successfully")
    finally:
        with operation_lock:
            current_operation = None

def run_async(operation, *args):
    if threading.active_count() < 5:
        thread = threading.Thread(target=operation, args=args, daemon=True)
        thread.start()
        return True
    return False

# Initialize
go_home()
reset_all_objects()
print("Robot ready for gesture control!")

# ===== IMPROVED GESTURE RECOGNITION =====
class ImprovedGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1  # Better accuracy
        )
        
        self.prev_state = None
        self.stable_count = 0
        self.cooldown = 0
        self.STABLE_THRESHOLD = 3
        
        # Gesture mapping
        self.GESTURE_MAP = {
            (0, 1, 0, 0, 0): (0, 'pick'),    # Index finger only
            (0, 1, 1, 0, 0): (0, 'place'),   # Index + Middle
            (1, 0, 0, 0, 0): (1, 'pick'),    # Thumb only
            (1, 1, 0, 0, 0): (1, 'place'),   # Thumb + Index
            (0, 0, 1, 0, 0): (2, 'pick'),    # Middle only
            (0, 1, 1, 1, 0): (2, 'place'),   # Index + Middle + Ring
            (0, 0, 0, 0, 1): (3, 'pick'),    # Pinky only
            (0, 1, 1, 1, 1): (3, 'place'),   # All except thumb
            (1, 1, 1, 1, 1): ('home', 'home'), # All fingers
            (0, 0, 0, 0, 0): ('reset', 'reset') # Fist
        }

    def detect_fingers_improved(self, landmarks):
        """Improved finger detection with better thumb recognition"""
        if not landmarks:
            return [0, 0, 0, 0, 0], 0
            
        fingers = [0, 0, 0, 0, 0]
        
        # IMPROVED THUMB DETECTION
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        # Method 1: Check if thumb is extended to the side
        # For right hand: thumb tip x should be less than thumb IP x when extended
        thumb_extended_horizontal = thumb_tip.x < thumb_ip.x - 0.05
        
        # Method 2: Check distance from wrist
        thumb_to_wrist_distance = abs(thumb_tip.x - wrist.x)
        if thumb_to_wrist_distance > 0.15:  # Thumb is far from wrist
            thumb_extended_horizontal = True
        
        # Method 3: Check if thumb is not tucked in
        thumb_tucked = thumb_tip.y > thumb_mcp.y + 0.1  # Thumb tip below MCP
        
        fingers[0] = 1 if (thumb_extended_horizontal and not thumb_tucked) else 0
        
        # IMPROVED FINGER DETECTION
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_pips = [6, 10, 14, 18]  # PIP joints
        
        for i, (tip_idx, pip_idx) in enumerate(zip(finger_tips, finger_pips)):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            
            # Finger is extended if tip is significantly above PIP joint
            if tip.y < pip.y - 0.05:  # Increased threshold for better accuracy
                fingers[i + 1] = 1
        
        return fingers, sum(fingers)

    def process_frame(self, frame):
        """Process frame with better landmark visualization"""
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            return None, "Show hand in frame"
        
        # Draw hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        landmarks = hand_landmarks.landmark
        fingers, count = self.detect_fingers_improved(landmarks)
        finger_state = tuple(fingers)
        
        # Draw finger states on image
        self.draw_finger_states(frame, fingers)
        
        return finger_state, f"Fingers: {count} - State: {list(finger_state)}"

    def draw_finger_states(self, frame, fingers):
        """Draw finger states on frame for visual feedback"""
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        colors = [(0, 255, 0) if f else (0, 0, 255) for f in fingers]
        
        for i, (name, color) in enumerate(zip(finger_names, colors)):
            y_pos = 120 + i * 25
            status = "UP" if fingers[i] else "DOWN"
            cv.putText(frame, f"{name}: {status}", (300, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def check_gesture(self, finger_state):
        """Check gesture with stability"""
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
            
        if finger_state == self.prev_state:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.prev_state = finger_state
            
        if self.stable_count >= self.STABLE_THRESHOLD:
            if finger_state in self.GESTURE_MAP:
                self.stable_count = 0
                self.cooldown = 25  # Prevent rapid triggers
                return self.GESTURE_MAP[finger_state]
        
        return None

# ===== MAIN LOOP =====
def main():
    recognizer = ImprovedGestureRecognizer()
    
    # Camera setup
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return
    
    print("\n" + "="*50)
    print("IMPROVED GESTURE CONTROL READY")
    print("="*50)
    print(f"Controlling {len(cubes)} cubes")
    print("\nGESTURE MAPPINGS:")
    print("1 finger (Index):    Pick Cube 1")
    print("2 fingers:           Place Cube 1")
    print("Thumb only:          Pick Cube 2")
    print("Thumb + Index:       Place Cube 2")
    print("Middle only:         Pick Cube 3")
    print("3 fingers:           Place Cube 3")
    print("Pinky only:          Pick Cube 4")
    print("4 fingers:           Place Cube 4")
    print("All fingers:         Go Home")
    print("Fist:                Reset All")
    print("\nPress ESC to exit")
    print("="*50)
    
    last_gesture_time = 0
    gesture_cooldown = 1.0  # 1 second between gestures
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv.flip(frame, 1)
            
            # Process gesture
            finger_state, status_text = recognizer.process_frame(frame)
            
            # Check for gestures with cooldown
            current_time = time.time()
            if finger_state and (current_time - last_gesture_time > gesture_cooldown):
                gesture = recognizer.check_gesture(finger_state)
                if gesture:
                    last_gesture_time = current_time
                    cube_idx, operation = gesture
                    
                    # Execute operation with visual feedback
                    action_text = ""
                    if operation == 'pick' and cube_idx < len(cubes):
                        if pick_targets[cube_idx] and pick_targets[cube_idx].Valid():
                            action_text = f'PICKING CUBE {cube_idx + 1}'
                            cv.putText(frame, action_text, (50, 50), 
                                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            run_async(pick_sequence, cube_idx)
                        else:
                            action_text = f"NO PICK TARGET FOR CUBE {cube_idx + 1}"
                            cv.putText(frame, action_text, (50, 50), 
                                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    elif operation == 'place' and cube_idx < len(cubes):
                        if place_targets[cube_idx] and place_targets[cube_idx].Valid():
                            action_text = f'PLACING CUBE {cube_idx + 1}'
                            cv.putText(frame, action_text, (50, 50), 
                                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            run_async(place_sequence, cube_idx)
                        else:
                            action_text = f"NO PLACE TARGET FOR CUBE {cube_idx + 1}"
                            cv.putText(frame, action_text, (50, 50), 
                                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    elif operation == 'home':
                        action_text = 'GOING HOME'
                        cv.putText(frame, action_text, (50, 50), 
                                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        run_async(go_home)
                    
                    elif operation == 'reset':
                        action_text = 'RESETTING ALL OBJECTS'
                        cv.putText(frame, action_text, (50, 50), 
                                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        run_async(reset_all_objects)
            
            # Display information
            cv.putText(frame, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Operation status
            with operation_lock:
                op_status = current_operation if current_operation else "Ready"
            cv.putText(frame, f"Robot: {op_status}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(frame, f"Cubes: {len(cubes)}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Instructions
            cv.putText(frame, "Show clear hand gestures", (10, 400), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.putText(frame, "ESC to exit | R=Reset | H=Home", (10, 430), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv.imshow(f'Improved Gesture Control - {len(cubes)} Cubes', frame)
            
            # Key handling
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):
                run_async(reset_all_objects)
            elif key == ord('h'):
                run_async(go_home)
                
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        cap.release()
        cv.destroyAllWindows()
        reset_all_objects()
        print("Program ended cleanly")

if __name__ == "__main__":
    main()