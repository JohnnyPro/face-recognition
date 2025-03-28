import cv2
import mediapipe as mp
import math
import time
import numpy as np

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

class LipMovementDetector:
    def __init__(self,
                 max_num_faces=5,
                 detection_confidence=0.5,
                 tracking_confidence=0.5,
                 buffer_size=5,
                 ratio_start_threshold=0.35,
                 ratio_stop_threshold=0.28,
                 min_talking_duration=0.3,
                 tracking_threshold=0.05):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence)
        # For face tracking & ID assignment.
        self.face_trackers = {}  # face_id -> (center_x, center_y)
        self.next_face_id = 0
        self.tracking_threshold = tracking_threshold

        # Parameters for talking detection smoothing and hysteresis.
        self.BUFFER_SIZE = buffer_size
        self.RATIO_START_THRESHOLD = ratio_start_threshold
        self.RATIO_STOP_THRESHOLD = ratio_stop_threshold
        self.MIN_TALKING_DURATION = min_talking_duration

        # Buffers and state dictionaries.
        self.ratio_buffers = {}           # face_id -> list of recent mouth ratio values
        self.current_talking_state = {}   # face_id -> Boolean (True if currently talking)
        self.talking_start_time = {}      # face_id -> timestamp when talking started
        # For returning a bounding box per face.
        # We will compute bounding box from all landmarks.
    
    @staticmethod
    def euclidean_distance(pt1, pt2):
        return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    
    @staticmethod
    def compute_face_bbox(landmarks, img_width, img_height):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x_min = int(min(xs) * img_width)
        x_max = int(max(xs) * img_width)
        y_min = int(min(ys) * img_height)
        y_max = int(max(ys) * img_height)
        return (x_min, y_min, x_max, y_max)
    
    @staticmethod
    def get_face_center(landmarks):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    
    def process_frame(self, frame: np.ndarray):
        """
        Process one frame (BGR image) and return a dict mapping tracked face_id to:
          { "bbox": (x_min, y_min, x_max, y_max), "talking": bool }
        """
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w, _ = frame.shape
        current_faces = {}
        current_time = time.time()
        output = {}  # face_id -> detection info
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Compute face center for tracking.
                center = self.get_face_center(face_landmarks.landmark)
                assigned_id = None
                for face_id, prev_center in self.face_trackers.items():
                    if self.euclidean_distance(center, prev_center) < self.tracking_threshold:
                        assigned_id = face_id
                        break
                if assigned_id is None:
                    assigned_id = self.next_face_id
                    self.next_face_id += 1
                current_faces[assigned_id] = center

                # Compute bounding box for the entire face from all landmarks.
                bbox = self.compute_face_bbox(face_landmarks.landmark, w, h)

                # Compute mouth aspect ratio.
                try:
                    lm13 = face_landmarks.landmark[13]
                    lm14 = face_landmarks.landmark[14]
                    lm78 = face_landmarks.landmark[78]
                    lm308 = face_landmarks.landmark[308]
                    mouth_height = self.euclidean_distance((lm13.x, lm13.y), (lm14.x, lm14.y))
                    mouth_width = self.euclidean_distance((lm78.x, lm78.y), (lm308.x, lm308.y))
                    ratio = mouth_height / mouth_width if mouth_width != 0 else 0
                except Exception:
                    ratio = 0

                # Update the moving average buffer.
                self.ratio_buffers.setdefault(assigned_id, []).append(ratio)
                if len(self.ratio_buffers[assigned_id]) > self.BUFFER_SIZE:
                    self.ratio_buffers[assigned_id].pop(0)
                avg_ratio = sum(self.ratio_buffers[assigned_id]) / len(self.ratio_buffers[assigned_id])

                # Hysteresis-based talking decision.
                if self.current_talking_state.get(assigned_id, False):
                    is_talking = avg_ratio > self.RATIO_STOP_THRESHOLD
                else:
                    is_talking = avg_ratio > self.RATIO_START_THRESHOLD

                # Update talking intervals (we only keep state per frame here).
                if is_talking:
                    if not self.current_talking_state.get(assigned_id, False):
                        self.talking_start_time[assigned_id] = current_time
                    self.current_talking_state[assigned_id] = True
                else:
                    if self.current_talking_state.get(assigned_id, False):
                        # Record the interval duration here.
                        self.talking_start_time[assigned_id] = None
                    self.current_talking_state[assigned_id] = False

                # Save output info.
                output[assigned_id] = {"bbox": bbox, "talking": is_talking}

        # Update trackers.
        self.face_trackers = current_faces
        return output

    def process_frames(self, frames):
        """
        Process a list of frames and aggregate talking status.
        Returns a dict: tracked_face_id -> aggregated { "bbox": last_bbox, "talking": bool }
        If a face is detected as talking in more than half of the frames, we mark it as talking.
        """
        aggregated = {}
        count = {}
        for frame in frames:
            detections = self.process_frame(frame)
            for face_id, info in detections.items():
                aggregated.setdefault(face_id, {"bbox": info["bbox"], "talking_count": 0, "total": 0})
                aggregated[face_id]["total"] += 1
                if info["talking"]:
                    aggregated[face_id]["talking_count"] += 1
        # Decide talking state.
        for face_id in aggregated:
            aggregated[face_id]["talking"] = aggregated[face_id]["talking_count"] > aggregated[face_id]["total"] / 2
            # Remove helper counts.
            del aggregated[face_id]["talking_count"]
            del aggregated[face_id]["total"]
        return aggregated
