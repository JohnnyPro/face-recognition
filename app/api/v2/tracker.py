# Add these imports at the top of your api file
from collections import deque


class SimpleFaceTracker:
    def __init__(self, iou_threshold=0.3, max_missed_frames=20):  # Increased to 20 frames
        # {track_id: {"person_id": str, "bbox": list, "confidence": float, "missed": int}}
        self.tracked_faces = {}
        self.next_track_id = 0
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames

    def update_tracks(self, matches):
        """Update tracks with new recognition results"""
        current_tracks = {}

        # First increment missed frames for all existing tracks
        for track_id, track in list(self.tracked_faces.items()):
            track["missed"] += 1
            if track["missed"] <= self.max_missed_frames:
                current_tracks[track_id] = track

        # Try to match new detections with existing tracks
        for match in matches:
            if match.person_id == "Unknown":
                continue  # Only track known persons

            best_match_id = None
            best_iou = 0

            for track_id, track in current_tracks.items():
                if track["person_id"] != match.person_id:
                    continue  # Only match same person IDs

                iou = self.calculate_iou(track["bbox"], match.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_match_id = track_id

            if best_match_id is not None:
                # Update existing track
                current_tracks[best_match_id]["bbox"] = match.bbox
                current_tracks[best_match_id]["confidence"] = match.confidence
                current_tracks[best_match_id]["missed"] = 0
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                current_tracks[track_id] = {
                    "person_id": match.person_id,
                    "bbox": match.bbox,
                    "confidence": match.confidence,
                    "missed": 0
                }

        self.tracked_faces = current_tracks
        return current_tracks

    def get_previous_frame_tracks(self):
        """Get tracked faces from previous frame (before current update)"""
        return [
            {
                "person_id": track["person_id"],
                "bbox": track["bbox"],
                # Slight confidence reduction
                "confidence": track["confidence"] * 0.95
            }
            for track in self.tracked_faces.values()
        ]

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        # box format: [x, y, width, height]

        # Convert to [x1, y1, x2, y2]
        box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

        # Calculate intersection area
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0
