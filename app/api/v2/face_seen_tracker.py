from datetime import datetime, timedelta


class MatchTimeTracker:
    def __init__(self):
        self.face_states = {}  # person_id -> state

    def update(self, matches):
        now = datetime.utcnow()
        for match in matches:
            if match.person_id.lower() == "unknown":
                continue

            state = self.face_states.get(match.person_id)

            if state:
                time_since_last_seen = now - state["last_seen_time"]

                if time_since_last_seen > timedelta(hours=2):
                    # Remove if not seen for over 2 hours
                    self.face_states.pop(match.person_id)
                    continue

                elif time_since_last_seen > timedelta(minutes=1):
                    # Reset frames but keep greeted status
                    state["frames_present"] = 0
                    state["last_seen_time"] = now

                else:
                    state["frames_present"] += 1
                    state["last_seen_time"] = now

                # Suppress greeting if they linger too long without one
                if state["frames_present"] > 30:
                    state["greeted"] = True

            else:
                # First time seeing this person
                self.face_states[match.person_id] = {
                    "last_seen_time": now,
                    "frames_present": 1,
                    "greeted": False
                }
                
    def get_new_faces(self):
        """Return list of person_ids who have not been greeted and are in frame between 10–30 frames"""
        now = datetime.utcnow()
        return [
            person_id
            for person_id, state in self.face_states.items()
            if (
                not state["greeted"]
                and 10 <= state["frames_present"] <= 30
                and (now - state["last_seen_time"]) <= timedelta(minutes=1)
            )
        ]

    def mark_greeted(self, person_id):
        if person_id in self.face_states:
            self.face_states[person_id]["greeted"] = True
