import os

# Quiet the harmless TensorFlow Lite / MediaPipe C++ log spam.

# Must be set BEFORE importing mediapipe.

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("GLOG_minloglevel", "2")

import re

import json

import time

import threading

from datetime import datetime

import cv2

import numpy as np

from ultralytics import YOLO

from database import (

    create_database,

    save_observation,

    get_last_seen,

    save_fingerprint,

    prune_fingerprints,

    get_fingerprints,

    get_all_object_memories,

    get_highest_recall_id,

    save_holding_event,

    get_last_holding,

    get_holding_at_time,

    save_memory_event,

    get_recent_events,

    get_events_for_object,

    clear_all_memory,

    create_visual_fingerprint

)

from voice import VoiceAssistant

import mediapipe as mp

from mediapipe.tasks import python

from mediapipe.tasks.python import vision

# ------------------------------------------------------------

# CONFIG

# ------------------------------------------------------------

MODEL_PATH = "yolo11s.pt"

CAMERA_WIDTH = 640

CAMERA_HEIGHT = 480

OBJECT_CONFIDENCE = 0.70

HAND_CONFIDENCE = 0.80

MATCH_THRESHOLD = 0.72

FIND_MATCH_THRESHOLD = 0.65

MEMORY_INTERVAL = 3.0

HOLD_CONFIRMATION_TIME = 0.5

HOLD_RELEASE_TIME = 0.8

HOLD_DISTANCE_RATIO = 0.35

MAX_FINGERPRINTS_PER_OBJECT = 20

# ------------------------------------------------------------
# RELIABILITY
# ------------------------------------------------------------
DETECTION_CONFIRM_FRAMES = 3
DETECTION_CONFIRM_WINDOW = 6
CONFIDENCE_EMA_ALPHA = 0.35
LOCATION_CONFIRM_FRAMES = 3
TRACK_LOST_TIMEOUT = 1.5

# Voice

VOICE_ENABLED = True            # turn the whole voice layer on/off

SPEAK_ANSWERS = True            # read answers out loud

CONTINUOUS_LISTENING = False    # True = always listen for "recall ..."

MIC_DEVICE =None              # None = default microphone

HAND_MODEL_PATH = os.path.join("data", "hand_landmarker.task")

IGNORED_CLASSES = {"person"}

ALIASES = {

    "phone": "cell phone",

    "mobile": "cell phone",

    "smartphone": "cell phone",

    "cellphone": "cell phone",

    "computer": "laptop",

    "notebook": "laptop",

    "bag": "backpack",

    "school bag": "backpack",

    "television": "tv",

    "fridge": "refrigerator",

    "remote control": "remote",

    "plant": "potted plant",

    "flower pot": "potted plant"

}

KNOWN_NAMES = [

    "cell phone", "phone", "mobile", "smartphone", "book", "cup",

    "laptop", "computer", "backpack", "bag", "bottle", "keyboard",

    "mouse", "remote", "tv", "chair", "clock", "bicycle", "car",

    "scissors", "pen", "potted plant"

]

# ------------------------------------------------------------

# HELPERS

# ------------------------------------------------------------

def normalize_object_name(name):

    name = str(name).lower().strip()

    return ALIASES.get(name, name)

NUMBER_WORDS = {

    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,

    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,

    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,

    "thirty": 30, "forty": 40, "sixty": 60

}

CONTRACTIONS = {

    "where's": "where is", "wheres": "where is",

    "what's": "what is", "whats": "what is",

    "what're": "what are"

}

def normalize_command(text):

    """

    Make typed OR spoken commands uniform:

    lowercase, no punctuation, no wake word, "five" -> "5".

    """

    text = str(text).lower().strip().replace("\u2019", "'")

    text = re.sub(

        r"^(?:(?:hey|ok|okay)\s+)?(?:recall\s*cam|recall)\b[,:]?\s*",

        "",

        text

    )

    for short, full in CONTRACTIONS.items():

        text = re.sub(r"\b" + re.escape(short) + r"\b", full, text)

    text = re.sub(r"[?!.,;:]", "", text)

    text = re.sub(

        r"\b(" + "|".join(NUMBER_WORDS) + r")\b(?=\s+seconds?\b)",

        lambda m: str(NUMBER_WORDS[m.group(1)]),

        text

    )

    return re.sub(r"\s+", " ", text).strip()

def clean_object_text(text):

    """'my phone' / 'the cup' / 'a book' -> 'phone' / 'cup' / 'book'"""

    text = text.strip()

    for _ in range(2):

        text = re.sub(r"^(?:my|the|a|an)\s+", "", text)

    return text.strip()

def join_names(names):

    names = list(names)

    if not names:

        return "nothing"

    if len(names) == 1:

        return names[0]

    return ", ".join(names[:-1]) + " and " + names[-1]

def time_ago(timestamp_text):

    """'2026-10-06 10:00:00' -> '3 minutes ago' (for speaking)."""

    try:

        then = datetime.strptime(str(timestamp_text), "%Y-%m-%d %H:%M:%S")

        seconds = max(0, int((datetime.now() - then).total_seconds()))

    except Exception:

        return "at an unknown time"

    if seconds < 10:

        return "just now"

    if seconds < 60:

        return f"{seconds} seconds ago"

    minutes = seconds // 60

    if minutes < 60:

        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"

    hours = minutes // 60

    if hours < 24:

        return f"{hours} hour{'s' if hours != 1 else ''} ago"

    days = hours // 24

    return f"{days} day{'s' if days != 1 else ''} ago"

def describe_event(event_type, name):

    return str(event_type).replace("_", " ").lower() + " " + str(name)

def get_location(center_x, frame_width):

    ratio = center_x / max(1, frame_width)

    if ratio < 0.33:

        return "LEFT"

    if ratio > 0.66:

        return "RIGHT"

    return "CENTER"

def fingerprint_similarity(a, b):

    try:

        if a is None or b is None:

            return 0.0

        a = np.asarray(a, dtype=np.float32)

        b = np.asarray(b, dtype=np.float32)

        if len(a) != len(b) or len(a) < 273:

            return 0.0

        # Colour histogram

        hist_a = a[:256] / (np.linalg.norm(a[:256]) + 1e-8)

        hist_b = b[:256] / (np.linalg.norm(b[:256]) + 1e-8)

        color_similarity = float(np.dot(hist_a, hist_b))

        # Edge layout

        edge_distance = float(np.mean(np.abs(a[256:272] - b[256:272])))

        edge_similarity = max(0.0, 1.0 - edge_distance)

        # Aspect ratio

        shape_difference = abs(float(a[272]) - float(b[272]))

        shape_similarity = max(0.0, 1.0 - min(shape_difference, 1.0))

        similarity = (

            0.70 * color_similarity

            + 0.20 * edge_similarity

            + 0.10 * shape_similarity

        )

        return float(max(0.0, min(1.0, similarity)))

    except Exception:

        return 0.0

def bbox_similarity(current_bbox, old_width, old_height):

    try:

        x1, y1, x2, y2 = current_bbox

        current_ratio = max(1, x2 - x1) / max(1, y2 - y1)

        old_ratio = float(old_width) / max(1.0, float(old_height))

        difference = abs(current_ratio - old_ratio)

        return max(0.0, 1.0 - min(difference, 1.0))

    except Exception:

        return 0.0

def center_similarity(

    current_bbox, old_center_x, old_center_y, frame_width, frame_height

):

    try:

        x1, y1, x2, y2 = current_bbox

        dx = ((x1 + x2) / 2.0 - old_center_x) / max(1, frame_width)

        dy = ((y1 + y2) / 2.0 - old_center_y) / max(1, frame_height)

        distance = np.sqrt(dx * dx + dy * dy)

        return max(0.0, 1.0 - min(distance * 2.0, 1.0))

    except Exception:

        return 0.0

# ------------------------------------------------------------

# MAIN CLASS

# ------------------------------------------------------------

class RecallCam:

    def __init__(self):

        create_database()

        self.model = YOLO(MODEL_PATH)

        self.camera = None

        self.recall_counter = max(1, get_highest_recall_id() + 1)

        # Tracker ID -> Recall ID

        self.tracker_to_recall = {}

        # Recall ID -> current tracker

        self.recall_to_tracker = {}

        self.memory_last_saved = {}

        # Reliability state
        self.detection_history = {}
        self.smoothed_confidence = {}
        self.location_state = {}
        self.recall_last_seen = {}
        self.recall_last_bbox = {}

        self.holding_candidates = {}

        self.active_holding = {}

        self.find_mode = False

        self.find_target_name = None

        self.find_target_id = None

        self.hand_landmarker = None

        self.setup_hands()

        self.running = True

        self.pending_forget = 0.0

        self.find_announced = False

        self.find_last_found = 0.0

        self.voice = None

        if VOICE_ENABLED:

            self.voice = VoiceAssistant(

                on_command=lambda text: self.handle_command(

                    text, source="voice"

                ),

                speak_enabled=SPEAK_ANSWERS,

                mic_device=MIC_DEVICE

            )

    # --------------------------------------------------------

    # HAND DETECTION

    # --------------------------------------------------------

    def setup_hands(self):

        if not os.path.exists(HAND_MODEL_PATH):

            print(f"[ERROR] Hand model not found: {HAND_MODEL_PATH}")

            return

        try:

            base_options = python.BaseOptions(

                model_asset_path=HAND_MODEL_PATH

            )

            options = vision.HandLandmarkerOptions(

                base_options=base_options,

                running_mode=vision.RunningMode.VIDEO,

                num_hands=2,

                min_hand_detection_confidence=HAND_CONFIDENCE,

                min_hand_presence_confidence=HAND_CONFIDENCE,

                min_tracking_confidence=HAND_CONFIDENCE

            )

            self.hand_landmarker = (

                vision.HandLandmarker.create_from_options(options)

            )

        except Exception as e:

            print(f"[ERROR] Could not initialize hands: {e}")

    def detect_hands(self, frame, timestamp_ms):

        if self.hand_landmarker is None:

            return []

        try:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(

                image_format=mp.ImageFormat.SRGB,

                data=rgb

            )

            result = self.hand_landmarker.detect_for_video(

                mp_image, timestamp_ms

            )

            if not result.hand_landmarks:

                return []

            frame_h, frame_w = frame.shape[:2]

            hands = []

            for landmarks in result.hand_landmarks:

                xs = [lm.x * frame_w for lm in landmarks]

                ys = [lm.y * frame_h for lm in landmarks]

                hands.append((

                    int(max(0, min(xs))),

                    int(max(0, min(ys))),

                    int(min(frame_w - 1, max(xs))),

                    int(min(frame_h - 1, max(ys)))

                ))

            return hands

        except Exception:

            return []

    # --------------------------------------------------------

    # RELIABILITY
    # --------------------------------------------------------

    def update_detection_reliability(self, object_name, tracker_id, confidence, bbox):
        """Require several consistent frames before trusting a detection."""
        if tracker_id >= 0:
            key = (object_name, tracker_id)
        else:
            x1, y1, x2, y2 = bbox
            key = (object_name, -1, int(((x1 + x2) / 2) / 40), int(((y1 + y2) / 2) / 40))

        history = self.detection_history.setdefault(key, [])
        now = time.time()
        history.append({"time": now, "confidence": float(confidence), "bbox": bbox})
        cutoff = now - 1.0
        history[:] = [item for item in history if item["time"] >= cutoff]
        history[:] = history[-DETECTION_CONFIRM_WINDOW:]

        previous = self.smoothed_confidence.get(key)
        if previous is None:
            smoothed = float(confidence)
        else:
            smoothed = (
                CONFIDENCE_EMA_ALPHA * float(confidence)
                + (1.0 - CONFIDENCE_EMA_ALPHA) * previous
            )
        self.smoothed_confidence[key] = smoothed
        return len(history) >= DETECTION_CONFIRM_FRAMES, smoothed

    def update_stable_location(self, recall_id, detected_location):
        """Require a location to persist before changing LEFT/CENTER/RIGHT."""
        state = self.location_state.setdefault(
            recall_id,
            {"stable": detected_location, "candidate": detected_location, "count": 0}
        )
        if detected_location == state["stable"]:
            state["candidate"] = detected_location
            state["count"] = 0
            return state["stable"]
        if detected_location != state["candidate"]:
            state["candidate"] = detected_location
            state["count"] = 1
            return state["stable"]
        state["count"] += 1
        if state["count"] >= LOCATION_CONFIRM_FRAMES:
            old_location = state["stable"]
            state["stable"] = detected_location
            state["candidate"] = detected_location
            state["count"] = 0
            if old_location != detected_location:
                print(f"[LOCATION] R{recall_id}: {old_location} -> {detected_location}")
        return state["stable"]

    def cleanup_reliability_state(self):
        """Remove stale tracker-level reliability history."""
        now = time.time()
        active_tracker_ids = set(self.tracker_to_recall.keys())
        for key in list(self.detection_history.keys()):
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            tracker_id = key[1]
            if tracker_id < 0 or tracker_id in active_tracker_ids:
                continue
            history = self.detection_history.get(key, [])
            if history and now - history[-1]["time"] > TRACK_LOST_TIMEOUT:
                self.detection_history.pop(key, None)
                self.smoothed_confidence.pop(key, None)

    # --------------------------------------------------------

    # RECALL IDS

    # --------------------------------------------------------

    def next_recall_id(self):

        recall_id = self.recall_counter

        self.recall_counter += 1

        return recall_id

    def strong_reidentification(

        self, object_name, frame, bbox, blocked_recall_ids=None

    ):

        """

        Find the strongest historical identity for this object.

        blocked_recall_ids stops one Recall ID being given to two

        objects in the same frame.

        """

        blocked = blocked_recall_ids or set()

        fingerprint = create_visual_fingerprint(frame, bbox)

        if fingerprint is None:

            return None

        memories = get_all_object_memories(object_name)

        if not memories:

            return None

        frame_h, frame_w = frame.shape[:2]

        current_location = get_location(

            (bbox[0] + bbox[2]) / 2, frame_w

        )

        best_match = None

        for row in memories:

            (

                recall_id, name, location, old_width, old_height,

                old_center_x, old_center_y, old_fingerprint, timestamp

            ) = row

            if recall_id is None:

                continue

            recall_id = int(recall_id)

            if recall_id in blocked:

                continue

            try:

                fingerprints = get_fingerprints(

                    recall_id, MAX_FINGERPRINTS_PER_OBJECT

                )

            except Exception:

                fingerprints = []

            if not fingerprints and old_fingerprint:

                try:

                    fingerprints = [json.loads(old_fingerprint)]

                except Exception:

                    fingerprints = []

            if not fingerprints:

                continue

            visual_score = max(

                fingerprint_similarity(fingerprint, fp)

                for fp in fingerprints

            )

            shape_score = bbox_similarity(bbox, old_width, old_height)

            position_score = center_similarity(

                bbox, old_center_x, old_center_y, frame_w, frame_h

            )

            location_score = 1.0 if current_location == location else 0.0

            combined_score = (

                0.65 * visual_score

                + 0.15 * shape_score

                + 0.10 * position_score

                + 0.10 * location_score

            )

            if combined_score >= MATCH_THRESHOLD and visual_score >= 0.60:

                if best_match is None or combined_score > best_match[2]:

                    best_match = (recall_id, visual_score, combined_score)

        return best_match

    def get_or_create_recall_id(

        self, object_name, tracker_id, frame, bbox, blocked_recall_ids=None

    ):

        blocked = blocked_recall_ids or set()

        # Existing tracker mapping (ignore untracked objects, id < 0)

        if tracker_id >= 0 and tracker_id in self.tracker_to_recall:

            recall_id = self.tracker_to_recall[tracker_id]

            if recall_id not in blocked:

                self.recall_to_tracker[recall_id] = tracker_id

                return recall_id

            # Another object already claimed it this frame.

            del self.tracker_to_recall[tracker_id]

        # Visual re-identification

        match = self.strong_reidentification(

            object_name, frame, bbox, blocked

        )

        if match is not None:

            recall_id, visual_score, combined_score = match

            print(

                f"[RE-ID] {object_name} matched R{recall_id} "

                f"visual={visual_score:.2f} combined={combined_score:.2f}"

            )

        else:

            recall_id = self.next_recall_id()

            print(f"[MEMORY] NEW {object_name} | Recall ID={recall_id}")

        if tracker_id >= 0:

            self.tracker_to_recall[tracker_id] = recall_id

        self.recall_to_tracker[recall_id] = tracker_id

        return recall_id

    # --------------------------------------------------------

    # MEMORY

    # --------------------------------------------------------

    def save_object_memory(
        self, object_name, tracker_id, recall_id, confidence, frame, bbox, location=None
    ):

        now = time.time()

        last = self.memory_last_saved.get(recall_id)

        if last is not None and now - last < MEMORY_INTERVAL:

            return

        self.memory_last_saved[recall_id] = now

        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) / 2

        center_y = (y1 + y2) / 2

        if location is None:
            location = get_location(center_x, frame.shape[1])

        frame_dir = os.path.join("data", "frames")

        os.makedirs(frame_dir, exist_ok=True)

        file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        image_path = os.path.join(

            frame_dir,

            f"{file_stamp}_{object_name.replace(' ', '_')}_"

            f"recall{recall_id}_track{tracker_id}.jpg"

        )

        cv2.imwrite(image_path, frame)

        fingerprint = create_visual_fingerprint(frame, bbox)

        try:

            save_observation(

                object_name=object_name,

                tracking_id=tracker_id,

                confidence=confidence,

                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                image_path=image_path,

                location=location,

                recall_id=recall_id,

                fingerprint=fingerprint,

                bbox_width=x2 - x1,

                bbox_height=y2 - y1,

                center_x=center_x,

                center_y=center_y

            )

            if fingerprint is not None:

                save_fingerprint(object_name, recall_id, fingerprint)

                prune_fingerprints(recall_id, MAX_FINGERPRINTS_PER_OBJECT)

        except Exception as e:

            # Try again on the next frame instead of crashing.

            self.memory_last_saved.pop(recall_id, None)

            print(f"[DB] Could not save memory: {e}")

    # --------------------------------------------------------

    # HOLDING

    # --------------------------------------------------------

    @staticmethod

    def intersection_ratio(hand_bbox, object_bbox):

        hx1, hy1, hx2, hy2 = hand_bbox

        ox1, oy1, ox2, oy2 = object_bbox

        ix1 = max(hx1, ox1)

        iy1 = max(hy1, oy1)

        ix2 = min(hx2, ox2)

        iy2 = min(hy2, oy2)

        if ix2 <= ix1 or iy2 <= iy1:

            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)

        object_area = max(1, (ox2 - ox1) * (oy2 - oy1))

        return intersection / object_area

    def is_holding(self, hand_boxes, object_bbox):

        if not hand_boxes:

            return False

        ox1, oy1, ox2, oy2 = object_bbox

        object_center_x = (ox1 + ox2) / 2

        object_center_y = (oy1 + oy2) / 2

        max_distance = (

            max(ox2 - ox1, oy2 - oy1, 1) * HOLD_DISTANCE_RATIO

        )

        for hand_bbox in hand_boxes:

            if self.intersection_ratio(hand_bbox, object_bbox) > 0.05:

                return True

            hx1, hy1, hx2, hy2 = hand_bbox

            distance = np.sqrt(

                ((hx1 + hx2) / 2 - object_center_x) ** 2

                + ((hy1 + hy2) / 2 - object_center_y) ** 2

            )

            if distance <= max_distance:

                return True

        return False

    def end_hold(self, recall_id):

        """Finish a confirmed hold and write it to the database."""

        active = self.active_holding.pop(recall_id, None)

        self.holding_candidates.pop(recall_id, None)

        if not active:

            return

        object_name = active["object_name"]

        started = active["started"]

        last_seen = active["last_seen"]

        duration = max(0.0, last_seen - started)

        fmt = "%Y-%m-%d %H:%M:%S"

        try:

            save_holding_event(

                object_name=object_name,

                recall_id=recall_id,

                started_at=datetime.fromtimestamp(started).strftime(fmt),

                ended_at=datetime.fromtimestamp(last_seen).strftime(fmt),

                duration=duration,

                started_at_unix=started,

                ended_at_unix=last_seen

            )

            save_memory_event(

                event_type="PUT_DOWN",

                object_name=object_name,

                recall_id=recall_id,

                details=f"Held for {duration:.1f} seconds",

                event_time_unix=last_seen

            )

        except Exception as e:

            print(f"[DB] Could not save holding event: {e}")

        print(f"[EVENT] PUT DOWN | {object_name} | Held {duration:.1f}s")

    def update_holding(

        self, object_name, recall_id, object_bbox, hand_boxes, frame

    ):

        now = time.time()

        if self.is_holding(hand_boxes, object_bbox):

            candidate = self.holding_candidates.setdefault(

                recall_id,

                {

                    "object_name": object_name,

                    "started": now,

                    "last_touch": now

                }

            )

            candidate["last_touch"] = now

            if recall_id in self.active_holding:

                self.active_holding[recall_id]["last_seen"] = now

            elif now - candidate["started"] >= HOLD_CONFIRMATION_TIME:

                self.active_holding[recall_id] = {

                    "object_name": object_name,

                    "started": candidate["started"],

                    "last_seen": now

                }

                print(

                    f"[EVENT] PICKED UP | {object_name} | "

                    f"Recall ID={recall_id}"

                )

                save_memory_event(

                    event_type="PICKED_UP",

                    object_name=object_name,

                    recall_id=recall_id,

                    location=get_location(

                        (object_bbox[0] + object_bbox[2]) / 2,

                        frame.shape[1]

                    ),

                    details="Object picked up",

                    event_time_unix=candidate["started"]

                )

        else:

            if recall_id in self.active_holding:

                active = self.active_holding[recall_id]

                if now - active["last_seen"] >= HOLD_RELEASE_TIME:

                    self.end_hold(recall_id)

            elif recall_id in self.holding_candidates:

                candidate = self.holding_candidates[recall_id]

                if now - candidate["last_touch"] >= HOLD_RELEASE_TIME:

                    del self.holding_candidates[recall_id]

    def expire_holds(self, visible_recall_ids):

        """End holds for objects that disappeared from the camera."""

        now = time.time()

        for recall_id in list(self.active_holding.keys()):

            if recall_id in visible_recall_ids:

                continue

            active = self.active_holding.get(recall_id)

            if active and now - active["last_seen"] >= HOLD_RELEASE_TIME:

                self.end_hold(recall_id)

        for recall_id in list(self.holding_candidates.keys()):

            if recall_id in visible_recall_ids:

                continue

            candidate = self.holding_candidates.get(recall_id)

            if (

                candidate

                and recall_id not in self.active_holding

                and now - candidate["last_touch"] >= HOLD_RELEASE_TIME

            ):

                del self.holding_candidates[recall_id]

    # --------------------------------------------------------

    # FIND MODE

    # --------------------------------------------------------

    def start_find_mode(self, object_name):

        object_name = normalize_object_name(object_name)

        self.find_mode = True

        self.find_target_name = object_name

        self.find_announced = False

        latest = get_last_seen(object_name)

        if latest:

            self.find_target_id = latest[1]

            print(

                f"Find Mode: looking for {object_name} "

                f"[R{self.find_target_id}]"

            )

        else:

            self.find_target_id = None

            print(f"Find Mode: looking for {object_name}")

    def stop_find_mode(self):

        self.find_mode = False

        self.find_target_name = None

        self.find_target_id = None

        print("Find Mode stopped.")

    def check_find_match(self, object_name, frame, bbox, recall_id):

        if not self.find_mode or object_name != self.find_target_name:

            return False, 0.0

        # Never seen this object before: any object of that class counts.

        if self.find_target_id is None:

            return True, 1.0

        if recall_id == self.find_target_id:

            return True, 1.0

        current_fp = create_visual_fingerprint(frame, bbox)

        if current_fp is None:

            return False, 0.0

        best_score = 0.0

        for stored_fp in get_fingerprints(

            self.find_target_id, MAX_FINGERPRINTS_PER_OBJECT

        ):

            best_score = max(

                best_score, fingerprint_similarity(current_fp, stored_fp)

            )

        return best_score >= FIND_MATCH_THRESHOLD, best_score

    # --------------------------------------------------------

    # COMMANDS

    # --------------------------------------------------------

    @staticmethod

    def find_object_name_from_text(text):

        text = text.lower()

        for name in KNOWN_NAMES:

            if name in text:

                return normalize_object_name(name)

        return None

    def reply(self, text, spoken=None):

        """Print an answer and read it out loud."""

        print(text)

        if self.voice is not None:

            self.voice.say(spoken if spoken is not None else text)

    def do_forget(self):

        clear_all_memory()

        self.tracker_to_recall.clear()

        self.recall_to_tracker.clear()

        self.memory_last_saved.clear()

        self.holding_candidates.clear()

        self.active_holding.clear()

        self.find_mode = False

        self.find_target_name = None

        self.find_target_id = None

        self.recall_counter = 1

        self.reply("All RecallCam memory has been cleared.")

    def handle_command(self, command, source="typed"):

        command = normalize_command(command)

        if not command:

            return

        # Confirm a pending (voice) "forget everything"

        if command in {"confirm", "confirm forget", "yes confirm"}:

            if time.time() - self.pending_forget <= 15:

                self.pending_forget = 0.0

                self.do_forget()

            else:

                self.reply("There is nothing to confirm.")

            return

        # Voice controls

        if command in {"mute", "be quiet", "stop talking"}:

            if self.voice is not None:

                self.voice.mute()

            print("Spoken answers muted. Type or say unmute to resume.")

            return

        if command in {"unmute", "start talking"}:

            if self.voice is not None:

                self.voice.speak_enabled = True

            self.reply("Spoken answers are on.")

            return

        if command in {"voice on", "start listening"}:

            if self.voice is not None and self.voice.set_continuous(True):

                self.reply(

                    "I'm listening. Start commands with the word recall."

                )

            return

        if command in {"voice off", "stop listening"}:

            if self.voice is not None:

                self.voice.set_continuous(False)

            self.reply("Voice listening is off.")

            return

        if command in {"listen", "listen now"}:

            if self.voice is not None:

                self.voice.listen_once()

            return

        # Exit

        if command in {"q", "quit", "exit", "close"}:

            self.reply("Closing RecallCam.")

            self.running = False

            return

        # Stop find

        if command in {"stop finding", "stop find", "cancel find"}:

            self.stop_find_mode()

            return

        # Forget everything

        if command in {

            "forget everything", "forget all", "clear memory",

            "clear everything", "forget memory", "delete memory",

            "clear all"

        }:

            if source == "voice":

                self.pending_forget = time.time()

                self.reply(

                    "This will erase all memory. "

                    "Say confirm within 15 seconds to continue."

                )

                return

            self.do_forget()

            return

        # Currently holding

        if command in {

            "what am i holding", "what am i holding?",

            "what is in my hand", "what is in my hand?",

            "what's in my hand", "whats in my hand",

            "what am i carrying", "what am i carrying?",

            "what do i have in my hand", "what do i have in my hand?"

        }:

            holds = list(self.active_holding.values())

            if not holds:

                self.reply("You are not currently holding anything.")

                return

            latest = max(holds, key=lambda x: x["last_seen"])

            self.reply(f"You are currently holding {latest['object_name']}.")

            return

        # Last holding

        if command in {

            "what was i holding", "what was i holding?",

            "what was in my hand", "what was in my hand?",

            "what did i have in my hand", "what did i have in my hand?",

            "what did i just hold", "what did i just hold?",

            "what was i carrying", "what was i carrying?"

        }:

            row = get_last_holding()

            if row is None:

                self.reply("I don't have any previous holding event.")

                return

            self.reply(f"You were holding {row[1]} for {row[5]:.1f} seconds.")

            return

        # Holding X seconds ago

        seconds_match = re.search(

            r"(?:what was i holding|what was in my hand|"

            r"what did i just hold).*?(\d+(?:**\\.**\d+)?)\s*seconds?\s*ago",

            command

        )

        if seconds_match:

            seconds_ago = float(seconds_match.group(1))

            row = get_holding_at_time(

                time.time() - seconds_ago, tolerance=3

            )

            if row is None:

                self.reply(

                    f"I don't have a holding record from "

                    f"{seconds_ago:g} seconds ago."

                )

                return

            self.reply(

                f"{seconds_ago:g} seconds ago, you were holding {row[1]}."

            )

            return

        # Was I holding X?

        if command.startswith("was i holding "):

            wanted = normalize_object_name(

                clean_object_text(command[len("was i holding "):])

            )

            row = get_last_holding()

            if row is None:

                self.reply("I don't have a previous holding record.")

                return

            if normalize_object_name(row[1]) == wanted:

                self.reply(

                    f"Yes. You were holding {row[1]} "

                    f"for {row[5]:.1f} seconds."

                )

            else:

                self.reply(f"No. Your latest recorded holding was {row[1]}.")

            return

        # Find mode

        find_match = re.match(

            r"^(?:find|look for)\s+(?:my\s+)?(.+)$", command

        )

        if find_match:

            self.start_find_mode(clean_object_text(find_match.group(1)))

            if self.voice is not None:

                self.voice.say(

                    f"Looking for your {self.find_target_name}."

                )

            return

        # Where is object?

        where_match = re.match(r"^where (?:is|are) (.+)$", command)

        if where_match:

            object_name = normalize_object_name(

                clean_object_text(where_match.group(1))

            )

            row = get_last_seen(object_name)

            if row is None:

                self.reply(f"I haven't seen {object_name} yet.")

                return

            self.reply(

                f"{row[0]} was last seen at {row[2]} "

                f"(Recall ID={row[1]}) at {row[3]}.",

                spoken=(

                    f"Your {row[0]} was last seen on the "

                    f"{str(row[2]).lower()}, {time_ago(row[3])}."

                )

            )

            return

        # What have you seen?

        if command in {

            "what have you seen", "what have you seen?",

            "what did you see", "what did you see?"

        }:

            memories = get_all_object_memories()

            if not memories:

                self.reply("I haven't seen any objects yet.")

                return

            lines = [

                f"- {row[1]} (Recall ID={row[0]})" for row in memories

            ]

            names = sorted({row[1] for row in memories})

            self.reply(

                "I have seen:\n" + "\n".join(lines),

                spoken="I have seen " + join_names(names) + "."

            )

            return

        # Recent events

        if command in {

            "what happened recently", "what happened recently?",

            "what happened in the last minute",

            "what happened in the last minute?"

        }:

            events = get_recent_events(60)

            if not events:

                self.reply("Nothing recorded in the last minute.")

                return

            lines = []

            for (

                event_type, name, recall_id, location,

                timestamp, details, event_time

            ) in events:

                line = f"{timestamp} | {event_type} | {name} | R{recall_id}"

                if details:

                    line += f" | {details}"

                lines.append(line)

            latest = [

                describe_event(e[0], e[1]) for e in events[:3]

            ]

            self.reply(

                "\n".join(lines),

                spoken="Recently: " + join_names(latest) + "."

            )

            return

        # Object timeline

        object_name = self.find_object_name_from_text(command)

        if object_name and (

            "timeline" in command

            or "history" in command

            or "events" in command

        ):

            events = get_events_for_object(object_name)

            if not events:

                self.reply(f"No events recorded for {object_name}.")

                return

            lines = []

            for (

                event_type, name, recall_id, location,

                timestamp, details, event_time

            ) in events:

                lines.append(

                    f"{timestamp} | {event_type} | R{recall_id} | "

                    f"{location or ''} | {details or ''}"

                )

            latest = [

                describe_event(e[0], object_name) for e in events[:3]

            ]

            self.reply(

                "\n".join(lines),

                spoken=(

                    "Latest events for " + object_name + ": "

                    + join_names(latest) + "."

                )

            )

            return

        self.reply("I don't understand that yet.")

    def command_loop(self):

        while True:

            try:

                self.handle_command(input("\nRecallCam > "))

                if not self.running:

                    return

            except (KeyboardInterrupt, EOFError):

                return

            except Exception as e:

                print(f"Command error: {e}")

    # --------------------------------------------------------

    # MAIN CAMERA LOOP

    # --------------------------------------------------------

    def run(self):

        self.camera = cv2.VideoCapture(0)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)

        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not self.camera.isOpened():

            print("[ERROR] Could not open camera.")

            return

        print("RecallCam started.")

        print(f"Hand confidence: {HAND_CONFIDENCE}")

        print(f"Object confidence: {OBJECT_CONFIDENCE}")

        print("Press Q to quit, F to stop Find Mode, V to talk.")

        threading.Thread(

            target=self.command_loop, daemon=True

        ).start()

        if self.voice is not None:

            self.voice.print_status()

            if CONTINUOUS_LISTENING:

                self.voice.set_continuous(True)

            self.voice.say("RecallCam is ready.")

        start_time = time.time()

        try:

            while self.running:

                success, frame = self.camera.read()

                if not success:

                    continue

                frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.time() - start_time) * 1000)

                hand_boxes = self.detect_hands(frame, timestamp_ms)

                results = self.model.track(

                    frame,

                    persist=True,

                    conf=OBJECT_CONFIDENCE,

                    imgsz=480,

                    verbose=False

                )

                current_recall_ids = set()

                current_objects = []

                if results:

                    result = results[0]

                    if result.boxes is not None and len(result.boxes) > 0:

                        boxes = result.boxes

                        xyxy = boxes.xyxy.cpu().numpy()

                        classes = boxes.cls.cpu().numpy()

                        confidences = boxes.conf.cpu().numpy()

                        if boxes.id is not None:

                            tracker_ids = boxes.id.cpu().numpy().astype(int)

                        else:

                            tracker_ids = [-1] * len(xyxy)

                        # FIRST PASS: assign unique Recall IDs

                        for i, bbox in enumerate(xyxy):

                            object_name = normalize_object_name(

                                self.model.names[int(classes[i])]

                            )

                            if object_name in IGNORED_CLASSES:

                                continue

                            tracker_id = int(tracker_ids[i])

                            x1, y1, x2, y2 = map(int, bbox)

                            bbox_tuple = (x1, y1, x2, y2)

                            confirmed, stable_confidence = self.update_detection_reliability(
                                object_name, tracker_id, float(confidences[i]), bbox_tuple
                            )

                            if not confirmed:
                                current_objects.append({
                                    "object_name": object_name,
                                    "tracker_id": tracker_id,
                                    "recall_id": None,
                                    "confidence": stable_confidence,
                                    "bbox": bbox_tuple,
                                    "confirmed": False,
                                    "found": False,
                                    "find_score": 0.0
                                })
                                continue

                            recall_id = self.get_or_create_recall_id(
                                object_name, tracker_id, frame, bbox_tuple,
                                blocked_recall_ids=current_recall_ids
                            )

                            current_recall_ids.add(recall_id)

                            current_objects.append({
                                "object_name": object_name,
                                "tracker_id": tracker_id,
                                "recall_id": recall_id,
                                "confidence": stable_confidence,
                                "bbox": bbox_tuple,
                                "confirmed": True,
                                "found": False,
                                "find_score": 0.0
                            })

                # SECOND PASS: memory, holding, find, drawing

                any_found = False

                best_find_score = 0.0

                for obj in current_objects:

                    object_name = obj["object_name"]
                    recall_id = obj["recall_id"]
                    confidence = obj["confidence"]
                    bbox = obj["bbox"]
                    x1, y1, x2, y2 = bbox

                    if not obj.get("confirmed", False):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                        cv2.putText(
                            frame, f"VERIFYING {object_name} {confidence:.2f}",
                            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (160, 160, 160), 1
                        )
                        continue

                    detected_location = get_location((x1 + x2) / 2, frame.shape[1])
                    location = self.update_stable_location(recall_id, detected_location)
                    self.recall_last_seen[recall_id] = time.time()
                    self.recall_last_bbox[recall_id] = bbox

                    self.save_object_memory(
                        object_name, obj["tracker_id"], recall_id,
                        confidence, frame, bbox, location=location
                    )

                    self.update_holding(

                        object_name, recall_id, bbox, hand_boxes, frame

                    )

                    if self.find_mode:

                        obj["found"], obj["find_score"] = (

                            self.check_find_match(

                                object_name, frame, bbox, recall_id

                            )

                        )

                        if obj["found"]:

                            any_found = True

                        if object_name == self.find_target_name:

                            best_find_score = max(

                                best_find_score, obj["find_score"]

                            )

                    box_color = (0, 255, 0) if obj["found"] else (0, 200, 0)

                    cv2.rectangle(

                        frame, (x1, y1), (x2, y2), box_color, 2

                    )

                    cv2.putText(

                        frame,

                        f"{object_name} R{recall_id} {confidence:.2f}",

                        (x1, max(20, y1 - 8)),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2

                    )

                    cv2.putText(

                        frame, location, (x1, y2 + 20),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 1

                    )

                    if obj["found"]:

                        cv2.rectangle(

                            frame,

                            (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),

                            (0, 255, 255), 3

                        )

                # End holds for objects that left the frame

                self.expire_holds(current_recall_ids)

                # Draw hands

                for hx1, hy1, hx2, hy2 in hand_boxes:

                    cv2.rectangle(

                        frame, (hx1, hy1), (hx2, hy2), (255, 180, 0), 2

                    )

                    cv2.putText(

                        frame, "HAND", (hx1, max(20, hy1 - 8)),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2

                    )

                # Top bar

                cv2.rectangle(

                    frame, (0, 0), (frame.shape[1], 38), (20, 20, 20), -1

                )

                cv2.putText(

                    frame, "RecallCam", (12, 26),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2

                )

                cv2.putText(

                    frame, f"Hand {HAND_CONFIDENCE:.2f}", (145, 25),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1

                )

                cv2.putText(

                    frame, f"Object {OBJECT_CONFIDENCE:.2f}", (260, 25),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1

                )

                if self.voice is not None and self.voice.status_text():

                    cv2.putText(

                        frame, self.voice.status_text(), (380, 25),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2

                    )

                # Announce a find-mode hit out loud (once per sighting)

                if self.find_mode:

                    if any_found:

                        self.find_last_found = time.time()

                        if not self.find_announced:

                            self.find_announced = True

                            spot = next(

                                o for o in current_objects if o["found"]

                            )

                            where = get_location(

                                (spot["bbox"][0] + spot["bbox"][2]) / 2,

                                frame.shape[1]

                            ).lower()

                            self.reply(

                                f"Found your {self.find_target_name} "

                                f"on the {where}."

                            )

                    elif (

                        self.find_announced

                        and time.time() - self.find_last_found > 5.0

                    ):

                        self.find_announced = False

                # Find mode bar

                if self.find_mode:

                    target_text = self.find_target_name or ""

                    if self.find_target_id:

                        target_text += f" [R{self.find_target_id}]"

                    if any_found:

                        find_text = (

                            f"FINDING: {target_text} FOUND "

                            f"{best_find_score:.2f}"

                        )

                    else:

                        find_text = f"SEARCHING: {target_text}"

                    cv2.rectangle(

                        frame, (0, 40), (frame.shape[1], 72),

                        (30, 30, 30), -1

                    )

                    cv2.putText(

                        frame, find_text, (12, 62),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2

                    )

                cv2.imshow("RecallCam", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    break

                if key == ord("f"):

                    self.stop_find_mode()

                if key == ord("v") and self.voice is not None:

                    self.voice.listen_once()

        except KeyboardInterrupt:

            print("\nStopping RecallCam...")

        finally:

            if self.voice is not None:

                self.voice.stop()

            # Save any hold still in progress

            for recall_id in list(self.active_holding.keys()):

                self.end_hold(recall_id)

            if self.camera is not None:

                self.camera.release()

            if self.hand_landmarker is not None:

                try:

                    self.hand_landmarker.close()

                except Exception:

                    pass

            cv2.destroyAllWindows()

if __name__ == "__main__":

    RecallCam().run()