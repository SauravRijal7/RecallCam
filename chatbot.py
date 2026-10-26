```python
import cv2
import json
import sqlite3
import threading
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from database import (
    create_database,
    save_observation,
    create_visual_fingerprint
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolo11s.pt"

CAMERA_INDEX = 0

MIN_CONFIDENCE = 0.60
YOLO_CONFIDENCE = 0.25

MATCH_THRESHOLD = 0.85

MEMORY_INTERVAL = 3

MEMORY_CLASSES = {
    "cup",
    "book",
    "cell phone",
    "bottle",
    "laptop",
    "mouse",
    "keyboard",
    "backpack",
    "umbrella",
}

DATABASE_PATH = Path("data/recallcam.db")
RECALL_FILE = Path("data/recall_objects.json")


# ============================================================
# GLOBALS
# ============================================================

chatbot_running = True
camera_running = True

memory_lock = threading.Lock()


# ============================================================
# RECALL MEMORY
# ============================================================

def load_memory():

    RECALL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not RECALL_FILE.exists():

        RECALL_FILE.write_text(
            "[]",
            encoding="utf-8"
        )

        return []

    try:

        with open(
            RECALL_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return []


def save_memory(memory):

    with open(
        RECALL_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            memory,
            file,
            indent=4
        )


memory = load_memory()


def get_next_recall_id():

    if not memory:
        return 1

    ids = []

    for item in memory:

        if item.get("recall_id") is not None:

            try:
                ids.append(
                    int(item["recall_id"])
                )
            except:
                pass

    if not ids:
        return 1

    return max(ids) + 1


def find_matching_memory(object_name, fingerprint):

    best_match = None
    best_score = -1

    for item in memory:

        if item.get("object_name") != object_name:
            continue

        old_fingerprint = item.get("fingerprint")

        if not old_fingerprint:
            continue

        old_hist = (
            __import__("numpy")
            .array(old_fingerprint, dtype="float32")
        )

        new_hist = (
            __import__("numpy")
            .array(fingerprint, dtype="float32")
        )

        score = cv2.compareHist(
            old_hist,
            new_hist,
            cv2.HISTCMP_CORREL
        )

        if score > best_score:

            best_score = score
            best_match = item

    if (
        best_match is not None
        and best_score >= MATCH_THRESHOLD
    ):

        return best_match, best_score

    return None, best_score


# ============================================================
# LOCATION
# ============================================================

def get_location(x_center, frame_width):

    if x_center < frame_width / 3:

        return "LEFT"

    elif x_center < (frame_width * 2 / 3):

        return "CENTER"

    else:

        return "RIGHT"


# ============================================================
# DATABASE HELPERS FOR CHATBOT
# ============================================================

def get_objects():

    if not DATABASE_PATH.exists():
        return []

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            object_name,
            recall_id,
            MAX(timestamp)
        FROM observations
        WHERE recall_id IS NOT NULL
          AND LOWER(object_name) != 'person'
        GROUP BY recall_id
        ORDER BY MAX(id) DESC
    """)

    results = cursor.fetchall()

    conn.close()

    return results


def get_last_seen(object_name):

    if not DATABASE_PATH.exists():
        return None

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            recall_id,
            timestamp,
            location,
            confidence
        FROM observations
        WHERE LOWER(object_name) = LOWER(?)
          AND recall_id IS NOT NULL
        ORDER BY id DESC
        LIMIT 1
    """, (object_name,))

    result = cursor.fetchone()

    conn.close()

    return result


# ============================================================
# OBJECT NAME UNDERSTANDING
# ============================================================

def normalize_object_name(question):

    question = question.lower()

    aliases = {

        "phone": "cell phone",
        "mobile": "cell phone",
        "mobile phone": "cell phone",
        "smartphone": "cell phone",
        "cellphone": "cell phone",

        "computer": "laptop",

        "bag": "backpack",

    }

    for word, object_name in aliases.items():

        if word in question:
            return object_name

    for object_name in MEMORY_CLASSES:

        if object_name in question:
            return object_name

    return None


# ============================================================
# FORGET EVERYTHING
# ============================================================

def forget_everything():

    global memory

    with memory_lock:

        memory.clear()

        save_memory(memory)

    if DATABASE_PATH.exists():

        conn = sqlite3.connect(
            DATABASE_PATH
        )

        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM observations"
        )

        conn.commit()
        conn.close()

    return (
        "Done. I've forgotten everything "
        "I've remembered."
    )


# ============================================================
# CHATBOT
# ============================================================

def answer_question(question):

    question_lower = (
        question
        .lower()
        .strip()
    )

    # --------------------------------------------------------
    # FORGET
    # --------------------------------------------------------

    forget_commands = [
        "forget everything",
        "forget all",
        "clear memory",
        "clear all memory",
        "delete memory",
        "erase memory",
        "erase everything",
        "start fresh",
        "start over",
        "reset memory",
    ]

    if any(
        command in question_lower
        for command in forget_commands
    ):

        return forget_everything()

    # --------------------------------------------------------
    # LIST OBJECTS
    # --------------------------------------------------------

    list_commands = [
        "what objects",
        "which objects",
        "what do you remember",
        "show objects",
        "list objects",
        "what have you seen",
        "what have u seen",
        "what did you see",
        "what did u see",
        "everything you have seen",
        "everything you've seen",
        "objects you have seen",
        "objects you've seen",
    ]

    if any(
        command in question_lower
        for command in list_commands
    ):

        objects = get_objects()

        if not objects:

            return (
                "I haven't remembered any objects yet."
            )

        unique_objects = {}

        for object_name, recall_id, timestamp in objects:

            if recall_id not in unique_objects:

                unique_objects[recall_id] = object_name

        response = "I currently remember:\n"

        for recall_id, object_name in unique_objects.items():

            response += (
                f"• {object_name} — "
                f"Recall ID {recall_id}\n"
            )

        return response.strip()

    # --------------------------------------------------------
    # OBJECT SEARCH
    # --------------------------------------------------------

    object_name = normalize_object_name(
        question
    )

    if object_name:

        result = get_last_seen(
            object_name
        )

        if result is None:

            return (
                f"I haven't seen your "
                f"{object_name} yet."
            )

        recall_id, timestamp, location, confidence = result

        location_text = (
            location.lower()
            if location
            else "an unknown location"
        )

        return (
            f"Your {object_name} was last seen "
            f"on the {location_text}.\n"
            f"Time: {timestamp}\n"
            f"Recall ID: {recall_id}"
        )

    # --------------------------------------------------------
    # HELP
    # --------------------------------------------------------

    if "help" in question_lower:

        return (
            "You can ask me:\n"
            "• Where is my book?\n"
            "• Where is my phone?\n"
            "• Where is my cup?\n"
            "• What have u seen?\n"
            "• What do you remember?\n"
            "• Forget everything\n"
            "• Clear memory\n"
            "• Exit"
        )

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return (
        "I don't understand that yet.\n"
        "Try: \"Where is my book?\" "
        "or \"What have you seen?\""
    )


# ============================================================
# CHAT THREAD
# ============================================================

def chatbot_loop():

    global chatbot_running

    print()
    print("=" * 55)
    print("              RECALLCAM CHATBOT")
    print("=" * 55)
    print("Camera is running.")
    print("Ask me about objects whenever you want.")
    print("Type 'help' for commands.")
    print("Type 'exit' to close RecallCam.")
    print()

    while chatbot_running:

        try:

            question = input("You: ").strip()

        except (EOFError, KeyboardInterrupt):

            chatbot_running = False
            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            chatbot_running = False

            print(
                "RecallCam: Goodbye!"
            )

            break

        answer = answer_question(
            question
        )

        print()
        print(
            "RecallCam:",
            answer
        )
        print()


# ============================================================
# CAMERA
# ============================================================

def camera_loop():

    global camera_running

    create_database()

    print("Loading YOLO...")

    model = YOLO(
        MODEL_PATH
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        camera_running = False
        return

    # Smaller resolution = smoother camera
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    last_memory_time = {}

    print(
        "Camera started."
    )

    while camera_running:

        success, frame = cap.read()

        if not success:

            print(
                "Could not read camera frame."
            )

            break

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            conf=YOLO_CONFIDENCE,
            imgsz=416,
            verbose=False
        )

        output = frame.copy()

        if (
            results
            and results[0].boxes is not None
        ):

            boxes = results[0].boxes

            for i in range(
                len(boxes)
            ):

                cls_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                object_name = (
                    model.names[cls_id]
                )

                # Don't remember people
                if object_name == "person":
                    continue

                # Only remember selected objects
                if object_name not in MEMORY_CLASSES:
                    continue

                if confidence < MIN_CONFIDENCE:
                    continue

                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                x1, y1, x2, y2 = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                x_center = (
                    x1 + x2
                ) // 2

                location = get_location(
                    x_center,
                    frame_width
                )

                # ------------------------------------------------
                # VISUAL FINGERPRINT
                # ------------------------------------------------

                fingerprint = (
                    create_visual_fingerprint(
                        frame,
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )
                )

                if fingerprint is None:
                    continue

                # ------------------------------------------------
                # RECALL ID
                # ------------------------------------------------

                with memory_lock:

                    match, score = (
                        find_matching_memory(
                            object_name,
                            fingerprint
                        )
                    )

                    if match:

                        recall_id = (
                            match["recall_id"]
                        )

                    else:

                        recall_id = (
                            get_next_recall_id()
                        )

                        memory.append({

                            "recall_id":
                                recall_id,

                            "object_name":
                                object_name,

                            "fingerprint":
                                fingerprint,

                            "created":
                                datetime.now().isoformat()
                        })

                        save_memory(
                            memory
                        )

                        print(
                            f"NEW MEMORY | "
                            f"{object_name} | "
                            f"Recall ID={recall_id}"
                        )

                # ------------------------------------------------
                # DRAW
                # ------------------------------------------------

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"{object_name} "
                    f"#{recall_id} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    output,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # SAVE OBSERVATION
                # ------------------------------------------------

                now = time.time()

                last_saved = (
                    last_memory_time.get(
                        recall_id,
                        0
                    )
                )

                if (
                    now - last_saved
                    >= MEMORY_INTERVAL
                ):

                    timestamp = (
                        datetime.now()
                        .strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )

                    save_observation(
                        object_name=object_name,
                        tracking_id=(
                            int(
                                boxes.id[i].item()
                            )
                            if boxes.id is not None
                            else None
                        ),
                        confidence=confidence,
                        timestamp=timestamp,
                        image_path="",
                        location=location,
                        recall_id=recall_id
                    )

                    last_memory_time[
                        recall_id
                    ] = now

                    print(
                        f"MEMORY SAVED | "
                        f"{object_name} | "
                        f"Recall ID={recall_id} | "
                        f"Location={location} | "
                        f"confidence={confidence:.2f}"
                    )

        # --------------------------------------------------------
        # CAMERA UI
        # --------------------------------------------------------

        cv2.putText(
            output,
            "RecallCam",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            "Q = Quit | Chat in terminal",
            (20, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "RecallCam",
            output
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            camera_running = False
            chatbot_running = False
            break

    cap.release()

    cv2.destroyAllWindows()

    camera_running = False
    chatbot_running = False


# ============================================================
# START RECALLCAM
# ============================================================

def main():

    print()
    print("=" * 55)
    print("                 RECALLCAM")
    print("=" * 55)
    print()
    print("Starting camera + memory + chatbot...")
    print()

    # Start chatbot in another thread
    chat_thread = threading.Thread(
        target=chatbot_loop,
        daemon=True
    )

    chat_thread.start()

    # Camera stays in main thread
    camera_loop()

    print()
    print("RecallCam stopped.")


if __name__ == "__main__":

    main()
```
