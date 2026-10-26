import sqlite3
import os
import json
import time
import threading
from datetime import datetime

import cv2
import numpy as np


DB_PATH = os.path.join("data", "recallcam.db")

# One lock for every database operation in this process.
# The camera loop and the command-prompt thread both write,
# so without this they collide with "database is locked".
_DB_LOCK = threading.RLock()

LOCK_RETRIES = 8
LOCK_RETRY_DELAY = 0.25

def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _connect():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def execute(sql, params=(), fetch=None):
    """
    Run ONE statement safely.

    - holds the global lock
    - always commits/rolls back and closes the connection
    - retries if SQLite still reports the file as locked
    - fetch: None (no result), "one", or "all"
    """
    last_error = None

    for attempt in range(LOCK_RETRIES):
        with _DB_LOCK:
            conn = None
            try:
                conn = _connect()
                cursor = conn.cursor()
                cursor.execute(sql, params)

                if fetch == "one":
                    result = cursor.fetchone()
                elif fetch == "all":
                    result = cursor.fetchall()
                else:
                    result = None

                conn.commit()
                return result

            except sqlite3.OperationalError as e:
                last_error = e

                if conn is not None:
                    try:
                        conn.rollback()
                    except Exception:
                        pass

                if "locked" not in str(e).lower():
                    raise

            finally:
                if conn is not None:
                    conn.close()

        # Wait OUTSIDE the lock, then try again.
        time.sleep(LOCK_RETRY_DELAY * (attempt + 1))

    raise last_error


# ============================================================
# SCHEMA
# ============================================================

SCHEMAS = {
    "observations": {
        "create": """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_name TEXT,
                tracking_id INTEGER,
                confidence REAL,
                timestamp TEXT,
                image_path TEXT,
                location TEXT,
                recall_id INTEGER,
                fingerprint TEXT,
                bbox_width REAL,
                bbox_height REAL,
                center_x REAL,
                center_y REAL
            )
        """,
        "columns": {
            "object_name": "TEXT", "tracking_id": "INTEGER",
            "confidence": "REAL", "timestamp": "TEXT",
            "image_path": "TEXT", "location": "TEXT",
            "recall_id": "INTEGER", "fingerprint": "TEXT",
            "bbox_width": "REAL", "bbox_height": "REAL",
            "center_x": "REAL", "center_y": "REAL"
        }
    },
    "holding_events": {
        "create": """
            CREATE TABLE IF NOT EXISTS holding_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_name TEXT,
                recall_id INTEGER,
                started_at TEXT,
                ended_at TEXT,
                duration REAL,
                started_at_unix REAL,
                ended_at_unix REAL
            )
        """,
        "columns": {
            "object_name": "TEXT", "recall_id": "INTEGER",
            "started_at": "TEXT", "ended_at": "TEXT",
            "duration": "REAL", "started_at_unix": "REAL",
            "ended_at_unix": "REAL"
        }
    },
    "memory_events": {
        "create": """
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                object_name TEXT,
                recall_id INTEGER,
                location TEXT,
                timestamp TEXT,
                details TEXT,
                event_time_unix REAL
            )
        """,
        "columns": {
            "event_type": "TEXT", "object_name": "TEXT",
            "recall_id": "INTEGER", "location": "TEXT",
            "timestamp": "TEXT", "details": "TEXT",
            "event_time_unix": "REAL"
        }
    },
    "object_fingerprints": {
        "create": """
            CREATE TABLE IF NOT EXISTS object_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_name TEXT,
                recall_id INTEGER,
                fingerprint TEXT,
                timestamp TEXT,
                created_at TEXT
            )
        """,
        "columns": {
            "object_name": "TEXT", "recall_id": "INTEGER",
            "fingerprint": "TEXT", "timestamp": "TEXT",
            "created_at": "TEXT"
        }
    }
}


def add_missing_columns(cursor, table_name, columns):
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}

    for column, column_type in columns.items():
        if column not in existing:
            try:
                cursor.execute(
                    f"ALTER TABLE {table_name} "
                    f"ADD COLUMN {column} {column_type}"
                )
            except sqlite3.OperationalError as e:
                print(f"[DB] Could not add {table_name}.{column}: {e}")


def migrate_table(cursor, table_name):
    """
    Old databases may contain extra columns (often NOT NULL, like
    memory_events.event_time or object_fingerprints.created_at) that
    this code never fills in, which makes every INSERT fail.

    If a table has any column we don't know about, rebuild it with the
    clean schema and copy over every column we do know about.
    """
    schema = SCHEMAS[table_name]
    known = set(schema["columns"]) | {"id"}

    cursor.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}

    if not (existing - known):
        return

    print(
        f"[DB] Migrating {table_name} "
        f"(dropping legacy columns: {sorted(existing - known)})"
    )

    old_name = f"{table_name}_old"

    cursor.execute(f"DROP TABLE IF EXISTS {old_name}")
    cursor.execute(f"ALTER TABLE {table_name} RENAME TO {old_name}")
    cursor.execute(schema["create"])

    common = ["id"] + [
        c for c in schema["columns"] if c in existing
    ]

    if common:
        cols = ", ".join(common)
        cursor.execute(
            f"INSERT INTO {table_name} ({cols}) "
            f"SELECT {cols} FROM {old_name}"
        )

    # Rebuild event_time_unix from legacy text columns if it was missing.
    if table_name == "memory_events":
        legacy = [c for c in ("event_time", "timestamp") if c in existing]

        if legacy:
            source = ", ".join(f"o.{c}" for c in legacy)
            cursor.execute(f"""
                UPDATE memory_events
                SET event_time_unix = (
                    SELECT CAST(
                        strftime('%s', COALESCE({source}), 'utc') AS REAL
                    )
                    FROM {old_name} o
                    WHERE o.id = memory_events.id
                )
                WHERE event_time_unix IS NULL
            """)

    cursor.execute(f"DROP TABLE {old_name}")


def create_database():

    last_error = None

    for attempt in range(LOCK_RETRIES):
        with _DB_LOCK:
            conn = None
            try:
                conn = _connect()
                cursor = conn.cursor()
                _create_schema(cursor)
                conn.commit()
                return

            except sqlite3.OperationalError as e:
                last_error = e

                if conn is not None:
                    try:
                        conn.rollback()
                    except Exception:
                        pass

                if "locked" not in str(e).lower():
                    raise

            finally:
                if conn is not None:
                    conn.close()

        print(
            "[DB] Database is locked. Is another RecallCam / python "
            "process or DB Browser still open? Retrying..."
        )
        time.sleep(LOCK_RETRY_DELAY * (attempt + 1))

    raise last_error


def _create_schema(cursor):

    for table_name, schema in SCHEMAS.items():
        cursor.execute(schema["create"])
        migrate_table(cursor, table_name)
        add_missing_columns(cursor, table_name, schema["columns"])

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_obs_name "
        "ON observations(object_name)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_obs_recall "
        "ON observations(recall_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_fp_recall "
        "ON object_fingerprints(recall_id)"
    )


# ============================================================
# OBSERVATIONS
# ============================================================

def save_observation(
    object_name, tracking_id, confidence, timestamp, image_path,
    location, recall_id, fingerprint=None, bbox_width=0,
    bbox_height=0, center_x=0, center_y=0
):
    execute("""
        INSERT INTO observations (
            object_name, tracking_id, confidence, timestamp,
            image_path, location, recall_id, fingerprint,
            bbox_width, bbox_height, center_x, center_y
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        object_name, tracking_id, confidence, timestamp,
        image_path, location, recall_id,
        json.dumps(fingerprint) if fingerprint is not None else None,
        bbox_width, bbox_height, center_x, center_y
    ))


def get_last_seen(object_name):
    return execute("""
        SELECT object_name, recall_id, location, timestamp, image_path
        FROM observations
        WHERE object_name = ?
        ORDER BY id DESC
        LIMIT 1
    """, (object_name,), fetch="one")


def get_all_object_memories(object_name=None):
    """
    One row per Recall ID (its most recent observation):
    recall_id, object_name, location, bbox_width, bbox_height,
    center_x, center_y, fingerprint, timestamp
    """
    sql = """
        SELECT recall_id, object_name, location, bbox_width,
               bbox_height, center_x, center_y, fingerprint, timestamp
        FROM observations
        WHERE id IN (
            SELECT MAX(id) FROM observations
            WHERE recall_id IS NOT NULL
            {name_filter}
            GROUP BY recall_id
        )
        ORDER BY id DESC
    """

    if object_name:
        return execute(
            sql.format(name_filter="AND object_name = ?"),
            (object_name,), fetch="all"
        )

    return execute(sql.format(name_filter=""), (), fetch="all")


def get_memory_by_recall_id(recall_id):
    return execute("""
        SELECT object_name, recall_id, location, timestamp, image_path
        FROM observations
        WHERE recall_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (recall_id,), fetch="one")


def get_highest_recall_id():
    best = 0

    for table in ("observations", "object_fingerprints", "holding_events"):
        row = execute(
            f"SELECT MAX(recall_id) FROM {table} "
            f"WHERE recall_id IS NOT NULL",
            (), fetch="one"
        )
        best = max(best, (row[0] if row else 0) or 0)

    return best


# ============================================================
# FINGERPRINTS
# ============================================================

def save_fingerprint(object_name, recall_id, fingerprint):
    if fingerprint is None:
        return

    stamp = now_text()

    execute("""
        INSERT INTO object_fingerprints (
            object_name, recall_id, fingerprint, timestamp, created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        object_name, recall_id, json.dumps(fingerprint), stamp, stamp
    ))


def prune_fingerprints(recall_id, keep=20):
    execute("""
        DELETE FROM object_fingerprints
        WHERE recall_id = ?
        AND id NOT IN (
            SELECT id FROM object_fingerprints
            WHERE recall_id = ?
            ORDER BY id DESC
            LIMIT ?
        )
    """, (recall_id, recall_id, keep))


def get_fingerprints(recall_id, limit=20):
    rows = execute("""
        SELECT fingerprint
        FROM object_fingerprints
        WHERE recall_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (recall_id, limit), fetch="all")

    fingerprints = []

    for row in rows or []:
        try:
            fingerprints.append(json.loads(row[0]))
        except Exception:
            pass

    return fingerprints


# ============================================================
# HOLDING EVENTS
# ============================================================

def save_holding_event(
    object_name, recall_id, started_at, ended_at,
    duration, started_at_unix, ended_at_unix
):
    execute("""
        INSERT INTO holding_events (
            object_name, recall_id, started_at, ended_at,
            duration, started_at_unix, ended_at_unix
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        object_name, recall_id, started_at, ended_at,
        duration, started_at_unix, ended_at_unix
    ))


def get_last_holding():
    return execute("""
        SELECT id, object_name, recall_id, started_at, ended_at,
               duration, started_at_unix, ended_at_unix
        FROM holding_events
        WHERE ended_at_unix IS NOT NULL
        ORDER BY ended_at_unix DESC, id DESC
        LIMIT 1
    """, (), fetch="one")


def get_holding_at_time(target_time, tolerance=15):
    return execute("""
        SELECT id, object_name, recall_id, started_at, ended_at,
               duration, started_at_unix, ended_at_unix
        FROM holding_events
        WHERE started_at_unix <= ?
        AND ended_at_unix >= ?
        ORDER BY id DESC
        LIMIT 1
    """, (target_time + tolerance, target_time - tolerance), fetch="one")


# ============================================================
# MEMORY EVENTS
# ============================================================

def save_memory_event(
    event_type, object_name, recall_id=None, location=None,
    timestamp=None, details=None, event_time_unix=None
):
    if timestamp is None:
        timestamp = now_text()

    if event_time_unix is None:
        event_time_unix = time.time()

    execute("""
        INSERT INTO memory_events (
            event_type, object_name, recall_id, location,
            timestamp, details, event_time_unix
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        event_type, object_name, recall_id, location,
        timestamp, details, event_time_unix
    ))


def get_recent_events(seconds=60):
    return execute("""
        SELECT event_type, object_name, recall_id, location,
               timestamp, details, event_time_unix
        FROM memory_events
        WHERE event_time_unix >= ?
        ORDER BY event_time_unix DESC
    """, (time.time() - seconds,), fetch="all")


def get_events_for_object(object_name):
    return execute("""
        SELECT event_type, object_name, recall_id, location,
               timestamp, details, event_time_unix
        FROM memory_events
        WHERE object_name = ?
        ORDER BY event_time_unix DESC
    """, (object_name,), fetch="all")


# ============================================================
# CLEAR MEMORY
# ============================================================

def clear_all_memory():
    for table in (
        "observations", "holding_events",
        "memory_events", "object_fingerprints"
    ):
        execute(f"DELETE FROM {table}")

    try:
        execute("""
            DELETE FROM sqlite_sequence
            WHERE name IN (
                'observations', 'holding_events',
                'memory_events', 'object_fingerprints'
            )
        """)
    except sqlite3.OperationalError:
        pass


def clear_memory_events():
    execute("DELETE FROM memory_events")


# ============================================================
# VISUAL FINGERPRINT
# ============================================================

def create_visual_fingerprint(frame, bbox):
    """
    Layout (274 floats):
      [0:256]   HSV histogram (16x16)
      [256:272] 4x4 edge map
      [272]     aspect ratio
      [273]     mean edge density
    """
    try:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (64, 64))

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]
        )
        histogram = cv2.normalize(histogram, histogram).flatten()

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)

        edge_small = cv2.resize(edges, (4, 4)).astype(np.float32)
        edge_small /= 255.0

        aspect_ratio = max(1, x2 - x1) / max(1, y2 - y1)
        edge_mean = float(np.mean(edges) / 255.0)

        fingerprint = np.concatenate([
            histogram.astype(np.float32),
            edge_small.flatten(),
            np.array([aspect_ratio, edge_mean], dtype=np.float32)
        ])

        return fingerprint.tolist()

    except Exception:
        return None