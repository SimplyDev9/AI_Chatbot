import json
import os

TRACKER_FILE = "sharepoint_tracker.json"


def load_tracker():
    if not os.path.exists(TRACKER_FILE):
        return {}

    with open(TRACKER_FILE, "r") as f:
        return json.load(f)


def save_tracker(data):
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)


def is_file_updated(file_id, last_modified):
    tracker = load_tracker()

    if file_id not in tracker:
        return True  # new file

    return tracker[file_id] != last_modified


def update_tracker(file_id, last_modified):
    tracker = load_tracker()
    tracker[file_id] = last_modified
    save_tracker(tracker)