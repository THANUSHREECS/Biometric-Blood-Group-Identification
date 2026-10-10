import os
import uuid
import shutil

def save_anonymous_capture(source_path, dest_dir="data/anonymous_captures"):
    """
    Saves a raw live capture with an anonymous ID for later analysis.
    """
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(source_path):
        anon_id = str(uuid.uuid4())
        dest_path = os.path.join(dest_dir, f"{anon_id}.bmp")
        shutil.copy2(source_path, dest_path)
        return dest_path
    return None

if __name__ == "__main__":
    print("Example usage of save_anonymous_capture")
