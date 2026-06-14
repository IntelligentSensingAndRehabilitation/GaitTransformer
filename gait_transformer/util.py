import cv2
import numpy as np
from tqdm import tqdm

import cv2
import numpy as np
from tqdm import tqdm


def video_reader(filename: str, batch_size: int = 8, width: int | None = None):
    """
    Read a video file and yield frames in batches.

    In theory, tensorflow_io has tools for this but they don't seem to work for me. That
    is probably more efficient if it works as they can prefetch. This also will optionally
    downsample the video if compute is a limit.

    Args:
        filename: (str) The path to the video file.
        batch_size: (int) The number of frames to yield at once.
        width: (int | None) The width to downsample to. If None, the original width is used.

    Returns:
        A generator that yields batches
    """

    cap = cv2.VideoCapture(filename)

    # 1. Disable OpenCV's auto-rotation (can be buggy and cause double-rotations)
    # The flag CAP_PROP_ORIENTATION_AUTO has value 49.
    auto_prop = getattr(cv2, 'CAP_PROP_ORIENTATION_AUTO', 49)
    try:
        cap.set(auto_prop, 0)
    except Exception:
        pass  # Fails gracefully on older OpenCV versions
        
    # 2. Get the actual rotation metadata
    # The flag CAP_PROP_ORIENTATION_META has value 48.
    meta_prop = getattr(cv2, 'CAP_PROP_ORIENTATION_META', 48)
    try:
        orientation = int(cap.get(meta_prop))
    except Exception:
        orientation = 0

    frames = []
    while True:

        ret, frame = cap.read()

        if ret is False:

            if len(frames) > 0:
                frames = np.array(frames)
                yield frames

            cap.release()
            return

        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply rotation based on the extracted metadata
            if orientation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif orientation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if width is not None:
                # downsample to keep the aspect ratio and output the specified width
                # frame.shape[1] and [0] accurately reflect the dimensions *after* rotation
                scale = width / frame.shape[1]
                height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (width, height))

            frames.append(frame)

            if len(frames) >= batch_size:
                frames = np.array(frames)
                yield frames

                frames = []
                
def get_dt_from_filename(filename: str) -> datetime | None:
    """
    Attempt to extract a datetime object from a video filename.
    Supports common formats like 'VID_20231027_153022.mp4'.
    Falls back to file modification time if no date is found in the name.
    """
    basename = os.path.basename(filename)
    
    # Look for YYYYMMDD_HHMMSS pattern (common in Android/some cameras)
    match = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', basename)
    if match:
        try:
            return datetime.strptime(match.group(), "%Y%m%d_%H%M%S")
        except ValueError:
            pass
            
    # Look for YYYY-MM-DD_HH-MM-SS or similar patterns
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})[_-](\d{2})[-:](\d{2})[-:](\d{2})', basename)
    if match:
        # Reconstruct to standard format for easy parsing
        dt_str = "".join(match.groups())
        try:
            return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        except ValueError:
            pass

    # Fallback to filesystem metadata
    if os.path.exists(filename):
        # Modification time is usually the most reliable fallback for "when it was taken"
        timestamp = os.path.getmtime(filename) 
        return datetime.fromtimestamp(timestamp)
        
    return None
