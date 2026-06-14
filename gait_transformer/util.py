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
                
def get_dt_from_filename(filename: str) -> float:
    """
    Return the frame time step dt (in seconds) for a video file, i.e. 1/fps.

    Args:
        filename: path to the video file.

    Returns:
        dt as a float in seconds (e.g. ~0.0333 for a 30 Hz video).

    Raises:
        ValueError: If the file cannot be opened or its frame rate cannot be read.
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {filename!r}.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Could not read a valid frame rate from {filename!r} (got {fps}).")
    return 1.0 / fps
