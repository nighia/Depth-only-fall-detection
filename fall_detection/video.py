import cv2


def video_fps(path, fallback=15.0):
    cap = cv2.VideoCapture(str(path))
    try:
        return cap.get(cv2.CAP_PROP_FPS) or fallback
    finally:
        cap.release()


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = ms / 1000.0 if ms > 0 else index / fps
            yield index, timestamp, frame
            index += 1
    finally:
        cap.release()
