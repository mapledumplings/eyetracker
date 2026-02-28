import argparse
import ctypes
import sys
import time
from dataclasses import dataclass
from ctypes import wintypes

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pygame


# FaceMesh landmark indices for iris + eye corners.
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)


@dataclass
class Point:
    x: float
    y: float


def create_overlay_window(screen_w, screen_h, windowed=False):
    if windowed:
        return pygame.display.set_mode((min(1100, screen_w), min(700, screen_h)))
    # Use borderless window for transparent overlays; true FULLSCREEN won't colorkey reliably.
    if sys.platform == "win32":
        ctypes.windll.user32.SetProcessDPIAware()
    return pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME)


def enable_windows_transparency(screen_w, screen_h):
    if sys.platform != "win32":
        return

    wm_info = pygame.display.get_wm_info()
    hwnd = wm_info.get("window")
    if not hwnd:
        return

    user32 = ctypes.windll.user32

    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TOPMOST = 0x00000008
    LWA_COLORKEY = 0x00000001
    HWND_TOPMOST = -1
    SWP_NOMOVE = 0x0002
    SWP_NOACTIVATE = 0x0010
    SWP_SHOWWINDOW = 0x0040

    try:
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.restype = ctypes.c_long
        user32.SetLayeredWindowAttributes.argtypes = [
            wintypes.HWND,
            wintypes.COLORREF,
            wintypes.BYTE,
            wintypes.DWORD,
        ]
        user32.SetLayeredWindowAttributes.restype = wintypes.BOOL

        ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED | WS_EX_TOPMOST)
        # COLORREF macro replacement: 0x00bbggrr
        colorkey = wintypes.COLORREF(0x00000000)
        ok = user32.SetLayeredWindowAttributes(hwnd, colorkey, 0, LWA_COLORKEY)
        if not ok:
            raise ctypes.WinError()
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            screen_w,
            screen_h,
            SWP_NOMOVE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )
    except Exception as exc:
        print(f"[WARN] Transparency setup failed, continuing without it: {exc}")


def mean_landmark(landmarks, indices):
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    return Point(float(np.mean(xs)), float(np.mean(ys)))


def normalize_iris(landmarks):
    left_iris = mean_landmark(landmarks, LEFT_IRIS)
    right_iris = mean_landmark(landmarks, RIGHT_IRIS)

    def eye_coords(iris, c0, c1):
        # Use only eye-corner geometry so eyelid opening changes have less effect.
        v_x = c1.x - c0.x
        v_y = c1.y - c0.y
        w2 = v_x * v_x + v_y * v_y
        if w2 < 1e-10:
            return 0.5, 0.5

        # Position along corner line: 0 at c0, 1 at c1.
        rel_x = ((iris.x - c0.x) * v_x + (iris.y - c0.y) * v_y) / w2

        # Position perpendicular to corner line, normalized by eye width.
        width = np.sqrt(w2)
        perp_x = -v_y / width
        perp_y = v_x / width
        center_x = (c0.x + c1.x) * 0.5
        center_y = (c0.y + c1.y) * 0.5
        rel_y = ((iris.x - center_x) * perp_x + (iris.y - center_y) * perp_y) / width + 0.5
        return rel_x, rel_y

    l0 = landmarks[LEFT_EYE_CORNERS[0]]
    l1 = landmarks[LEFT_EYE_CORNERS[1]]
    r0 = landmarks[RIGHT_EYE_CORNERS[0]]
    r1 = landmarks[RIGHT_EYE_CORNERS[1]]

    left_x, left_y = eye_coords(left_iris, l0, l1)
    right_x, right_y = eye_coords(right_iris, r0, r1)
    return Point((left_x + right_x) * 0.5, (left_y + right_y) * 0.5)


def eye_open_ratio(landmarks):
    # Blink/squint detection ratio (height / width).
    l_top, l_bottom = landmarks[159], landmarks[145]
    r_top, r_bottom = landmarks[386], landmarks[374]
    l0, l1 = landmarks[LEFT_EYE_CORNERS[0]], landmarks[LEFT_EYE_CORNERS[1]]
    r0, r1 = landmarks[RIGHT_EYE_CORNERS[0]], landmarks[RIGHT_EYE_CORNERS[1]]

    left_h = np.hypot(l_top.x - l_bottom.x, l_top.y - l_bottom.y)
    right_h = np.hypot(r_top.x - r_bottom.x, r_top.y - r_bottom.y)
    left_w = max(np.hypot(l0.x - l1.x, l0.y - l1.y), 1e-6)
    right_w = max(np.hypot(r0.x - r1.x, r0.y - r1.y), 1e-6)
    return float((left_h / left_w + right_h / right_w) * 0.5)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def remap(v, in_min, in_max, out_min, out_max):
    if abs(in_max - in_min) < 1e-6:
        return (out_min + out_max) * 0.5
    t = (v - in_min) / (in_max - in_min)
    t = clamp(t, 0.0, 1.0)
    return out_min + t * (out_max - out_min)


def create_face_mesh():
    # Support both MediaPipe module layouts seen across versions/builds.
    face_mesh_mod = None
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        face_mesh_mod = mp.solutions.face_mesh
    else:
        try:
            from mediapipe.python.solutions import face_mesh as face_mesh_mod  # type: ignore
        except Exception:
            face_mesh_mod = None

    if face_mesh_mod is None:
        raise RuntimeError(
            "Could not find FaceMesh in this MediaPipe install. "
            "Try: pip uninstall mediapipe -y && pip install mediapipe==0.10.14"
        )

    return face_mesh_mod.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def open_camera(camera_index):
    # On Windows, MSMF can hang on some webcam drivers; try DSHOW first.
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("DirectShow", cv2.CAP_DSHOW))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(("MSMF", cv2.CAP_MSMF))
    backends.append(("Default", cv2.CAP_ANY))

    for name, backend in backends:
        print(f"[INFO] Trying camera {camera_index} with backend: {name}")
        cap = cv2.VideoCapture(camera_index, backend)
        if cap is not None and cap.isOpened():
            print(f"[INFO] Camera opened with backend: {name}")
            return cap
        if cap is not None:
            cap.release()

    return None


def read_first_frame(cap, attempts=60, delay_sec=0.03):
    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        time.sleep(delay_sec)
    return None


def run_calibration(cap, face_mesh, screen_w, screen_h, windowed=False):
    # 9-point calibration grid.
    points = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
    ]
    settle_seconds = 1.5
    collect_seconds = 2.0

    samples = []

    if not pygame.get_init():
        pygame.init()
    if windowed:
        overlay = pygame.display.set_mode((min(1100, screen_w), min(700, screen_h)))
    else:
        overlay = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
    pygame.display.set_caption("Eye Calibration")
    font = pygame.font.SysFont("Segoe UI", 34)
    clock = pygame.time.Clock()

    for px, py in points:
        # Extra settle time between points.
        settle_start = time.time()
        while time.time() - settle_start < settle_seconds:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.display.quit()
                    return None

            overlay.fill((20, 20, 20))
            ow, oh = overlay.get_size()
            cx, cy = int(px * ow), int(py * oh)
            pygame.draw.circle(overlay, (255, 255, 255), (cx, cy), 28, 3)
            text = font.render("Look at the circle", True, (230, 230, 230))
            overlay.blit(text, (40, 40))
            pygame.display.flip()
            clock.tick(60)

        # Collect averaged samples for this calibration point.
        collected = []
        start_collect = time.time()
        while time.time() - start_collect < collect_seconds:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                if eye_open_ratio(lm) > 0.16:
                    norm = normalize_iris(lm)
                    collected.append((norm.x, norm.y))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.display.quit()
                    return None

            overlay.fill((20, 20, 20))
            ow, oh = overlay.get_size()
            cx, cy = int(px * ow), int(py * oh)
            pygame.draw.circle(overlay, (255, 255, 255), (cx, cy), 28, 3)
            text = font.render("Hold your gaze...", True, (230, 230, 230))
            overlay.blit(text, (40, 40))
            pygame.display.flip()
            clock.tick(60)

        if not collected:
            pygame.display.quit()
            return None

        mean_x = float(np.mean([c[0] for c in collected]))
        mean_y = float(np.mean([c[1] for c in collected]))
        samples.append((mean_x, mean_y, px, py))

    in_x = [s[0] for s in samples]
    in_y = [s[1] for s in samples]

    # Axis-aligned map from calibration envelope.
    calib = {
        "x_min": min(in_x),
        "x_max": max(in_x),
        "y_min": min(in_y),
        "y_max": max(in_y),
    }
    return calib


def main():
    parser = argparse.ArgumentParser(description="Eye gaze circle tracker")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--radius", type=int, default=26, help="Circle radius")
    parser.add_argument("--alpha", type=float, default=0.3, help="Smoothing factor (0-1)")
    parser.add_argument("--windowed", action="store_true", help="Run in windowed mode instead of fullscreen")
    parser.add_argument("--no-calibration", action="store_true", help="Skip calibration and use default mapping")
    parser.add_argument("--dot-only", action="store_true", help="Show circle window only (no eye tracking)")
    args = parser.parse_args()

    print("[INFO] Starting eye gaze tracker...")
    screen_w, screen_h = pyautogui.size()
    print(f"[INFO] Screen size: {screen_w}x{screen_h}")
    print("[INFO] Opening webcam...")
    cap = open_camera(args.camera)
    if cap is None:
        print(f"[ERROR] Could not open webcam index {args.camera}. Try --camera 1 or close other camera apps.")
        sys.exit(1)

    print("[INFO] Waiting for first camera frame...")
    first_frame = read_first_frame(cap)
    if first_frame is None:
        cap.release()
        print("[ERROR] Camera opened but no frames were received.")
        sys.exit(1)
    fh, fw = first_frame.shape[:2]
    print(f"[INFO] First frame received: {fw}x{fh}")

    pygame.init()
    print("[INFO] Creating display window...")
    screen = create_overlay_window(screen_w, screen_h, windowed=args.windowed)
    pygame.display.set_caption("Gaze Circle")
    ww, wh = screen.get_size()
    enable_windows_transparency(ww, wh)
    pygame.event.clear()
    print("[INFO] Display window created.")

    face_mesh = None
    if args.dot_only:
        print("[INFO] Dot-only mode enabled (no tracking). Press ESC to quit.")
        calib = {"x_min": 0.25, "x_max": 0.75, "y_min": 0.25, "y_max": 0.75}
    else:
        print("[INFO] Initializing MediaPipe FaceMesh...")
        try:
            face_mesh = create_face_mesh()
        except Exception as exc:
            cap.release()
            pygame.quit()
            print(f"[ERROR] MediaPipe initialization failed: {exc}")
            sys.exit(1)

        if args.no_calibration:
            print("[INFO] Skipping calibration.")
            calib = {"x_min": 0.25, "x_max": 0.75, "y_min": 0.25, "y_max": 0.75}
        else:
            print("[INFO] Calibration starting... Press ESC any time to quit.")
            calib = run_calibration(cap, face_mesh, screen_w, screen_h, windowed=args.windowed)
            if calib is None:
                cap.release()
                face_mesh.close()
                print("[ERROR] Calibration cancelled or failed.")
                sys.exit(1)
            print("[INFO] Calibration complete.")
            screen = create_overlay_window(screen_w, screen_h, windowed=args.windowed)
            pygame.display.set_caption("Gaze Circle")
            ww, wh = screen.get_size()
            enable_windows_transparency(ww, wh)
            pygame.event.clear()

        print("[INFO] Tracking started. Press ESC to quit.")
    clock = pygame.time.Clock()

    dot_x, dot_y = screen_w // 2, screen_h // 2
    alpha = clamp(args.alpha, 0.01, 1.0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                if face_mesh is not None:
                    face_mesh.close()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                cap.release()
                if face_mesh is not None:
                    face_mesh.close()
                pygame.quit()
                return

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if face_mesh is None:
            result = None
        else:
            result = face_mesh.process(rgb)

        if result and result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            if eye_open_ratio(lm) <= 0.16:
                norm = None
            else:
                norm = normalize_iris(lm)

            if norm is not None:
                gx = remap(norm.x, calib["x_min"], calib["x_max"], 0, screen_w)
                gy = remap(norm.y, calib["y_min"], calib["y_max"], 0, screen_h)

                dot_x = int((1 - alpha) * dot_x + alpha * gx)
                dot_y = int((1 - alpha) * dot_y + alpha * gy)

        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (255, 60, 60), (dot_x, dot_y), args.radius, 4)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
