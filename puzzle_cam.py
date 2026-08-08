#!/usr/bin/env python3
"""
PuzzleCam (Python): hand-frame capture + 3x3 puzzle.

Same flow as the browser PuzzleCam:
  both hands pinch -> frame -> countdown -> photo -> puzzle -> pinch to drag -> fist to save

Run:
  conda activate home
  python puzzle_cam.py

Keys:
  q  quit
  r  reset puzzle (or reset all if strip full)
  d  download photo strip (when 3 saves ready)
  m  toggle mirror preview
"""

from __future__ import annotations

import argparse
import math
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
OUTPUT_DIR = ROOT / "output"

GRID = 3
PINCH_THRESHOLD = 0.055
FRAME_PADDING = 28
FREEZE_HOLD_MS = 250
COUNTDOWN_SECONDS = 3
FIST_HOLD_FRAMES = 12
SNAP_DISTANCE_RATIO = 0.45
FRAME_GRACE_MS = 450
MAX_STRIP = 3

PHOTO_CONTRAST = 1.3
PHOTO_BRIGHTNESS = 10
PHOTO_NOISE_STD = 15

LM_WRIST = 0
LM_THUMB_TIP = 4
LM_INDEX_MCP = 5
LM_INDEX_TIP = 8
LM_MIDDLE_MCP = 9
LM_MIDDLE_TIP = 12
LM_RING_MCP = 13
LM_RING_TIP = 16
LM_PINKY_MCP = 17
LM_PINKY_TIP = 20

FINGER_PAIRS = [
    (LM_INDEX_TIP, LM_INDEX_MCP),
    (LM_MIDDLE_TIP, LM_MIDDLE_MCP),
    (LM_RING_TIP, LM_RING_MCP),
    (LM_PINKY_TIP, LM_PINKY_MCP),
]

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
SKELETON_COLOR = (255, 255, 255)
FRAME_COLOR = (245, 197, 24)
SOLVED_COLOR = (95, 174, 110)


@dataclass
class Box:
    x: int
    y: int
    width: int
    height: int


@dataclass
class Piece:
    row: int
    col: int
    img: np.ndarray
    x: float
    y: float
    w: int
    h: int
    placed: bool = False
    dragging: bool = False


def ensure_model() -> Path:
    if MODEL_PATH.is_file():
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def create_landmarker() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def lm_dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def is_pinching(landmarks) -> bool:
    return lm_dist(landmarks[LM_THUMB_TIP], landmarks[LM_INDEX_TIP]) < PINCH_THRESHOLD


def is_fist(landmarks) -> bool:
    wrist = landmarks[LM_WRIST]
    curled = 0
    for tip_i, mcp_i in FINGER_PAIRS:
        if lm_dist(landmarks[tip_i], wrist) < lm_dist(landmarks[mcp_i], wrist):
            curled += 1
    return curled >= 4


def lm_to_px(lm, width: int, height: int, mirror: bool) -> Tuple[float, float]:
    x = (1.0 - lm.x if mirror else lm.x) * width
    y = lm.y * height
    return x, y


def compute_hand_frame(
    index_a: Tuple[float, float],
    index_b: Tuple[float, float],
    width: int,
    height: int,
) -> Optional[Box]:
    min_x = min(index_a[0], index_b[0]) - FRAME_PADDING
    max_x = max(index_a[0], index_b[0]) + FRAME_PADDING
    min_y = min(index_a[1], index_b[1]) - FRAME_PADDING
    max_y = max(index_a[1], index_b[1]) + FRAME_PADDING
    x = max(0, int(min_x))
    y = max(0, int(min_y))
    w = min(width, int(max_x)) - x
    h = min(height, int(max_y)) - y
    if w <= 4 or h <= 4:
        return None
    return Box(x, y, w, h)


def apply_photobooth(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    adjusted = cv2.convertScaleAbs(gray, alpha=PHOTO_CONTRAST, beta=PHOTO_BRIGHTNESS)
    noise = np.random.normal(0, PHOTO_NOISE_STD, adjusted.shape).astype(np.float32)
    noisy = np.clip(adjusted.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)


def draw_hand_skeleton(frame: np.ndarray, landmarks, mirror: bool) -> None:
    h, w = frame.shape[:2]
    pts = [lm_to_px(lm, w, h, mirror) for lm in landmarks]
    int_pts = [(int(x), int(y)) for x, y in pts]
    for conn in HAND_CONNECTIONS:
        cv2.line(frame, int_pts[conn.start], int_pts[conn.end], SKELETON_COLOR, 2, cv2.LINE_AA)
    for x, y in int_pts:
        cv2.circle(frame, (x, y), 3, SKELETON_COLOR, -1, cv2.LINE_AA)


def draw_status_bar(frame: np.ndarray, lines: List[str], gallery_count: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 72), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines[:3]):
        cv2.putText(
            frame,
            line,
            (12, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"Strip: {gallery_count}/{MAX_STRIP}",
        (frame.shape[1] - 140, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def draw_frame_box(frame: np.ndarray, box: Box, color: Tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (box.x, box.y), (box.x + box.width, box.y + box.height), color, 2)


def apply_bw_region(frame: np.ndarray, box: Box) -> None:
    x, y, w, h = box.x, box.y, box.width, box.height
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    adj = cv2.convertScaleAbs(gray, alpha=PHOTO_CONTRAST, beta=PHOTO_BRIGHTNESS)
    frame[y:y + h, x:x + w] = cv2.cvtColor(adj, cv2.COLOR_GRAY2BGR)


def cell_center(box: Box, row: int, col: int, tile_w: int, tile_h: int) -> Tuple[float, float]:
    return box.x + col * tile_w, box.y + row * tile_h


def is_near_own_cell(piece: Piece, box: Box, tile_w: int, tile_h: int) -> bool:
    cx, cy = cell_center(box, piece.row, piece.col, tile_w, tile_h)
    tol = min(tile_w, tile_h) * SNAP_DISTANCE_RATIO
    return math.hypot(piece.x - cx, piece.y - cy) < tol


def snap_piece(piece: Piece, box: Box, tile_w: int, tile_h: int) -> None:
    piece.x = float(box.x + piece.col * tile_w)
    piece.y = float(box.y + piece.row * tile_h)
    piece.placed = True


def displace_at_cell(
    piece: Piece,
    target_row: int,
    target_col: int,
    box: Box,
    tile_w: int,
    tile_h: int,
    pieces: List[Piece],
) -> None:
    cell_x = box.x + target_col * tile_w
    cell_y = box.y + target_row * tile_h
    occupant = None
    for p in pieces:
        if p is piece:
            continue
        cx = p.x + p.w / 2
        cy = p.y + p.h / 2
        if cell_x <= cx < cell_x + tile_w and cell_y <= cy < cell_y + tile_h:
            occupant = p
            break
    if occupant is None:
        return
    if occupant.row == target_row and occupant.col == target_col and occupant.placed:
        return
    occupant.placed = False
    free = []
    for row in range(GRID):
        for col in range(GRID):
            if row == target_row and col == target_col:
                continue
            cx0 = box.x + col * tile_w
            cy0 = box.y + row * tile_h
            taken = any(
                p is not occupant
                and p is not piece
                and cx0 <= p.x + p.w / 2 < cx0 + tile_w
                and cy0 <= p.y + p.h / 2 < cy0 + tile_h
                for p in pieces
            )
            if not taken:
                free.append((row, col))
    if free:
        row, col = random.choice(free)
    else:
        row, col = occupant.row, occupant.col
    jitter_x = (random.random() - 0.5) * tile_w * 0.5
    jitter_y = (random.random() - 0.5) * tile_h * 0.5
    occupant.x = box.x + col * tile_w + jitter_x
    occupant.y = box.y + row * tile_h + jitter_y


def find_nearest_piece(px: float, py: float, pieces: List[Piece]) -> Optional[Piece]:
    best = None
    best_d = 1e9
    for p in pieces:
        cx = p.x + p.w / 2
        cy = p.y + p.h / 2
        d = math.hypot(px - cx, py - cy)
        if d < max(p.w, p.h) * 0.6 and d < best_d:
            best_d = d
            best = p
    return best


def reconcile_placed(pieces: List[Piece], box: Box, tile_w: int, tile_h: int) -> bool:
    for p in pieces:
        if not p.dragging:
            p.placed = is_near_own_cell(p, box, tile_w, tile_h)
    return all(p.placed for p in pieces)


def build_puzzle_from_crop(crop: np.ndarray, box: Box) -> Tuple[List[Piece], int, int]:
    tile_w = crop.shape[1] // GRID
    tile_h = crop.shape[0] // GRID
    pieces: List[Piece] = []
    for row in range(GRID):
        for col in range(GRID):
            sx = col * tile_w
            sy = row * tile_h
            w = crop.shape[1] - sx if col == GRID - 1 else tile_w
            h = crop.shape[0] - sy if row == GRID - 1 else tile_h
            tile = crop[sy:sy + h, sx:sx + w].copy()
            pieces.append(Piece(row=row, col=col, img=tile, x=0, y=0, w=w, h=h))
    slots = [(box.x + c * tile_w, box.y + r * tile_h) for r in range(GRID) for c in range(GRID)]
    random.shuffle(slots)
    for i, piece in enumerate(pieces):
        piece.x, piece.y = slots[i]
        if is_near_own_cell(piece, box, tile_w, tile_h):
            snap_piece(piece, box, tile_w, tile_h)
    return pieces, tile_w, tile_h


def draw_puzzle(frame: np.ndarray, pieces: List[Piece], box: Box, solved: bool) -> None:
    tile_w = box.width // GRID
    tile_h = box.height // GRID
    for i in range(1, GRID):
        x = box.x + i * tile_w
        y = box.y + i * tile_h
        cv2.line(frame, (x, box.y), (x, box.y + box.height), (80, 80, 80), 1)
        cv2.line(frame, (box.x, y), (box.x + box.width, y), (80, 80, 80), 1)
    order = sorted(pieces, key=lambda p: p.dragging)
    for p in order:
        x, y = int(p.x), int(p.y)
        if p.img.shape[0] != p.h or p.img.shape[1] != p.w:
            disp = cv2.resize(p.img, (p.w, p.h))
        else:
            disp = p.img
        frame[y:y + p.h, x:x + p.w] = disp
        if p.dragging:
            cv2.rectangle(frame, (x, y), (x + p.w, y + p.h), (100, 200, 255), 2)
    color = SOLVED_COLOR if solved else FRAME_COLOR
    draw_frame_box(frame, box, color)
    if solved:
        cx = box.x + box.width // 2
        cy = box.y + box.height // 2
        cv2.putText(
            frame,
            "COMPLETE! Fist to save",
            (cx - 120, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            SOLVED_COLOR,
            2,
            cv2.LINE_AA,
        )


def save_strip_image(images: List[np.ndarray], path: Path) -> None:
    if not images:
        return
    h = max(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            w = int(im.shape[1] * scale)
            im = cv2.resize(im, (w, h))
        resized.append(im)
    strip = np.hstack(resized)
    cv2.imwrite(str(path), strip)
    print(f"Saved strip: {path}")


class PuzzleCamApp:
    def __init__(self, camera: int, mirror: bool) -> None:
        self.camera = camera
        self.mirror = mirror
        self.landmarker = create_landmarker()
        self.state = "tracking"
        self.gallery: List[np.ndarray] = []
        self.pieces: List[Piece] = []
        self.board_box: Optional[Box] = None
        self.tile_w = 0
        self.tile_h = 0
        self.full_photo: Optional[np.ndarray] = None
        self.solved = False
        self.freeze_since: Optional[float] = None
        self.countdown_start: Optional[float] = None
        self.last_frame_box: Optional[Box] = None
        self.last_frame_at = 0.0
        self.fist_hold = 0
        self.drag_piece: Optional[Piece] = None
        self.drag_offset = (0.0, 0.0)
        self.status = "Starting camera..."

    def reset_puzzle(self) -> None:
        self.pieces = []
        self.board_box = None
        self.full_photo = None
        self.solved = False
        self.state = "tracking"
        self.countdown_start = None
        self.freeze_since = None
        self.fist_hold = 0
        self.drag_piece = None
        self.status = "Tracking hands - pinch both hands to frame"

    def reset_all(self) -> None:
        self.gallery.clear()
        self.reset_puzzle()
        self.status = "Reset all"

    def capture_and_build_puzzle(self, clean_frame: np.ndarray, box: Box) -> None:
        crop = clean_frame[box.y:box.y + box.height, box.x:box.x + box.width].copy()
        crop = apply_photobooth(crop)
        self.full_photo = crop.copy()
        self.pieces, self.tile_w, self.tile_h = build_puzzle_from_crop(crop, box)
        self.board_box = box
        self.solved = reconcile_placed(self.pieces, box, self.tile_w, self.tile_h)
        self.state = "puzzle"
        self.fist_hold = 0
        self.status = "Solve with pinch | q quit r reset"

    def save_to_gallery(self) -> None:
        if self.full_photo is None:
            return
        if len(self.gallery) >= MAX_STRIP:
            self.status = "Strip full - press d to download or r to reset all"
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = OUTPUT_DIR / f"puzzle_{ts}.png"
        cv2.imwrite(str(path), self.full_photo)
        self.gallery.append(self.full_photo.copy())
        print(f"Saved puzzle {len(self.gallery)}/{MAX_STRIP}: {path}")
        self.reset_puzzle()
        self.status = f"Saved! Strip {len(self.gallery)}/{MAX_STRIP}"

    def handle_fist(self) -> None:
        if self.state == "puzzle" and self.board_box:
            self.solved = reconcile_placed(
                self.pieces, self.board_box, self.tile_w, self.tile_h
            )
            if self.solved and self.full_photo is not None:
                self.save_to_gallery()
            else:
                self.reset_puzzle()
        else:
            self.reset_puzzle()

    def handle_drag(self, pinching: bool, index_px: Tuple[float, float]) -> None:
        if self.board_box is None:
            return
        box = self.board_box
        if pinching:
            if self.drag_piece is None:
                piece = find_nearest_piece(index_px[0], index_px[1], self.pieces)
                if piece:
                    self.drag_piece = piece
                    piece.dragging = True
                    piece.placed = False
                    self.drag_offset = (
                        index_px[0] - piece.x,
                        index_px[1] - piece.y,
                    )
            if self.drag_piece:
                p = self.drag_piece
                p.x = index_px[0] - self.drag_offset[0]
                p.y = index_px[1] - self.drag_offset[1]
                p.x = max(box.x, min(p.x, box.x + box.width - p.w))
                p.y = max(box.y, min(p.y, box.y + box.height - p.h))
        elif self.drag_piece:
            p = self.drag_piece
            p.dragging = False
            if is_near_own_cell(p, box, self.tile_w, self.tile_h):
                displace_at_cell(p, p.row, p.col, box, self.tile_w, self.tile_h, self.pieces)
                snap_piece(p, box, self.tile_w, self.tile_h)
            else:
                cx = p.x + p.w / 2
                cy = p.y + p.h / 2
                col = int((cx - box.x) / self.tile_w)
                row = int((cy - box.y) / self.tile_h)
                col = max(0, min(GRID - 1, col))
                row = max(0, min(GRID - 1, row))
                displace_at_cell(p, row, col, box, self.tile_w, self.tile_h, self.pieces)
                p.x = cx - p.w / 2
                p.y = cy - p.h / 2
            self.drag_piece = None
            self.solved = reconcile_placed(self.pieces, box, self.tile_w, self.tile_h)

    def process_hands(
        self,
        frame: np.ndarray,
        clean_frame: np.ndarray,
        hand_list: List,
        now: float,
    ) -> None:
        h, w = frame.shape[:2]
        any_fist = any(is_fist(lm) for lm in hand_list)
        dragging = self.drag_piece is not None

        if any_fist and not dragging and self.state != "tracking":
            self.fist_hold += 1
            if self.fist_hold >= FIST_HOLD_FRAMES:
                self.fist_hold = 0
                self.handle_fist()
                return
        else:
            self.fist_hold = 0

        if self.state == "tracking":
            if len(self.gallery) >= MAX_STRIP:
                self.status = "Strip full - d download | r reset all"
                return
            if len(hand_list) == 2:
                ia = lm_to_px(hand_list[0][LM_INDEX_TIP], w, h, self.mirror)
                ib = lm_to_px(hand_list[1][LM_INDEX_TIP], w, h, self.mirror)
                box = compute_hand_frame(ia, ib, w, h)
                if box:
                    self.last_frame_box = box
                    self.last_frame_at = now
                    apply_bw_region(frame, box)
                    draw_frame_box(frame, box, FRAME_COLOR)
                if box and is_pinching(hand_list[0]) and is_pinching(hand_list[1]):
                    if self.freeze_since is None:
                        self.freeze_since = now
                    self.status = "Hold the pinch..."
                    if (now - self.freeze_since) * 1000 > FREEZE_HOLD_MS:
                        self.freeze_since = None
                        self.board_box = box
                        self.countdown_start = now
                        self.state = "countdown"
                else:
                    self.freeze_since = None
                    self.status = "Tracking hands"
            else:
                self.freeze_since = None
                if self.last_frame_box and (now - self.last_frame_at) * 1000 < FRAME_GRACE_MS:
                    apply_bw_region(frame, self.last_frame_box)
                    draw_frame_box(frame, self.last_frame_box, FRAME_COLOR)
                self.status = "Show both hands"
            return

        if self.state == "countdown" and self.board_box:
            elapsed = now - (self.countdown_start or now)
            remaining = COUNTDOWN_SECONDS - elapsed
            box = self.board_box
            apply_bw_region(frame, box)
            draw_frame_box(frame, box, FRAME_COLOR)
            if remaining <= 0:
                self.capture_and_build_puzzle(clean_frame, box)
                return
            n = int(math.ceil(remaining))
            cx = box.x + box.width // 2
            cy = box.y + box.height // 2
            cv2.putText(
                frame,
                str(n),
                (cx - 20, cy + 20),
                cv2.FONT_HERSHEY_DUPLEX,
                2.0,
                FRAME_COLOR,
                3,
                cv2.LINE_AA,
            )
            self.status = f"Capturing in {n}..."
            return

        if self.state == "puzzle" and self.board_box:
            box = self.board_box
            active_lm = None
            for lm in hand_list:
                if is_pinching(lm):
                    active_lm = lm
                    break
            if active_lm is not None:
                px = lm_to_px(active_lm[LM_INDEX_TIP], w, h, self.mirror)
                self.handle_drag(True, px)
            elif self.drag_piece:
                self.handle_drag(False, (0, 0))
            if not self.drag_piece:
                self.solved = reconcile_placed(self.pieces, box, self.tile_w, self.tile_h)
            draw_puzzle(frame, self.pieces, box, self.solved)
            if self.solved:
                self.status = "Complete! Hold fist to save"
            else:
                placed = sum(1 for p in self.pieces if p.placed)
                self.status = f"Puzzle {placed}/{len(self.pieces)} - pinch to drag"

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera}")
        print("PuzzleCam Python")
        print("  Both hands pinch -> hold -> countdown -> puzzle")
        print("  Pinch piece to drag | fist to save | q quit r reset d download strip")

        frame_idx = 0
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame = cv2.flip(raw, 1) if self.mirror else raw
            clean_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / 30)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1
            now = time.time()
            hand_list = list(result.hand_landmarks) if result.hand_landmarks else []

            for lm in hand_list:
                draw_hand_skeleton(frame, lm, self.mirror)

            if hand_list:
                self.process_hands(frame, clean_frame, hand_list, now)
            elif self.state == "countdown" and self.board_box:
                self.process_hands(frame, clean_frame, [], now)
            elif self.state == "puzzle" and self.board_box:
                if self.drag_piece:
                    self.handle_drag(False, (0, 0))
                self.solved = reconcile_placed(
                    self.pieces, self.board_box, self.tile_w, self.tile_h
                )
                draw_puzzle(frame, self.pieces, self.board_box, self.solved)
                self.status = "Solve with pinch"
            elif self.state == "tracking":
                self.status = "Looking for hands..."

            draw_status_bar(frame, [self.status, "q quit | r reset | d download strip"], len(self.gallery))
            cv2.imshow("PuzzleCam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if len(self.gallery) >= MAX_STRIP:
                    self.reset_all()
                else:
                    self.reset_puzzle()
            if key == ord("d") and len(self.gallery) >= MAX_STRIP:
                path = OUTPUT_DIR / f"strip_{int(time.time())}.png"
                save_strip_image(self.gallery, path)
            if key == ord("m"):
                self.mirror = not self.mirror

        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="PuzzleCam Python")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    app = PuzzleCamApp(camera=args.camera, mirror=not args.no_mirror)
    app.run()


if __name__ == "__main__":
    main()
