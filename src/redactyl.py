import argparse
import os
import re
import sys
import time

import cv2
import mss
import numpy as np
import openvino as ov
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QApplication, QMainWindow

# --- V2: OCR-PROOF SETTINGS ---
SECRET_PATTERN = r"(sk[-_]?live[-_]?[a-z0-9]+|AKIA[0-9A-Z]{16}|(pass|password|secret|key|token|auth)\s*[:=]+)"

script_dir = os.path.dirname(os.path.abspath(__file__))

dict_path = os.path.join(script_dir, "en_dict.txt")
if not os.path.exists(dict_path):
    print(f"❌ ERROR: Could not find {dict_path}")
    sys.exit(1)

with open(dict_path, "r", encoding="utf-8") as f:
    CHAR_LIST = [" "] + [line.strip("\n") for line in f.readlines()]


def decode_ctc(preds):
    preds_idx = np.argmax(preds, axis=2)[0]
    text = ""
    for i in range(len(preds_idx)):
        idx = preds_idx[i]
        if idx > 0 and idx < len(CHAR_LIST) and (i == 0 or idx != preds_idx[i - 1]):
            text += CHAR_LIST[idx]
    return text


# --- 1. THE QUADRANT NPU COMPILATION ---
sct = mss.mss()
monitor = sct.monitors[1]
orig_w, orig_h = monitor["width"], monitor["height"]

npu_w, npu_h = 960, 544

print(f"Compiling NPU Detection Model: [1, 3, {npu_h}, {npu_w}]")

home_dir = os.path.expanduser("~")
det_onnx_path = os.path.join(
    home_dir, ".paddlex", "official_models", "PP-OCRv5_server_det", "model.onnx"
)
rec_model_path = os.path.join(script_dir, "en_PP-OCRv4_rec_infer", "inference.pdmodel")

core = ov.Core()
os.makedirs("./npu_cache", exist_ok=True)
core.set_property({"CACHE_DIR": "./npu_cache"})
# If you are plugged in and want the red boxes to snap to your
# secrets instantly, you can change the timer to self.timer.start(200)
# and bump the INFERENCE_NUM_THREADS back up to 4.
# battery saver
# core.set_property("CPU", {"INFERENCE_NUM_THREADS": 2})
# high performance
core.set_property("CPU", {"INFERENCE_NUM_THREADS": 4})

ov_model = core.read_model(model=det_onnx_path)
ov_model.reshape({ov_model.inputs[0].any_name: [1, 3, npu_h, npu_w]})
compiled_model = core.compile_model(model=ov_model, device_name="NPU")
output_layer = compiled_model.outputs[0]

print("Compiling CPU Recognition Model (Dynamic Width)...")
rec_model = core.read_model(model=rec_model_path)
rec_model.reshape({rec_model.inputs[0].any_name: ov.PartialShape([1, 3, 48, -1])})
compiled_rec_model = core.compile_model(model=rec_model, device_name="CPU")
rec_output_layer = compiled_rec_model.outputs[0]

print("✅ Vision Pipeline Ready. Waking up UI...")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Overlay(QMainWindow):
    def __init__(self, show_all_text=False):
        super().__init__()

        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]

        self.setGeometry(
            self.monitor["left"],
            self.monitor["top"],
            self.monitor["width"],
            self.monitor["height"],
        )

        self.show_all_text = show_all_text
        self.active_redactions = {}
        self.all_text_boxes = []

        mid_w, mid_h = orig_w // 2, orig_h // 2
        self.quadrants = [
            (0, 0, mid_w, mid_h),
            (mid_w, 0, orig_w - mid_w, mid_h),
            (0, mid_h, mid_w, orig_h - mid_h),
            (mid_w, mid_h, orig_w - mid_w, orig_h - mid_h),
        ]

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_screen)
        # If you are plugged in and want the red boxes to snap to your
        # secrets instantly, you can change the timer to
        # self.timer.start(200) and bump the INFERENCE_NUM_THREADS back up
        # to 4.
        # battery saver
        # self.timer.start(1000)
        # high performance
        self.timer.start(200)

    def process_screen(self):
        current_time = time.time()
        img_rgb = cv2.cvtColor(
            np.array(self.sct.grab(self.monitor))[:, :, :3], cv2.COLOR_BGR2RGB
        )

        screen = QApplication.primaryScreen()
        logical_w = screen.geometry().width()
        logical_h = screen.geometry().height()

        ratio_x = logical_w / orig_w
        ratio_y = logical_h / orig_h

        offset_x = self.geometry().x()
        offset_y = self.geometry().y()

        new_all_boxes = []

        for qx, qy, qw, qh in self.quadrants:
            quad_img = img_rgb[qy : qy + qh, qx : qx + qw]
            scale_factor = npu_w / qw
            scaled_w, scaled_h = int(qw * scale_factor), int(qh * scale_factor)

            canvas = np.zeros((npu_h, npu_w, 3), dtype=np.uint8)
            canvas[0:scaled_h, 0:scaled_w] = cv2.resize(
                quad_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )

            input_tensor = np.expand_dims(
                np.transpose(
                    (canvas.astype(np.float32) / 255.0 - MEAN) / STD, (2, 0, 1)
                ),
                axis=0,
            )
            valid_heatmap = compiled_model([input_tensor])[output_layer][0][0][
                0:scaled_h, 0:scaled_w
            ]

            binary_map = cv2.dilate(
                (valid_heatmap > 0.25).astype(np.uint8) * 255,
                np.ones((4, 20), np.uint8),
                iterations=1,
            )
            contours, _ = cv2.findContours(
                binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                if w > 4 and h > 3:
                    tight_px = max(0, int(x / scale_factor) + qx - 8)
                    tight_py = max(0, int(y / scale_factor) + qy - 4)
                    tight_pw = int(w / scale_factor) + 16
                    tight_ph = int(h / scale_factor) + 8

                    ui_px = max(0, tight_px - 4)
                    ui_py = max(0, tight_py - 4)
                    ui_pw = tight_pw + 8
                    ui_ph = tight_ph + 8

                    final_x = int((ui_px * ratio_x) - offset_x)
                    final_y = int((ui_py * ratio_y) - offset_y)
                    final_w = int(ui_pw * ratio_x)
                    final_h = int(ui_ph * ratio_y)

                    ui_box = (final_x, final_y, final_w, final_h)

                    if self.show_all_text:
                        new_all_boxes.append(ui_box)

                    crop = img_rgb[
                        tight_py : tight_py + tight_ph, tight_px : tight_px + tight_pw
                    ]
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        ratio = crop.shape[1] / crop.shape[0]
                        resized_crop = cv2.resize(
                            crop,
                            (max(int(48 * ratio), 10), 48),
                            interpolation=cv2.INTER_AREA,
                        )
                        norm_crop = np.transpose(
                            (resized_crop.astype(np.float32) / 255.0 - 0.5) / 0.5,
                            (2, 0, 1),
                        )[np.newaxis, ...]

                        rec_preds = compiled_rec_model([norm_crop])[rec_output_layer]
                        text = decode_ctc(rec_preds)

                        is_secret = re.search(SECRET_PATTERN, text, re.IGNORECASE)

                        if self.show_all_text and len(text.strip()) > 1:
                            if is_secret:
                                print("🧠 AI Read: [SECRET HIDDEN TO PREVENT LOOP]")
                            else:
                                print(f"🧠 AI Read: '{text}'")

                        if is_secret:
                            if not self.show_all_text:
                                print("🚨 REDACTING A SECRET!")

                            # --- FIX: SPATIAL DE-DUPLICATION ---
                            # Check if this new box heavily overlaps with any existing box
                            keys_to_remove = []
                            cx1, cy1 = final_x + (final_w / 2), final_y + (final_h / 2)

                            for existing_box in self.active_redactions.keys():
                                ex, ey, ew, eh = existing_box
                                cx2, cy2 = ex + (ew / 2), ey + (eh / 2)

                                # If centers are within 30 pixels, it's the same secret jittering around
                                if abs(cx1 - cx2) < 30 and abs(cy1 - cy2) < 30:
                                    keys_to_remove.append(existing_box)

                            for k in keys_to_remove:
                                del self.active_redactions[k]

                            # Now add the fresh box. Because we deleted the old ones, it won't stack!
                            self.active_redactions[ui_box] = current_time + 5.0

        self.all_text_boxes = new_all_boxes
        self.update()

    def paintEvent(self, event):
        current_time = time.time()
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)

        if self.show_all_text:
            painter.setBrush(QColor(0, 255, 0, 40))
            for box in self.all_text_boxes:
                painter.drawRect(*box)

        # Draw a consistent, un-stacked Alpha 180 box
        painter.setBrush(QColor(255, 0, 0, 180))

        expired = []
        for box, expiry in self.active_redactions.items():
            if current_time < expiry:
                painter.drawRect(*box)
            else:
                expired.append(box)

        for e in expired:
            del self.active_redactions[e]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Privacy Shield")
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Paint transparent green boxes around all detected text.",
    )
    args, unknown = parser.parse_known_args()

    app = QApplication([sys.argv[0]] + unknown)
    window = Overlay(show_all_text=args.show_all)
    window.show()
    sys.exit(app.exec())
