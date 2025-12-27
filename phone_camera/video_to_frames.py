#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2

# =========================
# EDIT ONLY THESE
# =========================
VIDEO_PATH = "/home/orr/university_projects/relative-pose-estimation/phone_camera/forward_with_stuff/forward_with_stuff.mp4"
OUT_DIR = "/home/orr/university_projects/relative-pose-estimation/phone_camera/forward_with_stuff/all_images"
SAVE_EVERY_N = 1          # 1 = כל פריים, 2 = כל פריים שני, וכו'
JPG_QUALITY = 100          # 0-100
MAX_FRAMES = 0            # 0 = בלי הגבלה, אחרת יעצור אחרי N פריימים שנשמרו
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    saved = 0
    read_idx = 0

    print(f"[INFO] Reading: {VIDEO_PATH}")
    print(f"[INFO] Output : {OUT_DIR}")
    print(f"[INFO] Save every {SAVE_EVERY_N} frame(s)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if (read_idx % SAVE_EVERY_N) == 0:
            out_path = os.path.join(OUT_DIR, f"{read_idx:06d}.png")
            ok2 = cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY)])
            if not ok2:
                print(f"[WARN] Failed to write: {out_path}")
            else:
                saved += 1
                if saved % 50 == 0:
                    print(f"[INFO] saved {saved} frames... (last: {out_path})")

                if MAX_FRAMES > 0 and saved >= MAX_FRAMES:
                    print(f"[INFO] Reached MAX_FRAMES={MAX_FRAMES}. Stopping.")
                    break

        read_idx += 1

    cap.release()
    print(f"[DONE] total read frames: {read_idx}")
    print(f"[DONE] total saved frames: {saved}")

if __name__ == "__main__":
    main()
