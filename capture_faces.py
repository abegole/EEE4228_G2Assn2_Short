"""
capture_faces.py
────────────────
Interactive tool to capture face images from the webcam for building the
recognition database.

Usage:
    python capture_faces.py --name "Alice" --count 15

Controls while capture window is open:
    SPACE   – capture a frame
    a       – toggle auto-capture every 1 s
    q       – quit
"""

import os
import time
import argparse
import cv2

DB_PATH = 'face_database'


def capture_faces(name: str, target_count: int = 15):
    save_dir = os.path.join(DB_PATH, name)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir)
                    if f.lower().endswith(('.jpg', '.png'))])
    print(f"[INFO] Saving to: {save_dir}  (already have {existing} images)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    captured   = 0
    auto_mode  = False
    last_auto  = 0.0

    print("[INFO] Controls:  SPACE=capture | a=toggle auto | q=quit")

    while captured < target_count:
        ret, frame = cap.read()
        if not ret:
            break

        # UI overlay
        remaining = target_count - captured
        cv2.putText(frame, f"Name: {name}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2)
        cv2.putText(frame, f"Captured: {captured}/{target_count}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2)
        mode_txt = "AUTO" if auto_mode else "MANUAL"
        cv2.putText(frame, f"Mode: {mode_txt}", (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, "SPACE=capture  a=auto  q=quit",
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow(f"Capture – {name}", frame)

        # Auto-capture
        if auto_mode and (time.time() - last_auto) >= 1.0:
            idx   = existing + captured + 1
            fname = os.path.join(save_dir, f"{name}_{idx:03d}.jpg")
            cv2.imwrite(fname, frame)
            captured += 1
            last_auto = time.time()
            print(f"  [AUTO] Saved {fname}")

        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            idx   = existing + captured + 1
            fname = os.path.join(save_dir, f"{name}_{idx:03d}.jpg")
            cv2.imwrite(fname, frame)
            captured += 1
            print(f"  [SAVE] Saved {fname}")
        elif key == ord('a'):
            auto_mode = not auto_mode
            last_auto = time.time()
            print(f"  [INFO] Auto-capture {'ON' if auto_mode else 'OFF'}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    total = existing + captured
    print(f"\n[DONE] {name}: {total} images in {save_dir}")
    if total < 10:
        print(f"[WARN] Recommended minimum is 10 images (currently {total}).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture face images for database')
    parser.add_argument('--name',  required=True, help='Person name (used as folder name)')
    parser.add_argument('--count', type=int, default=15, help='Number of images to capture')
    args = parser.parse_args()
    capture_faces(args.name, args.count)
