"""
capture_faces.py
────────────────
Interactive tool to capture face images from the webcam for building the
recognition database.

Controls while capture window is open:
    SPACE   – capture a frame
    a       – toggle auto-capture every 1 s
    q       – quit
"""

import os       # for file/directory operations
import time     # for auto capture timing
import argparse # for parsing command line arguments
import cv2      # OpenCV: webcam and image display

# folder where all captured face images will be stored
DB_PATH = 'face_database'


def capture_faces(name: str, target_count: int = 15):
    """
    Opens the webcam and lets the user capture face images for a given person.

    Args:
        name         : The person's name — used as the subfolder name.
        target_count : How many new images to capture (default 15).
    """

    # create path e.g. face_database/Alice/
    save_dir = os.path.join(DB_PATH, name)

    # create directory
    os.makedirs(save_dir, exist_ok=True)

    # count exisitng images of the user
    existing = len([f for f in os.listdir(save_dir)
                    if f.lower().endswith(('.jpg', '.png'))])
    print(f"[INFO] Saving to: {save_dir}  (already have {existing} images)")

    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    captured   = 0      # number of images captured
    auto_mode  = False  # auto capture toggle on/off
    last_auto  = 0.0    # timestamp of the last auto capture

    print("[INFO] Controls:  SPACE=capture | a=toggle auto | q=quit")

    # loop till sufficient images
    while captured < target_count:

        # read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break  # stop if webcam feed is lost

        # helpful display on screen
        remaining = target_count - captured

        # user name top left
        cv2.putText(frame, f"Name: {name}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2)

        # how many pictures have been captured so far
        cv2.putText(frame, f"Captured: {captured}/{target_count}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2)

        # show auto/manual mode
        mode_txt = "AUTO" if auto_mode else "MANUAL"
        cv2.putText(frame, f"Mode: {mode_txt}", (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # control legend
        cv2.putText(frame, "SPACE=capture  a=auto  q=quit",
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # window is named
        cv2.imshow(f"Capture – {name}", frame)

    
        # auto mode: save a frame every 1 second
        if auto_mode and (time.time() - last_auto) >= 1.0:
            
            # generate filename
            idx   = existing + captured + 1
            fname = os.path.join(save_dir, f"{name}_{idx:03d}.jpg")
            cv2.imwrite(fname, frame)   # save frame as a JPEG
            captured += 1
            last_auto = time.time()     # reset 1 second timer
            print(f"  [AUTO] Saved {fname}")

    
        # to read which key was pressed; the & 0xFF ensures that key is read correctly
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            # SPACE: manually save the current frame
            idx   = existing + captured + 1
            fname = os.path.join(save_dir, f"{name}_{idx:03d}.jpg")
            cv2.imwrite(fname, frame)
            captured += 1
            print(f"  [SAVE] Saved {fname}")

        elif key == ord('a'):
            # a: toggle auto-capture on/off
            auto_mode = not auto_mode
            last_auto = time.time()     # reset timer so it doesn't fire immediately
            print(f"  [INFO] Auto-capture {'ON' if auto_mode else 'OFF'}")

        elif key == ord('q'):
            # q: quit early before reaching target_count
            break

    
    cap.release()               # release webcam
    cv2.destroyAllWindows()     # close all OpenCV windows

    total = existing + captured
    print(f"\n[DONE] {name}: {total} images in {save_dir}")

    # warn user if less than the recommended minimum
    if total < 10:
        print(f"[WARN] Recommended minimum is 10 images (currently {total}).")


# only run when script is executed directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture face images for database')

    # name required: the persons name for folder name
    parser.add_argument('--name',  required=True, help='Person name (used as folder name)')

    # count optional: defaults to 15 if not provided
    parser.add_argument('--count', type=int, default=15, help='Number of images to capture')

    args = parser.parse_args()
    capture_faces(args.name, args.count)
