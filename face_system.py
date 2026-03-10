import os
import numpy as np
import cv2
import argparse

from PIL import Image
import torch
import pickle
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# MTCNN optimization of processor
if torch.cuda.is_available() == 1:
    DEVICE          = 'cuda'  
else: DEVICE = 'cpu'
DB_PATH         = 'face_database' 

EMBEDDINGS_FILE = 'embeddings.pkl'
# Detection threshold to be considered valid face. Also used for MTCNN
THRESHOLD       = 0.7           
# Basic image size for MTCNN        
IMG_SIZE        = 180     
# Minimum detection size - smaller is thrown out               
MIN_FACE_SIZE   = 60                     



def build_database(mtcnn, resnet, db_dir: str = DB_PATH) -> dict:
    """
    Walk db_dir/<person_name>/*.jpg|png and build {name: [embeddings]} dict.
    Saves result to EMBEDDINGS_FILE and returns it.
    """
    if not os.path.isdir(db_dir):
        os.makedirs(db_dir)
        print(f"[INFO] Created database directory: {db_dir}/")
        print("[INFO] Add sub-folders (one per person) with ≥10 face images each.")
        return {}

    database: dict[str, list[np.ndarray]] = {}
    total_imgs = 0

    for person in sorted(os.listdir(db_dir)):
        person_dir = os.path.join(db_dir, person)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for fname in os.listdir(person_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = os.path.join(person_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                face_tensor = mtcnn(img)          # (N,3,160,160) or None
                if face_tensor is None:
                    print(f"  [WARN] No face found in {img_path}")
                    continue
                # Use only the first detected face in each image
                if face_tensor.ndim == 4:
                    face_tensor = face_tensor[0]
                embeddings.append(get_embedding(resnet, face_tensor))
                total_imgs += 1
            except Exception as e:
                print(f"  [ERR] {img_path}: {e}")

        if embeddings:
            database[person] = embeddings
            print(f"  [OK] {person}: {len(embeddings)} embeddings")
        else:
            print(f"  [SKIP] {person}: no valid faces found")

    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(database, f)

    print(f"\n[INFO] Database built – {len(database)} identities, {total_imgs} images.")
    return database


def get_embedding(resnet, face_tensor: torch.Tensor) -> np.ndarray:
    """Return L2-normalised 512-D embedding for a (1,3,160,160) tensor."""
    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0).to(DEVICE))
    return emb.cpu().numpy()[0]


def load_models():
    # start up resnet. 
    model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    tsflw_mtcnn = MTCNN(
        # Check this website: https://mtcnn.readthedocs.io/en/latest/usage_params/#fine-tuning-parameters-for-each-detection-stage
        # This also helps: https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py
        # MTCNN input image size for FaceNet
        image_size=IMG_SIZE, 
        margin=10, # crop margin for edge of images. 
        # If too high, will throw first give bad recog
        # Error if = image size. More crop is faster
        # again this throws out tiny faces. Make bigger if smaller images
        min_face_size=MIN_FACE_SIZE,
        # MTCNN thresholds. Can define seperately or all of them here
        thresholds=[THRESHOLD, THRESHOLD, THRESHOLD],
        factor=0.709,
        # Refine images 
        post_process=True,

        keep_all=True, # false by default. 
        # Returns only the best detection if true
        device=DEVICE
        # Tell MTCNN what to use
    )
    return tsflw_mtcnn, model


def load_database() -> dict:
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}


# ────────────────────────── Recognition logic ─────────────────────────────
def recognise(embedding: np.ndarray, database: dict) -> tuple[str, float]:
    """
    Compare embedding against all stored embeddings.
    Returns (best_name, best_score).  'Unknown' if score < THRESHOLD.
    """
    best_name, best_score = 'Unknown', 0.0
    emb = embedding.reshape(1, -1)
    for name, stored_embs in database.items():
        if isinstance(stored_embs, list):
            stored_embs_array = np.array(stored_embs)
        else:
            stored_embs_array = stored_embs.reshape(1, -1)
        sims = cosine_similarity(emb, stored_embs_array)[0]
        score = float(np.max(sims))
        if score > best_score:
            best_score = score
            best_name = name
    if best_score < THRESHOLD:
        return 'Unknown', best_score
    return best_name, best_score


# Webcam Gui
def run_live(mtcnn, resnet, database: dict):
    """
    Capture frames from the default webcam, detect faces with MTCNN,
    recognise each face, and draw annotated bounding boxes.
    Press 'q' to quit, 'r' to rebuild the database.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0).")

    print("[INFO] Live feed started. Press 'q' to quit | 'r' to rebuild DB.")

    # Pre-compute one mean embedding per person for speed
    mean_db = {name: np.mean(embs, axis=0) for name, embs in database.items()}

    frame_count = 0
    boxes_cache, labels_cache = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run detection every 3 frames to keep UI responsive
        if frame_count % 3 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
            boxes_cache, labels_cache = [], []

            if boxes is not None:
                face_tensors = mtcnn.extract(pil_img, boxes, save_path=None)

                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob < 0.90 or face_tensors is None:
                        continue
                    ft = face_tensors[i]
                    if ft is None:
                        continue

                    emb = get_embedding(resnet, ft)
                    name, score = recognise(emb, mean_db)

                    x1, y1, x2, y2 = [int(v) for v in box]
                    boxes_cache.append((x1, y1, x2, y2))
                    labels_cache.append((name, score))

        # ── Draw cached annotations ─────────────────────────────────────
        for (x1, y1, x2, y2), (name, score) in zip(boxes_cache, labels_cache):
            color = (0, 200, 0) if name != 'Unknown' else (0, 0, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name}  {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # HUD
        cv2.putText(frame, f"Identities in DB: {len(database)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, "q=quit  r=rebuild DB", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Face Recognition System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("[INFO] Rebuilding database …")
            database = build_database(mtcnn, resnet)
            mean_db = {n: np.mean(e, axis=0) for n, e in database.items()}
            boxes_cache, labels_cache = [], []

    cap.release()
    cv2.destroyAllWindows()


# Check if running main program, otherwise this just provides functions
if __name__ == '__main__':
    # This allows us to use --rebuild or specify a new DB path when running from cmd line
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--rebuild', action='store_true',
                        help='Manually rebuild of DB')
    parser.add_argument('--db', default=DB_PATH,
                        help='Specify face DB path')
    args = parser.parse_args()

    print(f"[DEBUG] Using device: {DEVICE}")
    print("[DEBUG] Loading models…")
    tsflw_mtcnn, model = load_models()
    print("[DEBUG] Models loaded.")

    if args.rebuild == 1 or not os.path.exists(EMBEDDINGS_FILE) == 1:
        # Args added are run here if there are any, or if there's an embeddings file 
        database = build_database(tsflw_mtcnn, model, args.db)
        print(f"[ARGS] Loaded {len(database)} faces")

    else:
        database = load_database()
        print(f"[NO_ARGS] Loaded {len(database)} faces")

    run_live(tsflw_mtcnn, model, database)
