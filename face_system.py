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

# Pickle file (serialized bytestream for networking) to store {name: [embeddings]} dict)
EMBEDDINGS_FILE = 'embeddings.pkl'
# Detection threshold to be considered valid face. Also used for MTCNN
THRESHOLD       = 0.7           
# Basic image size for MTCNN        
IMG_SIZE        = 180     
# Minimum detection size - smaller is thrown out               
MIN_FACE_SIZE   = 60                     



def build_database(mtcnn, resnet, db_dir: str = DB_PATH) -> dict:
    # 
    """
    Walk db_dir/<person_name>/*.jpg|png and build {name: [embeddings]} dict.
    Saves result to EMBEDDINGS_FILE and returns it.
    """
    # if the current path does not refer to an existing database path, make a new database
    if not os.path.isdir(db_dir):
        os.makedirs(db_dir)
        print(f"[INFO] Created database directory: {db_dir}/")
        return {} # Since we just made an empty DB, there's nothing to load so return


    # create a database as a dict class, which stores data as key-value pairs
    # Key is a string (person's name) and value is a list of numpy arrays 
    # (embeddings for that person).

    database: dict[str, list[np.ndarray]] = {}
    # Start an int to store an iterative count. Probably a better way to do this 
    total_imgs = 0

    for person in sorted(os.listdir(db_dir)):
        # If somehow the person's name is not a directory, skip it
        person_dir = os.path.join(db_dir, person)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []

        for fname in os.listdir(person_dir):
            # If invalid image type, skip it
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            # Set a var img_path to the current path, which would be the database path + the person's name + the image file name
            img_path = os.path.join(person_dir, fname)
            # Try to open but if error is thrown it will escape
            try: # use try in case of bad/corrupt/invalid images
                img = Image.open(img_path).convert('RGB')

                # Use MTCNN to obtain a face tensor/detection from the image
                face_tensor = mtcnn(img)          # (N,3,160,160) or None
                # Call out if an image assigned to a person fails to detect a face with MTCNN and skip it
                if face_tensor is None:
                    print(f"  [WARN] No face found in {img_path}")
                    continue
                
                # Use only the first detected face in each image
                if face_tensor.ndim == 4:
                    face_tensor = face_tensor[0]
                # This means if there are many faces in training images, it will only use the first
                # Thus for training ensure ppl upload single faces
                embeddings.append(get_embedding(resnet, face_tensor))
                total_imgs += 1
            # If try fails, print the error pointing to the image
            except Exception as e:
                print(f"  [ERR] {img_path}: {e}")

        # check if embeddings from this person is not empty
        if embeddings:
            # dict key is person's name, embeddings is numpy arrays of features
            database[person] = embeddings
            print(f"  [INFO] {person}: {len(embeddings)} embeddings")
        else:
            # If no valid, skip it
            print(f"  [INFO] {person}: no valid faces found")

    # Pickle the database (compress it, basically)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(database, f)

    print(f"\n[INFO] Database built: {len(database)} identities, {total_imgs} images.")
    return database

# Uses a (1,3,IMG_SIZE,IMG_SIZE) array (a tensor) to make an embedding layer of 512-Dim data. 
# This gives each of the 512 data vectors a unique ID, & positions them
# near similar vectors/vectors that often go together, like Aiden's eyes and his mouth

# Requires the face tensor detected, and ResNet to be loaded to create embed layer
def get_embedding(resnet, face_tensor: torch.Tensor) -> np.ndarray: 
# Specify as numpy N-Dimensional array ahead of time to prevent faulty dynamic allocation
    # Turn off gradient calculation, since it saves compute. For training, turn this on. 
    with torch.no_grad():  
        # In embedding layer, use resnet to correlate 512 feature 
        # vectors into a feature map derived from the face tensor 
        # that is unsqueezed to 0, separating every single feature 
        emb = resnet(face_tensor.unsqueeze(0).to(DEVICE)) # Specify device to compute
    # Return our single resnet result as a numpy array (1 face at a time)
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
    # If the embeddings file specifies a path, load it
    if os.path.exists(EMBEDDINGS_FILE) ==1:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            # Unpickle (deserialize) the bytestream in embeddings 
            # to load the DB. 'rb' specifies to read it in binary mode.
            return pickle.load(f)
    return {}

# Create a recognition of a person, outputting a name and confidence level
def recognise(embedding: np.ndarray, database: dict) -> tuple[str, float]:
    # Takes a database and a recognized face embedding

    # Compare an embedding of the camera image to the whole database.
    # Returns the best match and its similarity score,
    # Or 'Unknown' if the confidence is too low

    #Start at a score of 0 with unknown
    best_name, best_score = 'Unknown', 0.0

    # Reshape the array for cosine similarity, which expects 2D arrays (1,512) and (N,512)
    emb = embedding.reshape(1, -1)

    # For every name and embedding in the dict database, compare
    for name, stored_embs in database.items():
        
        # If multiple embeddings for a person, make them an array
        if isinstance(stored_embs, list):
            stored_embs_array = np.array(stored_embs)
        # If only one embedding, reshape it to (1,512) for cosine similarity
        else:
            stored_embs_array = stored_embs.reshape(1, -1)
        
        # Compare the angle between the 2 vectors (cosine sim)
        sims = cosine_similarity(emb, stored_embs_array)[0]

        # Return the highest similarity score for this person, since they may have multiple embeddings
        score = float(np.max(sims))

        # If this is the new best confidence, then this person is recognized
        if score > best_score:
            best_score = score
            best_name = name
    # after comparing all the faces, if none are better than the 
    # requisite threshold, change the identity to Unknown.
    if best_score < THRESHOLD:
        return 'Unknown', best_score
    
    # Otherwise, return the final best name and confidence
    return best_name, best_score


# Webcam Gui
def run_live(mtcnn, resnet, database: dict):
    
    # Start video capture feed from webcam, and throw error if it fails
    # Usually fails cuz webcam already in use
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0).")

    # Explain commands in text box
    print("[INFO] Live feed started. Press 'q' to quit | 'r' to rebuild DB.")

    # Pre-compute one mean embedding per person for speed. Use 1 singles embedding per person
    # This helps speed for live webcame demos. Can delete for accuracy at huge comptue cost
    mean_db = {name: np.mean(embs, axis=0) for name, embs in database.items()}

    # Start at 0 frames
    frame_count = 0
    # Initialize caches
    boxes_cache, labels_cache = [], []

    # While webcam is open and not error code that != 1, loop
    while True:
        # Obtain the frame and return code. If return != 1, break
        ret, frame = cap.read()
        if not ret:
            break

        # Increment frame
        frame_count += 1

        # Run detection every 3 frames to prevent lag while being convincing
        if frame_count % 3 == 0:
            # Convert to proper color space and PIL format for MTCNN
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Obtain bounding boxes and confidences from MTCNN. 
            # Can include landmarks if you want, giving locations
            # of facial features
            boxes, probs = mtcnn.detect(pil_img, landmarks=False)
            
            # clear caches (reset them)
            boxes_cache, labels_cache = [], []

            # If there are boxes (faces) detected, extract the face tensors for each box
            if boxes is not None:
                face_tensors = mtcnn.extract(pil_img, boxes, save_path=None)


                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    
                    # If the confidence is too low, which is unlikely but possible since MTCNN just detected a face, skip it
                    if prob < 0.90 or face_tensors is None:
                        continue

                    # Check that there is a valid face tensor
                    ft = face_tensors[i]
                    if ft is None:
                        continue

                    emb = get_embedding(resnet, ft)
                    name, score = recognise(emb, mean_db)

                    # Extract bounding boxes and label data and add to cache
                    x1, y1, x2, y2 = [int(v) for v in box]
                    boxes_cache.append((x1, y1, x2, y2))
                    labels_cache.append((name, score))

        # After detection loop finishes, draw the boxes and labels from the cache on the frame
        for (x1, y1, x2, y2), (name, score) in zip(boxes_cache, labels_cache):
            color = (0, 200, 0) if name != 'Unknown' else (0, 0, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name}  {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Add info text
        cv2.putText(frame, f"Identities in DB: {len(database)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, "q=quit  r=rebuild DB", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Show the frame with detections and labels
        cv2.imshow("Face Recognition System", frame)

        # Cache keypresses
        key = cv2.waitKey(1) & 0xFF
        
        # If user presses a command, execute it (they could also command line it)
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("[INFO] Rebuilding database …")
            database = build_database(mtcnn, resnet)
            mean_db = {n: np.mean(e, axis=0) for n, e in database.items()}
            boxes_cache, labels_cache = [], []

    # Ensure we give up the cam, and close windows when done
    # This prevents us from losing our webcam akin because of 
    # our stupidity, akin to a memory leak
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
