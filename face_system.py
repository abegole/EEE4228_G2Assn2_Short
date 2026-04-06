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
import random
from time import perf_counter
from sklearn.metrics import r2_score

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


def augment_image(img_np: np.ndarray) -> list[np.ndarray]:
    """Returns a list of augmented versions of the input image."""
    augmented = []
    h, w = img_np.shape[:2]
    center = (w // 2, h // 2)

    # Horizontal flip
    augmented.append(cv2.flip(img_np, 1))

    # Brightness variations
    augmented.append(cv2.convertScaleAbs(img_np, alpha=1.2, beta=30))   # bright
    augmented.append(cv2.convertScaleAbs(img_np, alpha=0.8, beta=-30))  # dark

    # Slight rotations
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1)
        augmented.append(cv2.warpAffine(img_np, M, (w, h)))

    return augmented

def get_embeddings_batch(resnet, face_tensors: list[torch.Tensor]) -> list[np.ndarray]:
    # Use resnet to process a whole batch at once for speed
    batch = torch.stack([ft for ft in face_tensors]).to(DEVICE)
    with torch.no_grad():
        embs = resnet(batch)
    return [e.cpu().numpy() for e in embs]
def build_database(mtcnn, resnet, db_dir: str = DB_PATH, debug=False, no_aug=False) -> dict:
    # if the current path does not refer to an existing database path, make a new database
    if debug:
        totaltime = perf_counter()
        filename_map: dict[str, list[dict]] = {}
        ppl_embed_time: dict[str, list[float]] = {}
        # Per-image tracking lists for the benchmark plot
        all_file_sizes_kb: list[float] = []
        all_times_ms: list[float] = []
        all_person_labels: list[str] = []


    
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
        if debug:
            start_person_time = perf_counter()

        # If somehow the person's name is not a directory despite being pulled from directories, skip it
        person_dir = os.path.join(db_dir, person)
        if not os.path.isdir(person_dir):
            continue

        #initialize embeds
        embeddings = []

        # Store tuples of (tensor, filename, augment_index) to track source image
        # aug_idx 0 = original, 1-5 = augmented versions
        face_tensor_batch: list[tuple] = []
        
        if debug:
            ppl_embed_time[person] = [0.0, 0.0]  # Initialize list for this person

        for fname in os.listdir(person_dir):
            # If invalid image type, skip it
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            # Set a var img_path to the current path, which would be the database path + the person's name + the image file name
            img_path = os.path.join(person_dir, fname)

            # Try to open but if error is thrown it will escape
            try: # use try in case of bad/corrupt/invalid images
                # Get file size in KB before loading image
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img)
                if debug:
                    file_size_kb = os.path.getsize(img_path) / 1024
                    ppl_embed_time[person][1] += file_size_kb


                # Generate original + 5 augmented versions of every image
                # This ensures every image contributes equally to the database
                if no_aug == True:
                    all_versions = [img_np]
                else:
                    all_versions = [img_np] + augment_image(img_np)

                if debug:
                    aug_success = 0

                for aug_idx, version in enumerate(all_versions):
                    # aug_idx 0 = original image, 1+ = augmented versions
                    pil_version = Image.fromarray(version)

                    if debug:
                        # Time each individual image version through the detection pipeline
                        img_start = perf_counter()

                    # Detect faces and get bounding boxes and probabilities
                    boxes, probs = mtcnn.detect(pil_version, landmarks=False)

                    if boxes is None or len(boxes) == 0:
                        if debug:
                            print(f"  [WARN] No face found in {img_path}, aug index {aug_idx}")
                        continue
                    #elif len(boxes) > 1:
                     #   print(f"  [WARN] Multiple faces found in {img_path}, using highest confidence")

                    # Find the index of the face with the highest confidence
                    best_idx = np.argmax(probs)

                    # Extract the face tensor for the best face
                    face_tensors = mtcnn.extract(pil_version, boxes[best_idx:best_idx+1], save_path=None)
                    if face_tensors is None or face_tensors[0] is None:
                        print(f"  [WARN] Failed to extract face from {img_path}")
                        continue
                    if debug:
                        # Record this image version's file size and detection time for the plot
                        # File size is the original image's size, time is for this augmented version
                        img_end = perf_counter()
                        all_file_sizes_kb.append(file_size_kb)
                        all_times_ms.append((img_end - img_start) * 1000)
                        all_person_labels.append(person)
                        aug_success += 1


                    # Store tensor alongside its source filename and augment index
                    # so we can trace any misclassification back to the original file
                    face_tensor_batch.append((face_tensors[0], fname, aug_idx))
                    total_imgs += 1
                if debug and aug_success != len(all_versions): 
                    print(f"[INFO] {img_path}: Processed only {aug_success}/{len(all_versions)} augments")

            # If try fails, print the error pointing to the image
            except Exception as e:
                print(f"  [ERR] {img_path}: {e}")

        if face_tensor_batch:
            # Extract just the tensors for batch embedding, preserving order
            tensors = [ft for ft, _, _ in face_tensor_batch]
            embeddings = get_embeddings_batch(resnet, tensors)
            if debug:
                # Build the pre-balance filename map for this person
                # Parallel to embeddings list. Index i in embeddings matches index i here
                filename_map[person] = [
                    {'file': fn, 'augment': aug_idx}
                    for _, fn, aug_idx in face_tensor_batch
            ]

        # check if embeddings from this person is not empty
        if embeddings:
            # dict key is person's name, embeddings is numpy arrays of features
            database[person] = embeddings
            print(f"  [INFO] {person}: {len(embeddings)} embeddings added to database.")
            if debug:
                end_person_time = perf_counter()
                ppl_embed_time[person][0] = end_person_time - start_person_time
                print(f"Processing time: {ppl_embed_time[person][0]:.2f} seconds")
                print(f"Average time per image for {person}: {ppl_embed_time[person][0]/len(embeddings):.2f} seconds")
                print(f"Average file size per image for {person}: {ppl_embed_time[person][1]/len(embeddings):.2f} KB")
        else:
            # If no valid, skip it
            if debug:
                end_person_time = perf_counter()
                ppl_embed_time[person][0] = end_person_time - start_person_time
                print(f"  [INFO] {person}: no valid faces found. Processing time: {ppl_embed_time[person][0]:.2f} seconds")
            else:
                print(f"  [INFO] {person}: no valid faces found.")

    # Balance all classes to the smallest class size so no person
    # has more influence than another in the evaluation
    min_count = min(len(embs) for embs in database.values())
    max_count = max(len(embs) for embs in database.values())

    # Warn if there is a large imbalance even after augmentation
    if max_count > min_count * 2:
        print(f"[WARN] Large imbalance detected — min={min_count}, max={max_count}. "
              f"Consider collecting more images for smaller classes.")

    print(f"\n[INFO] Balancing all classes to {min_count} embeddings (smallest class size)")

    # Sample indices rather than values directly so filename_map stays in sync
    # with the balanced database — both are sampled using the same indices
    balanced_database = {}
    if debug:
        balanced_filenames = {}
    for name, embs in database.items():
        # Randomly select which embedding indices to keep after balancing
        sampled_indices = random.sample(range(len(embs)), min_count)
        balanced_database[name] = [embs[i] for i in sampled_indices]
        if debug:
            # Apply same indices to filename map so index i still points to the right file
            balanced_filenames[name] = [filename_map[name][i] for i in sampled_indices]

    database = balanced_database

    # Pickle the database (compress it, basically)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(database, f)
    if debug:
        # Save filename map separately so evaluate.py can trace misclassifications
        # back to their source image file without modifying the main database format
        with open('filenames.pkl', 'wb') as f:
            pickle.dump(balanced_filenames, f)

    print(f"\n[INFO] Database built: {len(database)} identities, "
          f"{min_count} embeddings each, {total_imgs} total images processed.")
    if debug:
        totaltime_end = perf_counter()
        print(f"\n[INFO] Total processing time: {totaltime_end - totaltime:.2f} seconds")
        print(f"Average time per image overall: {(totaltime_end - totaltime)/total_imgs:.2f} seconds")
        print(f"Average file size per image overall: {sum(ppl_embed_time[person][1] for person in ppl_embed_time)/total_imgs:.2f} KB")
        # Plot processing time vs file size for every image processed
        plot_build_benchmark(all_file_sizes_kb, all_times_ms, all_person_labels)

    return database

def plot_build_benchmark(file_sizes_kb: list[float], times_ms: list[float], labels: list[str]):
    """Scatter plot of processing time vs file size for every image, coloured by person."""
    from datetime import datetime
    import matplotlib.pyplot as plt

    # Get unique people and assign each a colour from the tab10 colormap
    unique_people = sorted(set(labels))
    colours = plt.cm.tab10(np.linspace(0, 1, len(unique_people)))
    colour_map = dict(zip(unique_people, colours))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each person's images as a separate coloured scatter series
    for person in unique_people:
        # Get indices belonging to this person
        idxs = [i for i, l in enumerate(labels) if l == person]
        x = [file_sizes_kb[i] for i in idxs]
        y = [times_ms[i] for i in idxs]
        ax.scatter(x, y, color=colour_map[person], label=person, alpha=0.6)

    # Fit a logarithmic trend line by transforming x to log(x) before polyfit
    # This captures diminishing returns in processing time as file size grows
    log_x = np.log(file_sizes_kb)
    z = np.polyfit(log_x, times_ms, 1)  # degree 1 = log fit, degree 2 = log+quadratic
    p = np.poly1d(z)

    # Generate smooth curve for plotting — linspace gives evenly spaced x values
    # then evaluate p in log space but display in original file size space
    x_sorted = np.linspace(min(file_sizes_kb), max(file_sizes_kb), 200)
    y_trend = p(np.log(x_sorted))

    # Calculate R² to measure how well the log trend fits the data
    # R² closer to 1.0 means the log model explains the variance well
    r2 = r2_score(times_ms, p(np.log(file_sizes_kb)))

    ax.plot(x_sorted, y_trend, color='black', linestyle='--',
            label=f'Log Trend (R²={r2:.3f})')

    ax.set_xlabel('File Size (KB)')
    ax.set_ylabel('Processing Time (ms)')
    ax.set_title('Processing Time vs File Size per Image')
    # Place legend outside plot so it doesn't obscure data points
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    out_path = f'benchmark_build_.png'
    plt.tight_layout()

    plt.savefig(out_path, dpi=120)
    print(f"[INFO] Build benchmark plot saved to [{out_path}]")
    
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
    parser.add_argument('--no_aug', action='store_true',
                    help='Disable data augmentation')
    parser.add_argument('--debug', action='store_true',
                    help='Enable debug output (per-threshold person breakdown, verbose logging)')
    args = parser.parse_args()

    print(f"[DEBUG] Using device: {DEVICE}")
    print("[DEBUG] Loading models…")
    tsflw_mtcnn, model = load_models()
    print("[DEBUG] Models loaded.")

    if args.rebuild == 1 or not os.path.exists(EMBEDDINGS_FILE) == 1:
        # Args added are run here if there are any, or if there's an embeddings file 
        database = build_database(tsflw_mtcnn, model, args.db, debug=args.debug, no_aug=args.no_aug)
        print(f"[ARGS] Loaded {len(database)} faces")

    else:
        database = load_database()
        print(f"[NO_ARGS] Loaded {len(database)} faces")

    run_live(tsflw_mtcnn, model, database)
