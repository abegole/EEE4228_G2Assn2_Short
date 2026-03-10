# Face Detection & Recognition System
# MTCNN + FaceNet (InceptionResnetV1 / VGGFace2)

# Basic Usage

```bash
pip install -r requirements.txt

python capture_faces.py --name "test1" --count 15
python capture_faces.py --name "test2"   --count 15
# etc...

# build then run
python face_system.py --rebuild
python face_system.py
python face_gui.py
```

---

### Directory Layout

```
face_recognition_system/
├── face_system.py          # Core engine (CLI)
├── face_gui.py             # Tkinter GUI (live demo)
├── capture_faces.py        # Webcam image capture helper
├── evaluate.py             # Leave-one-out accuracy / F1 evaluation
├── requirements.txt
├── embeddings.pkl          # Generated after --rebuild
└── face_database/          # Created by capture_faces.py
    ├── Alice/
    │   ├── Alice_001.jpg
    │   └── …
    └── Bob/
        └── …
```

---

### Architecture

#### 1 – Face Detection: MTCNN
| Stage | Purpose |
|-------|---------|
| P-Net | Proposal network – fast sliding-window candidate generation |
| R-Net | Refinement – rejects most false positives |
| O-Net | Output – precise bounding box + 5 facial landmarks |

MTCNN is used **as-is** (off-the-shelf) from `facenet-pytorch`.

#### 2 – Face Recognition: FaceNet
* Backbone: **InceptionResnetV1** pretrained on **VGGFace2** (~3.3M images, ~9k identities)
* Produces a **512-dimensional L2-normalised embedding** for each aligned face crop
* Recognition: **cosine similarity** between probe embedding and per-person mean gallery embedding
* Threshold τ (default 0.65) separates known/unknown

#### Recognition Decision Rule
```
similarity = cosine_sim(probe_emb, gallery_mean_emb)
if max(similarity over all persons) ≥ τ  →  predicted label = best match
else                                      →  "Unknown"
```

---

### Performance Tips
| Situation | Recommendation |
|-----------|---------------|
| Too many "Unknown" for real members | Lower τ (e.g. 0.55) |
| Too many false positives | Raise τ (e.g. 0.70) |
| Poor accuracy | Capture more diverse images (angles, lighting) |
| Slow on CPU | Reduce camera resolution; detect every N=5 frames |

---

### Key Controls (GUI)
| Control | Action |
|---------|--------|
| Similarity Threshold slider | Adjust recognition strictness live |
| Rebuild Database | Re-processes all images in `face_database/` |
| Set DB Folder | Point to a different folder |
| Save Screenshot | Save current annotated frame |

---

### Files Description
| File | Role |
|------|------|
| `face_system.py` | Loads models, builds DB, runs OpenCV live loop |
| `face_gui.py` | Tkinter GUI with embedded video feed |
| `capture_faces.py` | Webcam capture with manual / auto mode |
| `evaluate.py` | Leave-one-out cross-validation + confusion matrix |
