"""
face_gui.py  –  Tkinter GUI wrapper for the Face Recognition System
Embeds the live camera feed inside a window, shows identity log,
and provides controls for rebuilding the database.

Run:  python face_gui.py
"""

import os
import sys
import threading
import pickle
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────── Config ───────────────────────────────────────
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
DB_PATH         = 'face_database'
EMBEDDINGS_FILE = 'embeddings.pkl'
THRESHOLD       = 0.65
MIN_FACE_SIZE   = 40
IMG_SIZE        = 160
DISPLAY_W, DISPLAY_H = 800, 600   # camera canvas size


# ─────────────────────── Utility helpers ──────────────────────────────────
def get_embedding(resnet, face_tensor):
    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0).to(DEVICE))
    return emb.cpu().numpy()[0]


def recognise(embedding, mean_db):
    best_name, best_score = 'Unknown', 0.0
    emb = embedding.reshape(1, -1)
    for name, mean_emb in mean_db.items():
        sim = float(cosine_similarity(emb, mean_emb.reshape(1, -1))[0][0])
        if sim > best_score:
            best_score, best_name = sim, name
    if best_score < THRESHOLD:
        return 'Unknown', best_score
    return best_name, best_score


def build_database(mtcnn, resnet, db_dir, log_fn=print):
    database = {}
    if not os.path.isdir(db_dir):
        os.makedirs(db_dir)
        log_fn(f"Created {db_dir} – add sub-folders per person.")
        return database
    for person in sorted(os.listdir(db_dir)):
        pdir = os.path.join(db_dir, person)
        if not os.path.isdir(pdir):
            continue
        embs = []
        for f in os.listdir(pdir):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            try:
                img = Image.open(os.path.join(pdir, f)).convert('RGB')
                ft = mtcnn(img)
                if ft is None:
                    continue
                if ft.ndim == 4:
                    ft = ft[0]
                embs.append(get_embedding(resnet, ft))
            except Exception as e:
                log_fn(f"  WARN {f}: {e}")
        if embs:
            database[person] = embs
            log_fn(f"  {person}: {len(embs)} embeddings")
    with open(EMBEDDINGS_FILE, 'wb') as fp:
        pickle.dump(database, fp)
    log_fn(f"Database saved – {len(database)} identities.")
    return database


def load_database():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as fp:
            return pickle.load(fp)
    return {}


# ─────────────────────────── Main GUI ─────────────────────────────────────
class FaceRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Detection & Recognition System")
        self.resizable(False, False)
        self.configure(bg='#1e1e2e')

        # ── State ──
        self.running   = False
        self.mtcnn     = None
        self.resnet    = None
        self.database  = {}
        self.mean_db   = {}
        self._cap      = None
        self._frame_id = 0
        self._boxes_cache  = []
        self._labels_cache = []

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_models_async()

    # ─────────────── UI construction ──────────────────────────────────────
    def _build_ui(self):
        # Left panel – video
        left = tk.Frame(self, bg='#1e1e2e')
        left.grid(row=0, column=0, padx=10, pady=10)

        self.canvas = tk.Canvas(left, width=DISPLAY_W, height=DISPLAY_H,
                                bg='black', highlightthickness=0)
        self.canvas.pack()

        self.status_lbl = tk.Label(left, text="⏳  Loading models …",
                                   fg='#cdd6f4', bg='#1e1e2e',
                                   font=('Helvetica', 12))
        self.status_lbl.pack(pady=(6, 0))

        # Right panel – controls + log
        right = tk.Frame(self, bg='#1e1e2e', width=320)
        right.grid(row=0, column=1, padx=(0, 10), pady=10, sticky='n')

        tk.Label(right, text="Face Recognition", fg='#cba6f7', bg='#1e1e2e',
                 font=('Helvetica', 16, 'bold')).pack(pady=(0, 4))

        # DB info
        self.db_lbl = tk.Label(right, text="Database: 0 identities",
                               fg='#a6e3a1', bg='#1e1e2e', font=('Helvetica', 11))
        self.db_lbl.pack()
        self.device_lbl = tk.Label(right, text=f"Device: {DEVICE.upper()}",
                                   fg='#89b4fa', bg='#1e1e2e', font=('Helvetica', 10))
        self.device_lbl.pack(pady=(0, 10))

        # Threshold slider
        tk.Label(right, text="Similarity Threshold", fg='#cdd6f4',
                 bg='#1e1e2e', font=('Helvetica', 10)).pack()
        self.thresh_var = tk.DoubleVar(value=THRESHOLD)
        tk.Scale(right, from_=0.40, to=0.95, resolution=0.01,
                 orient='horizontal', variable=self.thresh_var,
                 bg='#313244', fg='#cdd6f4', highlightthickness=0,
                 troughcolor='#45475a', length=280).pack()

        # Buttons
        btn_style = dict(bg='#313244', fg='#cdd6f4', activebackground='#45475a',
                         activeforeground='white', relief='flat', width=28,
                         font=('Helvetica', 11), pady=6)

        self._start_btn = tk.Button(right, text="▶  Start Camera",
                                    command=self._start_camera, **btn_style)
        self._start_btn.pack(pady=(10, 4))

        self._stop_btn = tk.Button(right, text="⏹  Stop Camera",
                                   command=self._stop_camera,
                                   state='disabled', **btn_style)
        self._stop_btn.pack(pady=4)

        tk.Button(right, text="🗄  Rebuild Database",
                  command=self._rebuild_db_async, **btn_style).pack(pady=4)

        tk.Button(right, text="📁  Set DB Folder",
                  command=self._choose_db_folder, **btn_style).pack(pady=4)

        tk.Button(right, text="💾  Save Screenshot",
                  command=self._save_screenshot, **btn_style).pack(pady=4)

        # Log
        tk.Label(right, text="Event Log", fg='#cdd6f4', bg='#1e1e2e',
                 font=('Helvetica', 10, 'bold')).pack(pady=(14, 2))
        self.log_text = tk.Text(right, height=16, width=38, bg='#181825',
                                fg='#cdd6f4', font=('Courier', 9),
                                relief='flat', state='disabled',
                                insertbackground='white')
        scroll = ttk.Scrollbar(right, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side='left', pady=4)
        scroll.pack(side='left', fill='y', pady=4)

    # ─────────────── Logging ──────────────────────────────────────────────
    def _log(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        self.log_text.configure(state='normal')
        self.log_text.insert('end', f"[{ts}] {msg}\n")
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    # ─────────────── Model loading ────────────────────────────────────────
    def _load_models_async(self):
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        self._log("Loading MTCNN …")
        self.mtcnn = MTCNN(image_size=IMG_SIZE, margin=20,
                           min_face_size=MIN_FACE_SIZE,
                           thresholds=[0.6, 0.7, 0.7], factor=0.709,
                           post_process=True, keep_all=True, device=DEVICE)
        self._log("Loading FaceNet (VGGFace2) …")
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        self._log("Models ready.")

        if os.path.exists(EMBEDDINGS_FILE):
            self.database = load_database()
            self.mean_db  = {n: np.mean(e, axis=0) for n, e in self.database.items()}
            self._log(f"DB loaded: {len(self.database)} identities")
        else:
            self._log("No database found – click 'Rebuild Database'.")

        self.after(0, self._update_labels)
        self.after(0, lambda: self.status_lbl.config(text="✅  Models loaded"))

    # ─────────────── Camera control ───────────────────────────────────────
    def _start_camera(self):
        if self.running or self.resnet is None:
            return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return
        self.running = True
        self._start_btn.config(state='disabled')
        self._stop_btn.config(state='normal')
        self.status_lbl.config(text="🎥  Camera running")
        self._log("Camera started.")
        self._process_frame()

    def _stop_camera(self):
        self.running = False
        if self._cap:
            self._cap.release()
        self._start_btn.config(state='normal')
        self._stop_btn.config(state='disabled')
        self.status_lbl.config(text="⏹  Camera stopped")
        self._log("Camera stopped.")

    # ─────────────── Per-frame processing ────────────────────────────────
    def _process_frame(self):
        if not self.running:
            return
        ret, frame = self._cap.read()
        if not ret:
            self.after(33, self._process_frame)
            return

        self._frame_id += 1
        if self._frame_id % 3 == 0:
            self._detect_and_recognise(frame)

        annotated = self._draw_annotations(frame.copy())
        self._show_frame(annotated)
        self.after(15, self._process_frame)

    def _detect_and_recognise(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        boxes, probs, _ = self.mtcnn.detect(pil, landmarks=True)
        self._boxes_cache, self._labels_cache = [], []
        if boxes is None:
            return
        face_tensors = self.mtcnn.extract(pil, boxes, save_path=None)
        thresh = self.thresh_var.get()
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < 0.90 or face_tensors is None or face_tensors[i] is None:
                continue
            emb = get_embedding(self.resnet, face_tensors[i])
            # inline recognise with current threshold
            best_name, best_score = 'Unknown', 0.0
            e = emb.reshape(1, -1)
            for name, mean_emb in self.mean_db.items():
                sim = float(cosine_similarity(e, mean_emb.reshape(1, -1))[0][0])
                if sim > best_score:
                    best_score, best_name = sim, name
            if best_score < thresh:
                best_name = 'Unknown'
            self._boxes_cache.append([int(v) for v in box])
            self._labels_cache.append((best_name, best_score))

    def _draw_annotations(self, frame):
        for (x1, y1, x2, y2), (name, score) in zip(self._boxes_cache, self._labels_cache):
            color = (0, 200, 0) if name != 'Unknown' else (0, 0, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name}  {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        # HUD
        cv2.putText(frame, f"DB: {len(self.database)} identities | thresh={self.thresh_var.get():.2f}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        return frame

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((DISPLAY_W, DISPLAY_H))
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)

    # ─────────────── Database controls ────────────────────────────────────
    def _rebuild_db_async(self):
        if self.resnet is None:
            messagebox.showwarning("Wait", "Models are still loading.")
            return
        self._log("Rebuilding database …")
        threading.Thread(target=self._rebuild_db, daemon=True).start()

    def _rebuild_db(self):
        self.database = build_database(self.mtcnn, self.resnet, DB_PATH, self._log)
        self.mean_db  = {n: np.mean(e, axis=0) for n, e in self.database.items()}
        self.after(0, self._update_labels)

    def _choose_db_folder(self):
        global DB_PATH
        folder = filedialog.askdirectory(title="Select database root folder")
        if folder:
            DB_PATH = folder
            self._log(f"DB folder set to: {folder}")

    def _update_labels(self):
        self.db_lbl.config(text=f"Database: {len(self.database)} identities")

    def _save_screenshot(self):
        if not self._boxes_cache and not self.running:
            messagebox.showinfo("Info", "Start the camera first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            ret, frame = self._cap.read() if self._cap else (False, None)
            if ret:
                annotated = self._draw_annotations(frame)
                cv2.imwrite(path, annotated)
                self._log(f"Screenshot saved: {path}")

    # ─────────────── Cleanup ──────────────────────────────────────────────
    def _on_close(self):
        self.running = False
        if self._cap:
            self._cap.release()
        self.destroy()


if __name__ == '__main__':
    app = FaceRecognitionApp()
    app.mainloop()
