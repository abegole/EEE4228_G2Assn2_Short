"""
make_figures.py
───────────────
Generates figures for the Theory/Method sections of the report/presentation.

Figures produced:
  1. augmentation_grid.png   — Data augmentation pipeline visualization
  2. pipeline_diagram.png    — System pipeline block diagram
  3. cosine_concept.png      — Cosine similarity concept illustration
  4. tsne_embeddings.png     — t-SNE visualization of 512-D embeddings
"""

import os, pickle, random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from sklearn.manifold import TSNE

# ───────────────────────────────────────────────
# Shared constants
# ───────────────────────────────────────────────
DB_PATH        = 'face_database'
EMBEDDINGS_FILE = 'embeddings.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 180
MIN_FACE_SIZE = 60

def augment_image(img_np):
    """Returns list of augmented images (same as face_system.py)."""
    augmented = []
    h, w = img_np.shape[:2]
    center = (w // 2, h // 2)
    augmented.append(cv2.flip(img_np, 1))
    augmented.append(cv2.convertScaleAbs(img_np, alpha=1.2, beta=30))
    augmented.append(cv2.convertScaleAbs(img_np, alpha=0.8, beta=-30))
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1)
        augmented.append(cv2.warpAffine(img_np, M, (w, h)))
    return augmented


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Augmentation Grid
# ══════════════════════════════════════════════════════════════════════
def make_augmentation_grid():
    print("[FIG 1] Generating augmentation_grid.png ...")

    # Pick the first valid image from the database
    img_np = None
    person_name = None
    for person in sorted(os.listdir(DB_PATH)):
        person_dir = os.path.join(DB_PATH, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in os.listdir(person_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(person_dir, fname)
                img = Image.open(path).convert('RGB')
                img_np = np.array(img)
                person_name = person
                break
        if img_np is not None:
            break

    if img_np is None:
        print("  [WARN] No image found in face_database. Skipping Figure 1.")
        return

    augmented = augment_image(img_np)
    all_images = [img_np] + augmented  # original + 5 augmented

    titles = [
        "Original",
        "Horizontal\nFlip",
        "Brightness+\n(α=1.2, β=+30)",
        "Brightness-\n(α=0.8, β=−30)",
        "Rotation\n−10°",
        "Rotation\n+10°",
    ]

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, img, title in zip(axes, all_images, titles):
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
        ax.axis('off')
        # Highlight original with a colored border
        if title == "Original":
            for spine in ax.spines.values():
                spine.set_edgecolor('#00d4ff')
                spine.set_linewidth(3)
                spine.set_visible(True)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor('#ff6b6b')
                spine.set_linewidth(2)
                spine.set_visible(True)

    fig.suptitle(
        "Data Augmentation Pipeline  ·  1 image → 6 embeddings",
        color='white', fontsize=14, fontweight='bold', y=1.02
    )

    # Arrow between images — add text label underneath
    fig.text(0.5, -0.04,
             "Applied during database construction (build_database) to improve robustness against lighting and pose variation.",
             ha='center', va='center', color='#aaaaaa', fontsize=9)

    plt.tight_layout()
    plt.savefig('augmentation_grid.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("  [DONE] augmentation_grid.png saved.")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Cosine Similarity Concept
# ══════════════════════════════════════════════════════════════════════
def make_cosine_concept():
    print("[FIG 2] Generating cosine_concept.png ...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')

    colors = {'bg': '#1a1a2e', 'text': 'white', 'grid': '#2a2a4e',
              'same': '#00d4ff', 'diff': '#ff6b6b', 'arc': '#f0e68c',
              'neutral': '#aaaaaa'}

    for ax in axes:
        ax.set_facecolor(colors['bg'])
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-0.1, 1.3)
        ax.set_aspect('equal')
        ax.grid(True, color=colors['grid'], alpha=0.3)
        ax.spines[:].set_color(colors['grid'])
        ax.tick_params(colors=colors['neutral'])
        # Origin
        ax.plot(0, 0, 'o', color='white', ms=5, zorder=5)

    # ──── Left: same person (small angle) ────
    ax = axes[0]
    ax.set_title("Same Person  →  High Cosine Similarity", color=colors['text'],
                 fontsize=11, fontweight='bold', pad=10)
    v1 = np.array([1.0, 0.3])
    v2 = np.array([0.9, 0.5])
    for v, col, lbl in [(v1, colors['same'], 'Embedding A'), (v2, '#7fffd4', 'Embedding B')]:
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        ax.text(v[0]+0.03, v[1]+0.03, lbl, color=col, fontsize=10)
    # Arc for angle
    theta1 = np.degrees(np.arctan2(v1[1], v1[0]))
    theta2 = np.degrees(np.arctan2(v2[1], v2[0]))
    arc = matplotlib.patches.Arc((0, 0), 0.5, 0.5, angle=0,
                                  theta1=min(theta1, theta2),
                                  theta2=max(theta1, theta2),
                                  color=colors['arc'], lw=2)
    ax.add_patch(arc)
    cos_val = float(np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)))
    ax.text(0.28, 0.12, f"θ ≈ {np.degrees(np.arccos(cos_val)):.0f}°",
            color=colors['arc'], fontsize=12)
    ax.text(0.02, 1.15, f"cos(θ) ≈ {cos_val:.2f}  ✓  (≥ 0.7 → Recognized)",
            color='#90EE90', fontsize=11, fontweight='bold')

    # ──── Right: different person (large angle) ────
    ax = axes[1]
    ax.set_title("Different Person  →  Low Cosine Similarity", color=colors['text'],
                 fontsize=11, fontweight='bold', pad=10)
    v3 = np.array([1.0, 0.2])
    v4 = np.array([0.2, 1.0])
    for v, col, lbl in [(v3, colors['same'], 'Probe'), (v4, colors['diff'], 'Database entry')]:
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        ax.text(v[0]+0.03, v[1]+0.03, lbl, color=col, fontsize=10)
    theta3 = np.degrees(np.arctan2(v3[1], v3[0]))
    theta4 = np.degrees(np.arctan2(v4[1], v4[0]))
    arc2 = matplotlib.patches.Arc((0, 0), 0.5, 0.5, angle=0,
                                   theta1=min(theta3, theta4),
                                   theta2=max(theta3, theta4),
                                   color=colors['arc'], lw=2)
    ax.add_patch(arc2)
    cos_val2 = float(np.dot(v3/np.linalg.norm(v3), v4/np.linalg.norm(v4)))
    ax.text(0.22, 0.52, f"θ ≈ {np.degrees(np.arccos(cos_val2)):.0f}°",
            color=colors['arc'], fontsize=12)
    ax.text(0.02, 1.15, f"cos(θ) ≈ {cos_val2:.2f}  ✗  (< 0.7 → Unknown)",
            color='#FF7F7F', fontsize=11, fontweight='bold')

    fig.suptitle("Cosine Similarity: Measuring Angle Between 512-D Embedding Vectors",
                 color='white', fontsize=13, fontweight='bold', y=1.02)
    fig.text(0.5, -0.04,
             "Formula:  cos(θ) = (A · B) / (||A|| × ||B||)   |   Threshold = 0.7",
             ha='center', color='#aaaaaa', fontsize=10)

    plt.tight_layout()
    plt.savefig('cosine_concept.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("  [DONE] cosine_concept.png saved.")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: t-SNE Embedding Visualization
# ══════════════════════════════════════════════════════════════════════
def make_tsne():
    print("[FIG 3] Generating tsne_embeddings.png ...")

    if not os.path.exists(EMBEDDINGS_FILE):
        print("  [WARN] embeddings.pkl not found. Skipping t-SNE.")
        return

    with open(EMBEDDINGS_FILE, 'rb') as f:
        database = pickle.load(f)

    # Balance
    min_count = min(len(v) for v in database.values())
    database = {n: random.sample(e, min_count) for n, e in database.items()}

    all_embs, all_labels = [], []
    for name, embs in database.items():
        all_embs.extend([e if e.ndim == 1 else e.flatten() for e in embs])
        all_labels.extend([name] * len(embs))

    X = np.array(all_embs)
    print(f"  Running t-SNE on {len(X)} embeddings ...")
    tsne = TSNE(n_components=2, perplexity=min(15, len(X)//2 - 1),
                random_state=42, max_iter=1000)
    coords = tsne.fit_transform(X)

    # Color palette
    names = sorted(set(all_labels))
    palette = plt.cm.tab10(np.linspace(0, 1, len(names)))
    color_map = dict(zip(names, palette))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.grid(True, color='#2a2a4e', alpha=0.5)
    ax.spines[:].set_color('#2a2a4e')
    ax.tick_params(colors='#aaaaaa')

    for name in names:
        mask = np.array([l == name for l in all_labels])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color_map[name]], label=name,
                   s=60, alpha=0.85, edgecolors='white', linewidths=0.4)

        # Centroid label
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.text(cx, cy, name.split()[0], ha='center', va='center',
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color_map[name], alpha=0.7))

    ax.set_title("t-SNE Visualization of 512-D Face Embeddings",
                 color='white', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("t-SNE Dimension 1", color='#aaaaaa', fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", color='#aaaaaa', fontsize=11)
    legend = ax.legend(loc='upper right', framealpha=0.3, facecolor='#1a1a2e',
                       labelcolor='white', fontsize=9)

    fig.text(0.5, 0.01,
             "Each point = one 512-D embedding. Clusters confirm intra-class compactness and inter-class separation.",
             ha='center', color='#aaaaaa', fontsize=9)

    plt.tight_layout()
    plt.savefig('tsne_embeddings.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("  [DONE] tsne_embeddings.png saved.")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: MTCNN 3-Stage Pipeline
# ══════════════════════════════════════════════════════════════════════
def make_mtcnn_diagram():
    print("[FIG 4] Generating mtcnn_pipeline.png ...")

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')

    stage_colors = ['#2d6a4f', '#1d3557', '#6d2b91']
    stages = [
        ("Input\nImage", None, '#333355'),
        ("P-Net\n(Proposal)", "Rapidly scans at\nmultiple scales\n→ ~1000+ candidates", stage_colors[0]),
        ("R-Net\n(Refine)", "Filters false positives\n→ ~100 candidates", stage_colors[1]),
        ("O-Net\n(Output)", "Final refinement\n+ landmarks\n→ Final bounding boxes", stage_colors[2]),
        ("Cropped\nFace", None, '#553355'),
    ]

    xs = [0.7, 3.0, 5.8, 8.6, 11.4]
    w, h_box = 2.0, 2.8

    for i, (title, desc, color) in enumerate(stages):
        x = xs[i]
        rect = mpatches.FancyBboxPatch(
            (x, 1.1), w, h_box,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor='white', linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, 1.1 + h_box - 0.45, title,
                ha='center', va='center', color='white',
                fontsize=11, fontweight='bold')
        if desc:
            ax.text(x + w/2, 1.1 + h_box/2 - 0.3, desc,
                    ha='center', va='center', color='#cccccc',
                    fontsize=8.5)

    # Arrows between boxes + candidate count labels
    arrow_labels = ["~1000+\ncandidates", "~100\ncandidates", "Final\nbox"]
    arrow_xs = [(xs[i] + w, xs[i+1]) for i in range(len(xs)-1)]
    for j, (x_start, x_end) in enumerate(arrow_xs):
        ax.annotate("", xy=(x_end, 2.55), xytext=(x_start, 2.55),
                    arrowprops=dict(arrowstyle="-|>", color='#aaaaaa', lw=2))
        if j < len(arrow_labels):
            mx = (x_start + x_end) / 2
            ax.text(mx, 3.15, arrow_labels[j],
                    ha='center', va='center', color='#f0e68c', fontsize=8)

    ax.set_title("MTCNN: Multi-Task Cascaded Convolutional Networks — 3-Stage Face Detection",
                 color='white', fontsize=13, fontweight='bold', y=0.97)
    ax.text(7, 0.4,
            "Each stage progressively filters candidates, achieving high accuracy with low computational cost.",
            ha='center', color='#aaaaaa', fontsize=9)

    plt.tight_layout()
    plt.savefig('mtcnn_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("  [DONE] mtcnn_pipeline.png saved.")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    make_augmentation_grid()
    make_cosine_concept()
    make_mtcnn_diagram()
    make_tsne()
    print("\n[ALL DONE] All figures generated.")
