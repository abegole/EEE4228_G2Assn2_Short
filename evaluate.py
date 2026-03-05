"""
evaluate.py
───────────
Offline evaluation of the face recognition system.

Performs leave-one-out cross-validation across the database images
and reports accuracy, precision, recall, F1, and a confusion matrix.

Usage:
    python evaluate.py [--threshold 0.65]
"""

import os, pickle, argparse
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              ConfusionMatrixDisplay)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EMBEDDINGS_FILE = 'embeddings.pkl'


def evaluate(threshold: float = 0.65):
    if not os.path.exists(EMBEDDINGS_FILE):
        print("[ERR] No embeddings file found. Run face_system.py --rebuild first.")
        return

    with open(EMBEDDINGS_FILE, 'rb') as f:
        database: dict = pickle.load(f)

    if len(database) < 2:
        print("[ERR] Need at least 2 identities in the database.")
        return

    from sklearn.metrics.pairwise import cosine_similarity

    names     = sorted(database.keys())
    y_true, y_pred = [], []

    for true_name in names:
        embs = database[true_name]
        for i, probe_emb in enumerate(embs):
            # Leave-one-out: build gallery without this sample
            gallery = {}
            for n, e in database.items():
                others = [x for j, x in enumerate(e) if not (n == true_name and j == i)]
                if others:
                    gallery[n] = np.mean(others, axis=0)

            # Recognise
            p_emb = probe_emb.reshape(1, -1)
            best_name, best_score = 'Unknown', 0.0
            for n, g_emb in gallery.items():
                sim = float(cosine_similarity(p_emb, g_emb.reshape(1, -1))[0][0])
                if sim > best_score:
                    best_score, best_name = sim, n
            if best_score < threshold:
                best_name = 'Unknown'

            y_true.append(true_name)
            y_pred.append(best_name)

    all_labels = names + ['Unknown']
    labels_present = sorted(set(y_true + y_pred))

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, labels=names,
                          average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, labels=names,
                       average='macro', zero_division=0)
    f1  = f1_score(y_true, y_pred, labels=names,
                   average='macro', zero_division=0)

    print("\n══════════════════════════════════════════")
    print(f"  Leave-One-Out Evaluation (threshold={threshold})")
    print("══════════════════════════════════════════")
    print(f"  Identities   : {len(names)}")
    print(f"  Total probes : {len(y_true)}")
    print(f"  Accuracy     : {acc*100:.2f}%")
    print(f"  Precision    : {pre*100:.2f}%  (macro, known only)")
    print(f"  Recall       : {rec*100:.2f}%  (macro, known only)")
    print(f"  F1 Score     : {f1*100:.2f}%  (macro, known only)")

    unknown_rate = sum(1 for p in y_pred if p == 'Unknown') / len(y_pred)
    print(f"  Unknown rate : {unknown_rate*100:.2f}%")
    print("══════════════════════════════════════════\n")

    # Per-person breakdown
    print(f"{'Name':<20} {'Total':>6} {'Correct':>8} {'Acc':>8}")
    print("─" * 46)
    for n in names:
        idxs   = [i for i, t in enumerate(y_true) if t == n]
        correct = sum(1 for i in idxs if y_pred[i] == n)
        total   = len(idxs)
        print(f"{n:<20} {total:>6} {correct:>8} {correct/total*100:>7.1f}%")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    fig, ax = plt.subplots(figsize=(max(6, len(labels_present)), max(5, len(labels_present))))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels_present)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(f'Confusion Matrix (threshold={threshold})', fontsize=13)
    plt.tight_layout()
    out_path = 'confusion_matrix.png'
    plt.savefig(out_path, dpi=120)
    print(f"\n[INFO] Confusion matrix saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.65)
    args = parser.parse_args()
    evaluate(args.threshold)
