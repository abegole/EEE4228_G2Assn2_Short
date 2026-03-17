"""
evaluate.py
───────────
Offline evaluation of the face recognition system.

Performs leave-one-out cross-validation across the database images
and reports accuracy, precision, recall, F1, and a confusion matrix.

Usage:
    python evaluate.py [--threshold 0.65]
"""
#test edit
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
            # 1. Leave-One-Out Cross Validation
            # We treat the current image (probe_emb) as an unknown face to be tested,
            # and train the system using the remaining images of this person.
            gallery = {}Could not find import of `numpy`, looked at search roots () and site package path ()Pyre2(missing-module-attribute)
Module[numpy]
            for n, e in database.items():
                # Exclude the current test image using its index 'i'
                others = [x for j, x in enumerate(e) if not (n == true_name and j == i)]
                if others:
                    # Calculate the average embedding (face features) for this person
                    gallery[n] = np.mean(others, axis=0)

            # 2. Recognition Phase
            # Compare the test image (probe_emb) against the calculated gallery
            p_emb = probe_emb.reshape(1, -1)
            best_name, best_score = 'Unknown', 0.0
            
            # Find the most similar identity in the gallery
            for n, g_emb in gallery.items():
                # Measure image similarity using Cosine Distance
                sim = float(cosine_similarity(p_emb, g_emb.reshape(1, -1))[0][0])
                if sim > best_score:
                    best_score, best_name = sim, n
                    
            # If the best matching score is lower than the threshold, mark as 'Unknown'
            if best_score < threshold:
                best_name = 'Unknown'

            # Save the true identity and the predicted identity for final evaluation
            y_true.append(true_name)
            y_pred.append(best_name)

    all_labels = names + ['Unknown']
    labels_present = sorted(set(y_true + y_pred))

    # 3. Calculate Evaluation Metrics (Accuracy, Precision, Recall, F1 Score)
    # These metrics determine how well the evaluation performed across all test subsets.
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

    # 4. Per-person breakdown
    # Displays individual accuracy statistics for each person in the database.
    print(f"{'Name':<20} {'Total':>6} {'Correct':>8} {'Acc':>8}")
    print("─" * 46)
    for n in names:
        idxs   = [i for i, t in enumerate(y_true) if t == n]
        correct = sum(1 for i in idxs if y_pred[i] == n)
        total   = len(idxs)
        print(f"{n:<20} {total:>6} {correct:>8} {correct/total*100:>7.1f}%")

    # 5. Confusion Matrix Plot
    # Generates a visual heatmap showing correct predictions and misclassifications (false positives/negatives).
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
