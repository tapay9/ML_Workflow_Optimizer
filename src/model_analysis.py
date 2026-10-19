import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
TARGET = "is_canceled"

# Model registry (SQLite): find the project root so `src` is importable from a notebook,
# a script run inside src/, or the project root.
_here = Path.cwd().resolve()
ROOT = next(p for p in [_here, *_here.parents] if (p / "src" / "db.py").exists())
sys.path.insert(0, str(ROOT))
from src.model_registry import DEFAULT_MODEL_NAME, build_input_schema, register_model


# Data + models
LEAKY = ["reservation_status", "reservation_status_date"]
POST_BOOKING = ["assigned_room_type"]
ID_LIKE = ["agent", "company"]


def prepare_raw(df):
    """
    Prepare hotel booking data for model training.

    Creates engineered features:
        total_guests = adults + children + babies
        total_nights = stays_in_weekend_nights + stays_in_week_nights

    After creating those engineered features, the original source
    columns are removed so the model learns from the engineered
    features instead of redundant versions of the same information.
    """

    # -------------------------------------------------
    # Remove leakage / unusable columns
    # -------------------------------------------------

    drop = [
        c
        for c in LEAKY + POST_BOOKING + ID_LIKE
        if c in df.columns
    ]

    if drop:
        print(
            f"  Raw file detected -> "
            f"dropping leaky/unusable columns: {drop}"
        )

        df = df.drop(columns=drop)


    # -------------------------------------------------
    # Create hotel_type
    # -------------------------------------------------

    if "hotel" in df.columns:

        df["hotel_type"] = (
            df["hotel"]
            .str.split(" - ")
            .str[0]
        )

        df = df.drop(columns="hotel")


    # -------------------------------------------------
    # Feature Engineering: Total Guests
    # -------------------------------------------------

    guest_columns = [
        "adults",
        "children",
        "babies"
    ]

    if (
        "total_guests" not in df.columns
        and all(c in df.columns for c in guest_columns)
    ):

        df["total_guests"] = (
            df[guest_columns]
            .fillna(0)
            .sum(axis=1)
        )


    # -------------------------------------------------
    # Feature Engineering: Total Nights
    # -------------------------------------------------

    night_columns = [
        "stays_in_weekend_nights",
        "stays_in_week_nights"
    ]

    if (
        "total_nights" not in df.columns
        and all(c in df.columns for c in night_columns)
    ):

        df["total_nights"] = (
            df["stays_in_weekend_nights"]
            +
            df["stays_in_week_nights"]
        )


    # -------------------------------------------------
    # Remove original columns used to create the
    # engineered features
    # -------------------------------------------------

    engineered_source_columns = [
        "adults",
        "children",
        "babies",
        "stays_in_weekend_nights",
        "stays_in_week_nights"
    ]

    columns_to_drop = [
        c
        for c in engineered_source_columns
        if c in df.columns
    ]

    if columns_to_drop:

        print(
            "  Dropping source columns replaced by "
            f"engineered features: {columns_to_drop}"
        )

        df = df.drop(
            columns=columns_to_drop
        )


    # -------------------------------------------------
    # Remove constant columns
    # -------------------------------------------------

    const = [
        c
        for c in df.columns
        if c != TARGET
        and df[c].nunique(dropna=False) <= 1
    ]

    if const:

        print(
            f"  Dropping constant columns: {const}"
        )

        df = df.drop(columns=const)


    # -------------------------------------------------
    # Fill remaining numeric missing values
    # -------------------------------------------------

    num = df.select_dtypes(
        "number"
    ).columns

    if df[num].isna().any().any():

        print(
            "  Filling numeric NaNs with the median."
        )

        df[num] = df[num].fillna(
            df[num].median()
        )


    return df


def load_data(path, max_rows, seed):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")
    df = prepare_raw(df)
    dup = df.duplicated().mean()
    print(f"Duplicate rows after prep: {dup:.1%}")
    if dup > 0.05:
        print("  ! Many exact duplicates: identical rows can land in both train and test,\n"
              "    which inflates Decision Tree / KNN scores. Consider df.drop_duplicates().")
    df = df.drop(columns=["arrival_date_month"], errors="ignore")
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)

    cat = [c for c in df.columns if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])]
    parts = [df.drop(columns=cat + [TARGET])]
    groups = {c: c for c in parts[0].columns}           # dummy column -> original column
    for col in cat:
        d = pd.get_dummies(df[col], prefix=col, dtype=int)
        parts.append(d)
        groups.update({c: col for c in d.columns})
    X = pd.concat(parts, axis=1)
    y = df[TARGET].astype(int)
    is_dummy = np.array([groups[c] != c for c in X.columns])
    print(f"Encoded to {X.shape[1]} features from {len(set(groups.values()))} original columns | "
          f"cancel rate: {y.mean():.1%}")
    return X, y, groups, is_dummy


def make_models(seed, skip_knn):
    m = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=seed
            )
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            random_state=seed
        ),
    }

    if not skip_knn:
        m["KNN"] = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=41,
                weights="distance",
                p=1,
                n_jobs=-1
            )
        )

    return m



def score_all(y, pred, proba):
    return dict(accuracy=accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average="macro"),
                f1_canceled=f1_score(y, pred),
                roc_auc=roc_auc_score(y, proba))



# Part 1: best train/test split
def split_sweep(X, y, sizes, seeds, skip_knn):
    rows = []
    for ts in sizes:
        for s in seeds:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=s, stratify=y)
            for name, m in make_models(s, skip_knn).items():
                m.fit(Xtr, ytr)
                rows.append(dict(test_size=ts, seed=s, model=name,
                                 **score_all(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])))
        print(f"  test_size={ts:.2f} done")
    return pd.DataFrame(rows)


def pick_split(sweep, metric="macro_f1"):
    """Best mean score, then the LARGEST test_size within 1 standard error of it
    (bigger test set = more trustworthy estimate when the scores are statistically tied)."""
    per_seed = sweep.groupby(["test_size", "seed"])[metric].mean().reset_index()
    agg = per_seed.groupby("test_size")[metric].agg(["mean", "sem"]).fillna(0)
    best_ts = agg["mean"].idxmax()
    cutoff = agg.loc[best_ts, "mean"] - agg.loc[best_ts, "sem"]
    chosen = agg[agg["mean"] >= cutoff].index.max()
    return best_ts, chosen, agg


def learning_curve(X, y, skip_knn, seed, fracs=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """Fixed 20% holdout; vary only how much training data is used.
    Shows whether more training data is still buying accuracy (the real reason a
    smaller test_size can look 'better')."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    rows = []
    for f in fracs:
        n = int(len(Xtr) * f)
        Xs, ys = Xtr.iloc[:n], ytr.iloc[:n]
        for name, m in make_models(seed, skip_knn).items():
            m.fit(Xs, ys)
            rows.append(dict(train_frac=f, n_train=n, model=name,
                             **score_all(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])))
    return pd.DataFrame(rows)



# Part 2: classification-report evaluation
def evaluate_models(Xtr, Xte, ytr, yte, skip_knn, seed, out):
    fitted, tables, summary = {}, [], []
    for name, m in make_models(seed, skip_knn).items():
        m.fit(Xtr, ytr)
        pred, proba = m.predict(Xte), m.predict_proba(Xte)[:, 1]
        fitted[name] = (m, pred, proba)
        rep = classification_report(yte, pred, target_names=["not_canceled", "canceled"],
                                    output_dict=True)
        tables.append(pd.DataFrame(rep).T.assign(model=name))
        train_acc = accuracy_score(ytr, m.predict(Xtr))
        s = score_all(yte, pred, proba)
        summary.append(dict(model=name, train_acc=train_acc, test_acc=s["accuracy"],
                            overfit_gap=train_acc - s["accuracy"],
                            precision_canceled=rep["canceled"]["precision"],
                            recall_canceled=rep["canceled"]["recall"],
                            f1_canceled=rep["canceled"]["f1-score"],
                            macro_f1=s["macro_f1"], roc_auc=s["roc_auc"],
                            pr_auc=average_precision_score(yte, proba)))
    pd.concat(tables).to_csv(out / "classification_reports.csv")
    summary = pd.DataFrame(summary).set_index("model")
    summary.to_csv(out / "model_comparison.csv")

    print("\n" + summary.round(4).to_string())
    print("\nInterpretation of the classification reports:")
    for name, r in summary.iterrows():
        p, rc = r.precision_canceled, r.recall_canceled
        if rc < p - 0.05:
            beh = "misses real cancellations (recall < precision)"
        elif p < rc - 0.05:
            beh = "over-flags cancellations (precision < recall) -> false alarms"
        else:
            beh = "precision and recall are balanced"
        flag = "  | OVERFITTING (train-test gap > 5 pts)" if r.overfit_gap > 0.05 else ""
        print(f"  {name}: {beh}{flag}")

    # confusion matrices
    fig, axes = plt.subplots(1, len(fitted), figsize=(4.5 * len(fitted), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, (_, pred, _)) in zip(axes, fitted.items()):
        cm = confusion_matrix(yte, pred)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=130)
    plt.close(fig)

    # ROC + PR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (_, _, proba) in fitted.items():
        fpr, tpr, _ = roc_curve(yte, proba)
        pr, rc, _ = precision_recall_curve(yte, proba)
        a1.plot(fpr, tpr, label=f"{name} ({roc_auc_score(yte, proba):.3f})")
        a2.plot(rc, pr, label=f"{name} ({average_precision_score(yte, proba):.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=0.8)
    a1.set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")
    a2.set(title="Precision-Recall (canceled)", xlabel="Recall", ylabel="Precision")
    a1.legend(); a2.legend()
    fig.tight_layout()
    fig.savefig(out / "roc_pr_curves.png", dpi=130)
    plt.close(fig)
    return fitted, summary


# Part 3: which columns matter?
def group_sum(series, groups):
    return series.groupby(series.index.map(groups)).sum()


def grouped_permutation(model, X, y, groups, n_repeats, seed, max_rows=10000):
    """Permute all dummies of one original column TOGETHER (permuting a single
    one-hot column in isolation understates the real column's importance)."""
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    members = {}
    for c, g in groups.items():
        members.setdefault(g, []).append(c)
    Xp, out = X.copy(), {}
    for g, cols in members.items():
        orig = X[cols].to_numpy()
        drops = []
        for _ in range(n_repeats):
            Xp[cols] = orig[rng.permutation(len(X))]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        Xp[cols] = orig
        out[g] = (np.mean(drops), np.std(drops))
    return pd.DataFrame(out, index=["auc_drop", "auc_drop_std"]).T


def feature_significance(fitted, X, Xte, yte, y, groups, is_dummy, n_repeats, seed, out):
    cols = X.columns
    methods = {}

    # (a) Logistic regression: |standardized coefficient|  (inputs were standardized)
    lr = fitted["Logistic Regression"][0]
    coef = pd.Series(lr[-1].coef_[0], index=cols)
    methods["LR |coef|"] = group_sum(coef.abs(), groups)
    print("\nLogistic-regression direction (top dummies/columns by |coef|, + = more cancellations):")
    top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:10]
    for c, v in top.items():
        print(f"  {c:<45s} coef={v:+.3f}  odds x{np.exp(v):.2f} per +1 SD")

    # (b) Decision-tree impurity importance
    dt = fitted["Decision Tree"][0]
    methods["Tree importance"] = group_sum(pd.Series(dt.feature_importances_, index=cols), groups)

    # (c) Grouped permutation importance on held-out data (model-agnostic, honest)
    perm = {}
    for name in ("Logistic Regression", "Decision Tree"):
        perm[name] = grouped_permutation(fitted[name][0], Xte, yte, groups, n_repeats, seed)
        methods[f"Permutation ({name[:8].strip()})"] = perm[name]["auc_drop"].clip(lower=0)

    # (d) Mutual information (model-free, univariate) on a subsample
    n = min(30000, len(X))
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    mi = pd.Series(mutual_info_classif(X.iloc[idx], y.iloc[idx], discrete_features=is_dummy,
                                       random_state=seed), index=cols)
    methods["Mutual information"] = mi.groupby(mi.index.map(groups)).max()

    M = pd.DataFrame(methods)
    ranks = M.rank(ascending=False)
    norm = M / M.max()
    res = pd.DataFrame({"mean_rank": ranks.mean(axis=1), "mean_norm_score": norm.mean(axis=1)})
    res = res.join(M).sort_values("mean_rank")
    # "significant" = permuting the column hurts held-out AUC by more than 2 std, for either model
    sig = pd.DataFrame({n: perm[n]["auc_drop"] > 2 * perm[n]["auc_drop_std"] for n in perm})
    res["significant_perm"] = sig.any(axis=1)
    res.to_csv(out / "feature_significance.csv")

    k = 10
    print(f"\nMOST significant columns (consensus of {M.shape[1]} methods):")
    print(res.head(k)[["mean_rank", "mean_norm_score"]].round(3).to_string())
    print(f"\nLEAST significant columns:")
    print(res.tail(k)[["mean_rank", "mean_norm_score"]].round(3).to_string())
    print(f"\nColumns with no detectable held-out effect on either model: "
          f"{int((~res.significant_perm).sum())} of {len(res)}")

    # plot
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(res) + 1.5))
    r = res.sort_values("mean_norm_score")
    ax.barh(r.index, r.mean_norm_score, color=np.where(r.significant_perm, "#2a6fb0", "#b8c4d0"))
    ax.set(title="Consensus feature significance (blue = permutation-significant)",
           xlabel="Mean normalized importance across methods")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "feature_significance.png", dpi=130)
    plt.close(fig)
    return res


def ablation(res, X, groups, Xtr, Xte, ytr, yte, seed, ks=(5, 10, 15)):
    """Validate the ranking: dropping the 'least significant' columns should cost
    ~nothing; dropping the 'most significant' ones should hurt."""
    print("\nAblation check (Logistic Regression & Decision Tree, held-out ROC-AUC):")
    ordered = list(res.index)

    def run(drop_groups):
        keep = [c for c in X.columns if groups[c] not in set(drop_groups)]
        out = {}
        for name, m in make_models(seed, True).items():
            m.fit(Xtr[keep], ytr)
            out[name] = roc_auc_score(yte, m.predict_proba(Xte[keep])[:, 1])
        return out

    rows = [dict(dropped="none (baseline)", **run([]))]
    for k in ks:
        if k < len(ordered):
            rows.append(dict(dropped=f"bottom {k}", **run(ordered[-k:])))
            rows.append(dict(dropped=f"top {k}", **run(ordered[:k])))
    tbl = pd.DataFrame(rows).set_index("dropped")
    print(tbl.round(4).to_string())
    return tbl


# Part 4: extra data-science checks
def extra_diagnostics(X, y, groups, is_dummy, summary, skip_knn, Xtr, ytr, seed, out):
    print("\n--- Class balance ---")
    print(y.value_counts(normalize=True).rename({0: "not canceled", 1: "canceled"}).round(3).to_string())

    num = X.loc[:, ~is_dummy]
    corr_t = num.corrwith(y).dropna().sort_values(key=abs, ascending=False)
    print("\n--- Top numeric correlations with is_canceled ---")
    print(corr_t.head(8).round(3).to_string())

    cm = num.corr().abs()
    pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(cm.columns)
             for b in cm.columns[i + 1:] if cm.loc[a, b] > 0.85]
    print("\n--- Highly collinear numeric pairs (|r| > 0.85) ---")
    print("\n".join(f"  {a} ~ {b}: {r:.2f}" for a, b, r in pairs) or "  none")
    print("  (These split credit between each other in LR coefficients and tree importances.)")

    print("\n--- Decision-tree depth (3-fold CV on training data, ROC-AUC) ---")
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    depths = [3, 5, 8, 12, 20, None]
    sc = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=seed), Xtr, ytr,
                          cv=cv, scoring="roc_auc", n_jobs=-1).mean() for d in depths]
    for d, s in zip(depths, sc):
        print(f"  max_depth={str(d):<5s} AUC={s:.4f}")
    print(f"  -> best: max_depth={depths[int(np.argmax(sc))]} "
          "(the notebook's unpruned tree memorizes training data)")

    best = summary["macro_f1"].idxmax()
    print(f"\n--- 5-fold CV of best model ({best}) on all data ---")
    m = make_models(seed, skip_knn)[best]
    cv5 = StratifiedKFold(5, shuffle=True, random_state=seed)
    for metric in ("f1_macro", "roc_auc"):
        s = cross_val_score(m, X, y, cv=cv5, scoring=metric, n_jobs=1)
        print(f"  {metric}: {s.mean():.4f} +/- {s.std():.4f}")


def register_fitted(fitted, summary, X, groups, test_size, seed, max_rows):
    """Save every fitted model as a new version in the SQLite model registry."""
    schema = build_input_schema(X, groups)            # raw input fields + an example record
    scope = f"{max_rows:,}-row subsample" if max_rows else "full data"
    for name, (model, _, _) in fitted.items():
        metrics = {k: float(v) for k, v in summary.loc[name].items()}
        meta = register_model(model, DEFAULT_MODEL_NAME, metrics=metrics,
                              features=list(X.columns), schema=schema,
                              notes=f"{name} | test_size={test_size} seed={seed} | {scope}")
        print(f"  registered {name:<20s} -> {DEFAULT_MODEL_NAME} v{meta['version']}")


def register_trained_models(fitted, summary, X, groups):
    """Register trained models in the SQLite model registry."""

    print("\n=== Registering trained models ===")

    schema = build_input_schema(X, groups)

    for model_name, model_info in fitted.items():
        model = model_info[0]

        metrics = {
            key: float(value)
            for key, value in summary.loc[model_name].to_dict().items()
        }

        registered = register_model(
            model=model,
            name=DEFAULT_MODEL_NAME,
            metrics=metrics,
            features=list(X.columns),
            schema=schema,
            algorithm=model_name,
            notes="Registered from model_analysis.py",
        )

        print(
            f"Registered {model_name} as "
            f"{DEFAULT_MODEL_NAME} v{registered['version']}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hotel_bookings_cleaned.csv")
    ap.add_argument("--out", default="src/analysis_output")
    ap.add_argument("--seeds", type=int, default=5, help="random seeds per split size")
    ap.add_argument("--sizes", type=float, nargs="+",
                    default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    ap.add_argument("--skip-knn", action="store_true", help="KNN is slow on ~100k x 100 data")
    ap.add_argument("--max-rows", type=int, default=0, help="subsample for speed (0 = all)")
    ap.add_argument("--perm-repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="3 seeds, 30k rows, skip KNN")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--no-register", action="store_true",
                    help="do not save the trained models to the SQLite model registry")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.max_rows, a.skip_knn, a.perm_repeats = 3, 30000, True, 3

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X, y, groups, is_dummy = load_data(a.data, a.max_rows, a.random_state)

    # ---- Part 1
    print(f"\n=== 1. Train/test split search ({len(a.sizes)} sizes x {a.seeds} seeds) ===")
    sweep = split_sweep(X, y, a.sizes, range(a.seeds), a.skip_knn)
    sweep.to_csv(out / "split_sweep_raw.csv", index=False)
    best_ts, chosen, agg = pick_split(sweep)
    per_model = sweep.groupby(["model", "test_size"])["macro_f1"].mean().unstack(0)
    print("\nMean macro-F1 by test_size:")
    print(per_model.assign(OVERALL=agg["mean"], sem=agg["sem"]).round(4).to_string())
    print(f"\nHighest mean score : test_size={best_ts}")
    print(f"RECOMMENDED        : test_size={chosen} "
          "(largest test size statistically tied with the best, within 1 SE)")
    if agg["mean"].max() - agg["mean"].min() < 2 * agg["sem"].max():
        print("  Note: the spread across split sizes is within noise -- the split choice "
              "barely matters for this data; prefer 5-fold CV for the final number.")

    lc = learning_curve(X, y, a.skip_knn, a.random_state)
    lc.to_csv(out / "learning_curve.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, g in sweep.groupby("model"):
        s = g.groupby("test_size")["macro_f1"].agg(["mean", "sem"])
        a1.errorbar(s.index, s["mean"], yerr=s["sem"], marker="o", capsize=3, label=name)
    a1.axvline(chosen, color="k", ls="--", lw=0.8, label=f"recommended ({chosen})")
    a1.set(title="Macro-F1 vs test_size (mean +/- SE over seeds)", xlabel="test_size",
           ylabel="macro-F1")
    a1.legend()
    for name, g in lc.groupby("model"):
        a2.plot(g["n_train"], g["macro_f1"], marker="o", label=name)
    a2.set(title="Learning curve (fixed 20% holdout)", xlabel="training rows", ylabel="macro-F1")
    a2.legend()
    fig.tight_layout()
    fig.savefig(out / "split_sweep.png", dpi=130)
    plt.close(fig)

    # ---- Part 2
    print(f"\n=== 2. Classification-report evaluation (test_size={chosen}, seed={a.random_state}) ===")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=chosen, random_state=a.random_state,
                                          stratify=y)
    fitted, summary = evaluate_models(Xtr, Xte, ytr, yte, a.skip_knn, a.random_state, out)

    if not a.no_register:
        print("\n=== Registering models in the SQLite registry ===")
        register_fitted(fitted, summary, X, groups, chosen, a.random_state, a.max_rows)

    # ---- Part 3
    print("\n=== 3. Feature significance ===")
    res = feature_significance(fitted, X, Xte, yte, y, groups, is_dummy, a.perm_repeats,
                               a.random_state, out)
    ablation(res, X, groups, Xtr, Xte, ytr, yte, a.random_state).to_csv(out / "ablation.csv")

    # ---- Part 4
    print("\n=== 4. Additional diagnostics ===")
    extra_diagnostics(X, y, groups, is_dummy, summary, a.skip_knn, Xtr, ytr, a.random_state, out)

    print(f"\nAll tables and plots saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
