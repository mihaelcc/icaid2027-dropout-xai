"""
Explainable Machine Learning for Early Prediction of Student Dropout
Experimento reproducible para ICAID 2027 (Track 5: Smart Education & Learning Analytics / Track 4: XAI)

Dataset: "Predict Students' Dropout and Academic Success" (UCI ML Repository, id=697)
         Realinho et al., Data 7(11):146, 2022. 4,424 estudiantes, 36 atributos, 3 clases.

Diseño:
  - Tarea principal: clasificación binaria Dropout vs. No-dropout (alerta temprana).
  - Tres "etapas" de información (cuándo podría actuar la universidad):
      S0 Admisión   : demográficos, socioeconómicos, antecedentes académicos, macroeconómicos
      S1 1er semestre: S0 + desempeño en 1er semestre
      S2 2do semestre: S1 + desempeño en 2do semestre
  - Modelos: Regresión Logística, Random Forest, XGBoost, LightGBM
  - Validación: hold-out 80/20 estratificado + CV estratificada repetida (5 folds x 3) en el 80%
  - Pruebas estadísticas: Friedman (modelos) y Wilcoxon (etapas)
  - Explicabilidad: SHAP (TreeExplainer) sobre el mejor modelo en S1 y S2

Uso:
  pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib scipy
  # Opción A: CSV local (dataset.csv con separador ',' o ';')
  python dropout_xai.py --csv dataset.csv
  # Opción B: descarga directa desde UCI
  pip install ucimlrepo && python dropout_xai.py
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                             balanced_accuracy_score, roc_curve, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

warnings.filterwarnings("ignore")
SEED = 42
OUT = "results"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. Datos
def load_data(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=697)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = [c.replace("\ufeff", "").strip().replace("\t", " ") for c in df.columns]
    # Normalizar nombres: la versión oficial de UCI usa mayúsculas distintas (p. ej. "Marital Status")
    canon = {c.lower(): c for c in NOMINAL}
    df = df.rename(columns=lambda c: canon.get(c.lower(), c))
    df = df.rename(columns={"Nacionality": "Nationality"})
    df = df.rename(columns=lambda c: "Nationality" if c.lower() == "nacionality" else c)
    print("Columnas:", len(df.columns) - 1, "predictores")
    missing = [c for c in NOMINAL if c not in df.columns]
    if missing: print("AVISO: columnas nominales no encontradas:", missing)
    return df

NOMINAL = ["Marital status", "Application mode", "Course", "Previous qualification",
           "Nationality", "Mother's qualification", "Father's qualification",
           "Mother's occupation", "Father's occupation"]

def feature_stages(cols):
    sem1 = [c for c in cols if "1st sem" in c]
    sem2 = [c for c in cols if "2nd sem" in c]
    s0 = [c for c in cols if c not in sem1 + sem2 + ["Target"]]
    return {"S0_Admission": s0, "S1_FirstSemester": s0 + sem1, "S2_SecondSemester": s0 + sem1 + sem2}

# ---------------------------------------------------------------- 2. Modelos
def make_models(features, pos_weight):
    nom = [c for c in NOMINAL if c in features]
    num = [c for c in features if c not in nom]
    pre_lr = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10), nom),
                                ("sc", StandardScaler(), num)])
    return {
        "LogReg": Pipeline([("pre", pre_lr),
                            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))]),
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                               n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                                 n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                                   random_state=SEED, verbose=-1),
    }

def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {"AUC": roc_auc_score(y, p), "F1_dropout": f1_score(y, yhat),
            "Recall_dropout": recall_score(y, yhat), "Precision_dropout": precision_score(y, yhat),
            "BalancedAcc": balanced_accuracy_score(y, yhat), "MacroF1": f1_score(y, yhat, average="macro")}

# ---------------------------------------------------------------- 3. Experimento
def main(csv_path):
    df = load_data(csv_path)
    y = (df["Target"] == "Dropout").astype(int).values
    stages = feature_stages(df.columns)
    X_tr_all, X_te_all, y_tr, y_te = train_test_split(df.drop(columns="Target"), y, test_size=0.2,
                                                      stratify=y, random_state=SEED)
    pos_w = (y_tr == 0).sum() / (y_tr == 1).sum()
    info = {"n": len(df), "class_counts": df["Target"].value_counts().to_dict(),
            "dropout_rate": float(y.mean()), "n_train": len(y_tr), "n_test": len(y_te),
            "stage_n_features": {k: len(v) for k, v in stages.items()}}
    json.dump(info, open(f"{OUT}/dataset_info.json", "w"), indent=2)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    cv_rows, test_rows, fold_auc, test_probs, fitted = [], [], {}, {}, {}
    for st, feats in stages.items():
        Xtr, Xte = X_tr_all[feats], X_te_all[feats]
        for name, model in make_models(feats, pos_w).items():
            aucs = []
            for k, (i_tr, i_va) in enumerate(cv.split(Xtr, y_tr)):
                m = make_models(feats, pos_w)[name]
                m.fit(Xtr.iloc[i_tr], y_tr[i_tr])
                r = metrics(y_tr[i_va], m.predict_proba(Xtr.iloc[i_va])[:, 1])
                r.update(stage=st, model=name, fold=k); cv_rows.append(r); aucs.append(r["AUC"])
            fold_auc[(st, name)] = aucs
            model.fit(Xtr, y_tr)
            p = model.predict_proba(Xte)[:, 1]
            r = metrics(y_te, p); r.update(stage=st, model=name); test_rows.append(r)
            test_probs[(st, name)] = p; fitted[(st, name)] = model
            print(f"{st:18s} {name:12s} CV-AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  Test-AUC={r['AUC']:.4f}  F1={r['F1_dropout']:.4f}")

    cv_df = pd.DataFrame(cv_rows); test_df = pd.DataFrame(test_rows)
    cv_df.to_csv(f"{OUT}/cv_folds.csv", index=False)
    summ = cv_df.groupby(["stage", "model"]).agg(["mean", "std"]).drop(columns="fold").round(4)
    summ.to_csv(f"{OUT}/cv_summary.csv"); test_df.round(4).to_csv(f"{OUT}/test_results.csv", index=False)

    # ---------------------------------------------------- 4. Pruebas estadísticas
    stats = {}
    models = list(make_models(stages["S0_Admission"], 1).keys())
    for st in stages:
        f = friedmanchisquare(*[fold_auc[(st, m)] for m in models])
        stats[f"friedman_{st}"] = {"chi2": float(f.statistic), "p": float(f.pvalue)}
    best = {st: max(models, key=lambda m: np.mean(fold_auc[(st, m)])) for st in stages}
    stats["best_model_per_stage"] = best
    st_names = list(stages)
    for a, b in [(0, 1), (1, 2)]:
        m = best[st_names[b]]
        w = wilcoxon(fold_auc[(st_names[a], m)], fold_auc[(st_names[b], m)])
        stats[f"wilcoxon_{st_names[a]}_vs_{st_names[b]}_{m}"] = {"W": float(w.statistic), "p": float(w.pvalue)}
    json.dump(stats, open(f"{OUT}/statistical_tests.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))

    # ---------------------------------------------------- 5. Figuras
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    # 5a. AUC por etapa y modelo
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    w = 0.2; xs = np.arange(len(stages))
    for i, m in enumerate(models):
        mu = [np.mean(fold_auc[(s, m)]) for s in stages]; sd = [np.std(fold_auc[(s, m)]) for s in stages]
        ax.bar(xs + (i - 1.5) * w, mu, w, yerr=sd, capsize=2, label=m)
    ax.set_xticks(xs); ax.set_xticklabels(["Admission (S0)", "1st semester (S1)", "2nd semester (S2)"])
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("ROC-AUC (5x3 CV)"); ax.legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_auc_by_stage.png", dpi=300); plt.close(fig)
    # 5b. Curvas ROC (test) del mejor modelo por etapa
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for st in stages:
        fpr, tpr, _ = roc_curve(y_te, test_probs[(st, best[st])])
        ax.plot(fpr, tpr, label=f"{st.split('_')[0]} {best[st]} (AUC={roc_auc_score(y_te, test_probs[(st, best[st])]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_roc_test.png", dpi=300); plt.close(fig)

    # ---------------------------------------------------- 6. SHAP
    shap_tables = {}
    for st in ["S1_FirstSemester", "S2_SecondSemester"]:
        m_name = best[st] if best[st] != "LogReg" else "XGBoost"
        model = fitted[(st, m_name)]; Xte = X_te_all[stages[st]]
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xte)
        if isinstance(sv, list): sv = sv[1]
        if sv.ndim == 3: sv = sv[:, :, 1]
        imp = pd.Series(np.abs(sv).mean(0), index=Xte.columns).sort_values(ascending=False)
        imp.round(4).to_csv(f"{OUT}/shap_importance_{st}.csv", header=["mean_abs_shap"])
        shap_tables[st] = (m_name, imp.head(10).round(4).to_dict())
        plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
        shap.summary_plot(sv, Xte, max_display=12, show=False, plot_size=(8, 4.8))
        plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap_{st}.png", dpi=300, bbox_inches="tight"); plt.close()
    json.dump(shap_tables, open(f"{OUT}/shap_top10.json", "w"), indent=2)

    # Matriz de confusión del mejor modelo S1 (escenario de alerta temprana)
    cm = confusion_matrix(y_te, (test_probs[("S1_FirstSemester", best["S1_FirstSemester"])] >= 0.5).astype(int))
    pd.DataFrame(cm, index=["true_no", "true_dropout"], columns=["pred_no", "pred_dropout"]).to_csv(f"{OUT}/confusion_S1.csv")
    print("Listo. Resultados en ./results")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", default=None)
    main(ap.parse_args().csv)
