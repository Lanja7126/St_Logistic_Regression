import streamlit as st
import numpy as np
import joblib

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Logistic Regression Demo", page_icon="🎓", layout="wide")

# ── Fonctions communes ───────────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w, b):
    m = X.shape[0]
    p = np.zeros(m)
    for i in range(m):
        z = np.dot(X[i], w) + b
        p[i] = 1 if sigmoid(z) > 0.5 else 0
    return p

def predict_proba(X, w, b):
    z = X @ w + b
    return sigmoid(z)

def map_feature(X1, X2):
    """Feature mapping polynomial degré 6 (identique à utils.py)."""
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))
    return np.stack(out, axis=1)

# ── Chargement des modèles ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        m1 = joblib.load("logistic_model.pkl")
    except FileNotFoundError:
        m1 = None
    try:
        m2 = joblib.load("logistic_model_reg.pkl")
    except FileNotFoundError:
        m2 = None
    return m1, m2

model1, model2 = load_models()

# ── Titre principal ──────────────────────────────────────────────────────────
st.title("📊 Régression Logistique — Démo")
st.markdown("Deux modèles indépendants entraînés sur des jeux de données différents.")
st.divider()

# ── Deux colonnes, un modèle chacune ────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

# ══════════════════════════════════════════════════════════
# MODÈLE 1 — Admission universitaire
# ══════════════════════════════════════════════════════════
with col1:
    st.subheader("🎓 Modèle 1 — Admission universitaire")
    st.caption("Prédit si un étudiant sera admis selon ses scores à deux examens.")

    if model1 is None:
        st.warning("⚠️ `logistic_model.pkl` introuvable. Lance `save_models.ipynb` d'abord.")
    else:
        exam1 = st.number_input("📝 Score Examen 1", min_value=0.0, max_value=100.0, step=0.5)
        exam2 = st.number_input("📝 Score Examen 2", min_value=0.0, max_value=100.0, step=0.5)

        if st.button("🔮 Prédire — Modèle 1", use_container_width=True):
            w, b = model1["w"], model1["b"]
            X_in = np.array([[exam1, exam2]])
            pred  = predict(X_in, w, b)
            proba = float(np.squeeze(predict_proba(X_in, w, b)))

            st.divider()
            if pred[0] == 1:
                st.success(f"✅ **ADMIS**  —  Probabilité : **{proba:.1%}**")
            else:
                st.error(f"❌ **REFUSÉ**  —  Probabilité : **{proba:.1%}**")
            st.progress(proba)

        with st.expander("📋 Exemples rapides"):
            exemples = np.array([[30,85],[60,65],[80,80],[45,55],[90,90]], dtype=float)
            w, b = model1["w"], model1["b"]
            preds  = predict(exemples, w, b)
            probas = np.squeeze(predict_proba(exemples, w, b))
            rows = []
            for x, p, pr in zip(exemples, preds, probas):
                rows.append({
                    "Exam 1": int(x[0]), "Exam 2": int(x[1]),
                    "P(admission)": f"{float(pr):.1%}",
                    "Décision": "ADMIS ✅" if p == 1 else "REFUSÉ ❌"
                })
            st.table(rows)

# ══════════════════════════════════════════════════════════
# MODÈLE 2 — Contrôle qualité microchips
# ══════════════════════════════════════════════════════════
with col2:
    st.subheader("🔬 Modèle 2 — Contrôle qualité microchips")
    st.caption("Prédit si un microchip passe le contrôle qualité. Utilise des features polynomiales (degré 6) + régularisation L2.")

    if model2 is None:
        st.warning("⚠️ `logistic_model_reg.pkl` introuvable. Lance `save_models.ipynb` d'abord.")
    else:
        test1 = st.slider("🔧 Test microchip 1", -1.5, 1.5, 0.5, 0.01, key="t1")
        test2 = st.slider("🔧 Test microchip 2", -1.5, 1.5, 0.5, 0.01, key="t2")

        if st.button("🔮 Prédire — Modèle 2", use_container_width=True):
            w, b = model2["w"], model2["b"]
            X_in = map_feature(np.array([test1]), np.array([test2]))
            pred  = predict(X_in, w, b)
            proba = float(np.squeeze(predict_proba(X_in, w, b)))

            st.divider()
            if pred[0] == 1:
                st.success(f"✅ **ACCEPTÉ**  —  Probabilité : **{proba:.1%}**")
            else:
                st.error(f"❌ **REJETÉ**  —  Probabilité : **{proba:.1%}**")
            st.progress(proba)

        with st.expander("📋 Exemples rapides"):
            exemples2 = np.array([[0.5,0.5],[-0.5,0.7],[0.3,-0.4],[-0.8,-0.8],[1.0,0.2]])
            w, b = model2["w"], model2["b"]
            X_map = map_feature(exemples2[:,0], exemples2[:,1])
            preds2  = predict(X_map, w, b)
            probas2 = np.squeeze(predict_proba(X_map, w, b))
            rows2 = []
            for x, p, pr in zip(exemples2, preds2, probas2):
                rows2.append({
                    "Test 1": x[0], "Test 2": x[1],
                    "P(accepté)": f"{float(pr):.1%}",
                    "Décision": "ACCEPTÉ ✅" if p == 1 else "REJETÉ ❌"
                })
            st.table(rows2)
