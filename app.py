import numpy as np
import pandas as pd
import streamlit as st

# 1. Konfigurasi Halaman (Centered Layout agar menyerupai frame aplikasi mobile)
st.set_page_config(
    page_title="CarAHP Mobile App",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# 2. Custom CSS: Mobile Smartphone App Skin
st.markdown(
    """
    <style>
    /* Styling Frame Kontainer Mobile */
    .block-container {
        max-width: 440px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }
    
    /* Header Bar Aplikasi */
    .mobile-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 15px;
        margin-bottom: 15px;
        border-bottom: 1px solid #2D3748;
    }
    .app-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F7FAFC;
    }
    .app-subtitle {
        font-size: 0.75rem;
        color: #A0AEC0;
    }

    /* Card Pemenang (Hero Card) */
    .winner-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 20px;
        padding: 20px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .winner-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .winner-name {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }
    .winner-score {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 4px;
    }

    /* Card Item List Alternatif */
    .mobile-card {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .car-name {
        font-size: 1rem;
        font-weight: 600;
        color: #EDF2F7;
    }
    .car-rank {
        font-size: 0.8rem;
        font-weight: 700;
        border-radius: 8px;
        padding: 4px 8px;
    }
    .rank-1 { background-color: #FEF3C7; color: #D97706; }
    .rank-2 { background-color: #E2E8F0; color: #475569; }
    .rank-3 { background-color: #FFEDD5; color: #C2410C; }
    .rank-other { background-color: #2D3748; color: #A0AEC0; }
    </style>
""",
    unsafe_allow_html=True,
)


# 3. Mesin Logika Komputasi AHP
class AHPMatrix:

    RI_DICT = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }

    def __init__(self, name: str, elements: list, matrix: np.ndarray):
        self.name = name
        self.elements = elements
        self.matrix = np.array(matrix, dtype=float)
        self.n = len(elements)
        self.weights = None
        self.lambda_max = None
        self.ci = None
        self.cr = None
        self.is_consistent = None
        self._compute()

    def _compute(self):
        norm_matrix = self.matrix / self.matrix.sum(axis=0)
        self.weights = norm_matrix.mean(axis=1)
        weighted_sum = np.dot(self.matrix, self.weights)
        self.lambda_max = float(np.mean(weighted_sum / self.weights))
        self.ci = (
            (self.lambda_max - self.n) / (self.n - 1) if self.n > 1 else 0.0
        )
        ri = self.RI_DICT.get(self.n, 1.49)
        self.cr = (self.ci / ri) if ri != 0 else 0.0
        self.is_consistent = self.cr <= 0.10


# Data Studi Kasus
criteria = ["Style (S)", "Keandalan (K)", "Konsumsi BBM (B)"]
alternatives = ["Avanza", "Xenia", "Ertiga", "Grand Livina"]

mat_criteria = np.array(
    [[1.0, 1.0 / 2.0, 3.0], [2.0, 1.0, 4.0], [1.0 / 3.0, 1.0 / 4.0, 1.0]]
)

mat_style = np.array(
    [
        [1.0, 2.0, 1.0 / 3.0, 1.0 / 4.0],
        [1.0 / 2.0, 1.0, 1.0 / 2.0, 1.0 / 5.0],
        [3.0, 2.0, 1.0, 2.0],
        [4.0, 5.0, 1.0 / 2.0, 1.0],
    ]
)

mat_keandalan = np.array(
    [
        [1.0, 2.0, 1.0 / 4.0, 1.0 / 3.0],
        [1.0 / 2.0, 1.0, 1.0 / 5.0, 1.0 / 4.0],
        [4.0, 5.0, 1.0, 3.0],
        [3.0, 4.0, 1.0 / 3.0, 1.0],
    ]
)

mat_bbm = np.array(
    [
        [1.0, 2.0, 1.0 / 2.0, 4.0],
        [1.0 / 2.0, 1.0, 2.0 / 3.0, 3.0],
        [2.0, 1.5, 1.0, 4.0],
        [1.0 / 4.0, 1.0 / 3.0, 1.0 / 4.0, 1.0],
    ]
)

# Kalkulasi Model
ahp_crit = AHPMatrix("Kriteria Utama", criteria, mat_criteria)
ahp_alts = {
    "Style (S)": AHPMatrix("Alternatif - Style", alternatives, mat_style),
    "Keandalan (K)": AHPMatrix(
        "Alternatif - Keandalan", alternatives, mat_keandalan
    ),
    "Konsumsi BBM (B)": AHPMatrix(
        "Alternatif - Konsumsi BBM", alternatives, mat_bbm
    ),
}

alt_weights_matrix = [m.weights for m in ahp_alts.values()]
df_scores = pd.DataFrame(
    np.array(alt_weights_matrix).T,
    index=alternatives,
    columns=list(ahp_alts.keys()),
)
final_scores = np.dot(df_scores.values, ahp_crit.weights)
df_scores["Skor Akhir"] = final_scores
df_scores["Peringkat"] = (
    df_scores["Skor Akhir"].rank(ascending=False, method="min").astype(int)
)
ranking_df = df_scores.sort_values(by="Peringkat")

best_car = ranking_df.index[0]
best_score = ranking_df.iloc[0]["Skor Akhir"]

# 4. Mobile Header
st.markdown(
    """
    <div class="mobile-header">
        <div>
            <div class="app-title">🚗 CarDecision AHP</div>
            <div class="app-subtitle">Sistem Pendukung Keputusan Mobil</div>
        </div>
        <div style="font-size: 1.4rem;">🔔</div>
    </div>
""",
    unsafe_allow_html=True,
)

# 5. Segment Navigation (Tab Berbasis Navigasi Mobile)
selected_tab = st.segmented_control(
    "Navigasi",
    options=["🏆 Home", "📊 Analisis", "⚙️ Matriks"],
    default="🏆 Home",
    label_visibility="collapsed",
)

st.write("")

if selected_tab == "🏆 Home":
    # Hero Card (Peringkat Teratas)
    st.markdown(
        f"""
        <div class="winner-card">
            <div class="winner-badge">⭐ PILIHAN TERBAIK</div>
            <div class="winner-name">{best_car}</div>
            <div class="winner-score">Skor Komposit: <b>{best_score:.4f}</b> (Rank #1)</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🏁 Peringkat Alternatif")

    # Daftar Item Model List Mobile
    for car_name, row in ranking_df.iterrows():
        rank = int(row["Peringkat"])
        score = row["Skor Akhir"]

        rank_class = (
            "rank-1"
            if rank == 1
            else (
                "rank-2"
                if rank == 2
                else ("rank-3" if rank == 3 else "rank-other")
            )
        )

        st.markdown(
            f"""
            <div class="mobile-card">
                <div>
                    <div class="car-name">{car_name}</div>
                    <div style="font-size: 0.78rem; color: #718096; margin-top: 2px;">
                        Skor: {score:.4f}
                    </div>
                </div>
                <div class="car-rank {rank_class}">Rank #{rank}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

elif selected_tab == "📊 Analisis":
    st.markdown("#### 📈 Perbandingan Skor")
    st.bar_chart(ranking_df["Skor Akhir"])

    st.markdown("#### ⚖️ Bobot Kriteria")
    crit_df = pd.DataFrame(
        {"Kriteria": criteria, "Bobot": ahp_crit.weights}
    ).set_index("Kriteria")
    st.dataframe(crit_df.style.format(precision=4), use_container_width=True)

    status_icon = "✅" if ahp_crit.is_consistent else "⚠️"
    st.info(f"{status_icon} Consistency Ratio (CR): **{ahp_crit.cr:.4f}**")

elif selected_tab == "⚙️ Matriks":
    st.markdown("#### 📋 Matriks Berpasangan")
    opsi = st.selectbox(
        "Kategori Matriks:", ["Kriteria Utama"] + list(ahp_alts.keys())
    )

    if opsi == "Kriteria Utama":
        target_obj = ahp_crit
    else:
        target_obj = ahp_alts[opsi]

    df_display = pd.DataFrame(
        target_obj.matrix,
        index=target_obj.elements,
        columns=target_obj.elements,
    )
    st.dataframe(df_display.style.format(precision=3), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("λ Max", f"{target_obj.lambda_max:.3f}")
    c2.metric("CR", f"{target_obj.cr:.3f}")