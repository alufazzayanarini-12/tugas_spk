import numpy as np
import pandas as pd
import streamlit as st

# 1. Konfigurasi Halaman Layar Laptop / Desktop (Wide Mode)
st.set_page_config(
    page_title="SPK AHP - Dashboard Pemilihan Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Styling CSS Dashboard Bersih & Kontras Tinggi
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }
    
    /* Header Banner */
    .desktop-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .header-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.92rem;
        color: #94A3B8 !important;
        margin-top: 4px;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-left: 5px solid #2563EB;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    .kpi-card-success { border-left-color: #10B981; }
    .kpi-card-accent { border-left-color: #F59E0B; }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 4px;
    }
    .kpi-subtext {
        font-size: 0.82rem;
        color: #CBD5E1;
        margin-top: 4px;
    }

    /* Hero Card Keputusan Utama */
    .winner-card {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        border-radius: 14px;
        padding: 24px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
    }
    .winner-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 10px;
    }
    .winner-car-name {
        font-size: 2.2rem;
        font-weight: 850;
        margin: 0;
        color: #FFFFFF !important;
    }
    .winner-desc {
        font-size: 0.95rem;
        color: #E2E8F0;
        margin-top: 8px;
        line-height: 1.5;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 3. Mesin Komputasi AHP
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


# 4. Data Studi Kasus
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

# Menjalankan Perhitungan
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
df_scores["Skor Komposit"] = final_scores
df_scores["Peringkat"] = (
    df_scores["Skor Komposit"].rank(ascending=False, method="min").astype(int)
)
ranking_df = df_scores.sort_values(by="Peringkat")

best_car = ranking_df.index[0]
best_score = ranking_df.iloc[0]["Skor Komposit"]

# 5. Header Banner Desktop
st.markdown(
    """
    <div class="desktop-header">
        <div>
            <h1 class="header-title">🚗 CarDecision Analytics Dashboard</h1>
            <p class="header-subtitle">Sistem Pendukung Keputusan Pemilihan Mobil Terbaik Berbasis Analytic Hierarchy Process (AHP)</p>
        </div>
        <div style="text-align: right; color: #94A3B8; font-size: 0.85rem;">
            <div>Metode: <b>AHP (Saaty)</b></div>
            <div>Status: <span style="color: #34D399; font-weight: bold;">● Konsisten (CR ≤ 0.1)</span></div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# 6. Baris KPI Cards
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-success">
            <div class="kpi-label">Pilihan Terbaik (Rank #1)</div>
            <div class="kpi-value">{best_car}</div>
            <div class="kpi-subtext">Skor Tertinggi: <b>{best_score:.4f}</b></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col_kpi2:
    top_crit = criteria[np.argmax(ahp_crit.weights)]
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Kriteria Terpenting</div>
            <div class="kpi-value">{top_crit}</div>
            <div class="kpi-subtext">Bobot: <b>{np.max(ahp_crit.weights):.4f}</b></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col_kpi3:
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-accent">
            <div class="kpi-label">Uji Rasio Konsistensi (CR)</div>
            <div class="kpi-value">{ahp_crit.cr:.4f}</div>
            <div class="kpi-subtext">Batas Maks: 0.10 (Valid)</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col_kpi4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Evaluasi Objek</div>
            <div class="kpi-value">{len(alternatives)} Mobil</div>
            <div class="kpi-subtext">{len(criteria)} Kriteria Pertimbangan</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.write("")

# 7. Tab Tampilan Data
tab_overview, tab_criteria, tab_alternatives = st.tabs(
    [
        "📊 Ringkasan & Peringkat Mobil",
        "⚖️ Analisis Kriteria & Uji Konsistensi",
        "🚗 Matriks Perbandingan Alternatif",
    ]
)

with tab_overview:
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.markdown(
            f"""
            <div class="winner-card">
                <div class="winner-badge">⭐ HASIL KEPUTUSAN UTAMA</div>
                <div class="winner-car-name">{best_car}</div>
                <div class="winner-desc">
                    Mobil terpilih dengan skor evaluasi tertinggi sebesar <b>{best_score:.4f}</b>, didukung keunggulan utama pada kriteria <b>Keandalan</b> dan efisiensi <b>Konsumsi BBM</b>.
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.subheader("📋 Tabel Peringkat Lengkap")
        st.dataframe(
            ranking_df[
                [
                    "Peringkat",
                    "Skor Komposit",
                    "Style (S)",
                    "Keandalan (K)",
                    "Konsumsi BBM (B)",
                ]
            ].style.format(
                {
                    "Skor Komposit": "{:.4f}",
                    "Style (S)": "{:.4f}",
                    "Keandalan (K)": "{:.4f}",
                    "Konsumsi BBM (B)": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

    with col_right:
        st.subheader("📈 Visualisasi Skor Komposit")
        st.bar_chart(ranking_df["Skor Komposit"])

        st.subheader("📊 Performa per Kriteria")
        st.bar_chart(
            ranking_df[["Style (S)", "Keandalan (K)", "Konsumsi BBM (B)"]]
        )

with tab_criteria:
    st.subheader("⚖️ Perbandingan Berpasangan Antar-Kriteria")
    col_c1, col_c2 = st.columns([1.5, 1])

    with col_c1:
        st.markdown("**Matriks Perbandingan Kriteria:**")
        df_kriteria = pd.DataFrame(
            ahp_crit.matrix, index=criteria, columns=criteria
        )
        st.dataframe(
            df_kriteria.style.format(precision=4), use_container_width=True
        )

    with col_c2:
        st.markdown("**Hasil Bobot Prioritas:**")
        df_bobot = pd.DataFrame(
            {"Bobot Prioritas": ahp_crit.weights}, index=criteria
        )
        st.dataframe(
            df_bobot.style.format(precision=4), use_container_width=True
        )

        st.info(
            f"**Uji Konsistensi (Saaty):**\n"
            f"- **Lambda Maks (λ max):** `{ahp_crit.lambda_max:.4f}`\n"
            f"- **Consistency Index (CI):** `{ahp_crit.ci:.4f}`\n"
            f"- **Consistency Ratio (CR):** `{ahp_crit.cr:.4f}`\n"
            f"- **Status:** {'✅ KONSISTEN (Diterima)' if ahp_crit.is_consistent else '❌ TIDAK KONSISTEN'}"
        )

with tab_alternatives:
    st.subheader("🚗 Matriks Perbandingan Berpasangan Setiap Kriteria")

    selected_crit = st.selectbox(
        "Pilih Kriteria yang Ingin Dianalisis:", criteria
    )
    mat_obj = ahp_alts[selected_crit]

    col_a1, col_a2 = st.columns([1.5, 1])

    with col_a1:
        st.markdown(f"**Matriks Perbandingan ({selected_crit}):**")
        df_alt_mat = pd.DataFrame(
            mat_obj.matrix, index=alternatives, columns=alternatives
        )
        st.dataframe(
            df_alt_mat.style.format(precision=4), use_container_width=True
        )

    with col_a2:
        st.markdown(f"**Bobot Prioritas Mobil ({selected_crit}):**")
        df_alt_weights = pd.DataFrame(
            {"Bobot": mat_obj.weights}, index=alternatives
        ).sort_values(by="Bobot", ascending=False)
        st.dataframe(
            df_alt_weights.style.format(precision=4), use_container_width=True
        )

        st.caption(
            f"λ max: {mat_obj.lambda_max:.4f} | CI: {mat_obj.ci:.4f} | CR: {mat_obj.cr:.4f} "
            f"({'✅ Konsisten' if mat_obj.is_consistent else '⚠️ Tidak Konsisten'})"
        )