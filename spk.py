import numpy as np
import pandas as pd

# Konfigurasi tampilan Pandas agar rapi di terminal
pd.set_option("display.precision", 4)
pd.set_option("display.width", 1000)


class AHPMatrix:
    """Kelas untuk mengelola perhitungan matriks perbandingan berpasangan AHP."""

    # Nilai Random Index (RI) standar Saaty berdasarkan ukuran matriks (n)
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

        # Variabel hasil komputasi
        self.weights = None
        self.lambda_max = None
        self.ci = None
        self.cr = None
        self.is_consistent = None

        self._compute()

    def _compute(self):
        """Menghitung bobot prioritas, Lambda Max, CI, dan CR."""
        # 1. Normalisasi matriks: bagi tiap elemen dengan jumlah kolomnya
        col_sum = self.matrix.sum(axis=0)
        norm_matrix = self.matrix / col_sum

        # 2. Bobot prioritas (vektor eigen pendekatan): rata-rata tiap baris
        self.weights = norm_matrix.mean(axis=1)

        # 3. Hitung Lambda Maks (λ max)
        # Mengalikan matriks awal dengan vektor bobot, lalu bagi dengan bobot masing-masing
        weighted_sum = np.dot(self.matrix, self.weights)
        self.lambda_max = np.mean(weighted_sum / self.weights)

        # 4. Consistency Index (CI)
        if self.n > 1:
            self.ci = (self.lambda_max - self.n) / (self.n - 1)
        else:
            self.ci = 0.0

        # 5. Consistency Ratio (CR)
        ri = self.RI_DICT.get(self.n, 1.49)
        self.cr = (self.ci / ri) if ri != 0 else 0.0
        self.is_consistent = self.cr <= 0.10

    def display_results(self):
        """Menampilkan tabel matriks, bobot prioritas, dan uji konsistensi."""
        print("=" * 70)
        print(f" MATRIKS PERBANDINGAN: {self.name.upper()}")
        print("=" * 70)

        df_matrix = pd.DataFrame(
            self.matrix, index=self.elements, columns=self.elements
        )
        print("\nMatriks Perbandingan Berpasangan:")
        print(df_matrix)

        df_weights = pd.DataFrame(
            {"Bobot Prioritas": self.weights}, index=self.elements
        )
        print("\nBobot Prioritas:")
        print(df_weights)

        status_konsistensi = (
            "KONSISTEN (Diterima)"
            if self.is_consistent
            else "TIDAK KONSISTEN (Perlu Ditinjau)"
        )
        print("\nUji Konsistensi:")
        print(f" - Lambda Max (λ max) : {self.lambda_max:.4f}")
        print(f" - Consistency Index (CI): {self.ci:.4f}")
        print(f" - Consistency Ratio (CR): {self.cr:.4f}")
        print(f" - Status                 : {status_konsistensi}")
        print("-" * 70 + "\n")


class AHPDecisionMaker:
    """Kelas orkestrator untuk menyusun hierarki AHP dan menghitung skor akhir."""

    def __init__(self, criteria_matrix: AHPMatrix, alt_matrices: dict):
        self.criteria_matrix = criteria_matrix
        self.alt_matrices = alt_matrices
        self.alternatives = criteria_matrix.elements
        self.criteria = list(alt_matrices.keys())

    def solve(self):
        """Menghitung nilai komposit akhir untuk setiap alternatif."""
        # Tampilkan hasil perhitungan kriteria
        self.criteria_matrix.display_results()

        # Tampilkan hasil perhitungan alternatif untuk tiap kriteria
        alt_weights_matrix = []
        for crit_name, alt_mat in self.alt_matrices.items():
            alt_mat.display_results()
            alt_weights_matrix.append(alt_mat.weights)

        # Matriks bobot alternatif (baris: alternatif, kolom: kriteria)
        # Transpose agar dimensi sesuai: (n_alternatif x n_kriteria)
        alt_weights_df = pd.DataFrame(
            np.array(alt_weights_matrix).T,
            index=list(self.alt_matrices.values())[0].elements,
            columns=self.criteria,
        )

        # Hitung skor komposit: perkalian matriks bobot alternatif dengan bobot kriteria
        crit_weights = self.criteria_matrix.weights
        final_scores = np.dot(alt_weights_df.values, crit_weights)

        alt_weights_df["Skor Akhir"] = final_scores
        alt_weights_df["Peringkat"] = (
            alt_weights_df["Skor Akhir"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        # Urutkan berdasarkan peringkat terbaik
        summary_df = alt_weights_df.sort_values(by="Peringkat")

        print("=" * 70)
        print(" RINGKASAN HASIL AKHIR & PERINGKAT KEPUTUSAN")
        print("=" * 70)
        print(summary_df)
        print("-" * 70)

        best_car = summary_df.index[0]
        best_score = summary_df.iloc[0]["Skor Akhir"]
        print(
            f"\nKeputusan: Mobil terbaik yang dipilih adalah '{best_car}' "
            f"dengan skor komposit tertinggi sebesar {best_score:.4f}."
        )


if __name__ == "__main__":
    # -------------------------------------------------------------
    # 1. Definisi Data Kriteria & Alternatif
    # -------------------------------------------------------------
    criteria = ["Style (S)", "Keandalan (K)", "Konsumsi BBM (B)"]
    alternatives = ["Avanza", "Xenia", "Ertiga", "Grand Livina"]

    # -------------------------------------------------------------
    # 2. Matriks Perbandingan Kriteria: [Style, Keandalan, BBM]
    # Data perbandingan:
    # - Keandalan : Style = 2  ->  (K, S) = 2, maka (S, K) = 1/2
    # - BBM : Style = 1/3      ->  (B, S) = 1/3, maka (S, B) = 3
    # - BBM : Keandalan = 1/4  ->  (B, K) = 1/4, maka (K, B) = 4
    # -------------------------------------------------------------
    mat_criteria = np.array(
        [[1.0, 1.0 / 2.0, 3.0], [2.0, 1.0, 4.0], [1.0 / 3.0, 1.0 / 4.0, 1.0]]
    )

    # -------------------------------------------------------------
    # 3. Matriks Perbandingan Alternatif terhadap Kriteria "Style"
    # Urutan: [Avanza, Xenia, Ertiga, Grand Livina]
    # - Xenia : Avanza = 1/2
    # - Ertiga : Avanza = 3
    # - Grand Livina : Avanza = 4
    # - Ertiga : Xenia = 2
    # - Grand Livina : Xenia = 5
    # - Grand Livina : Ertiga = 1/2
    # -------------------------------------------------------------
    mat_style = np.array(
        [
            [1.0, 2.0, 1.0 / 3.0, 1.0 / 4.0],
            [1.0 / 2.0, 1.0, 1.0 / 2.0, 1.0 / 5.0],
            [3.0, 2.0, 1.0, 2.0],
            [4.0, 5.0, 1.0 / 2.0, 1.0],
        ]
    )

    # -------------------------------------------------------------
    # 4. Matriks Perbandingan Alternatif terhadap Kriteria "Keandalan"
    # - Xenia : Avanza = 1/2
    # - Ertiga : Avanza = 4
    # - Grand Livina : Avanza = 3
    # - Ertiga : Xenia = 5
    # - Grand Livina : Xenia = 4
    # - Grand Livina : Ertiga = 1/3
    # -------------------------------------------------------------
    mat_keandalan = np.array(
        [
            [1.0, 2.0, 1.0 / 4.0, 1.0 / 3.0],
            [1.0 / 2.0, 1.0, 1.0 / 5.0, 1.0 / 4.0],
            [4.0, 5.0, 1.0, 3.0],
            [3.0, 4.0, 1.0 / 3.0, 1.0],
        ]
    )

    # -------------------------------------------------------------
    # 5. Matriks Perbandingan Alternatif terhadap Kriteria "Konsumsi BBM"
    # - Xenia : Avanza = 1/2
    # - Ertiga : Avanza = 2
    # - Grand Livina : Avanza = 1/4
    # - Ertiga : Xenia = 1.5 (3/2)
    # - Grand Livina : Xenia = 1/3
    # - Grand Livina : Ertiga = 1/4
    # -------------------------------------------------------------
    mat_bbm = np.array(
        [
            [1.0, 2.0, 1.0 / 2.0, 4.0],
            [1.0 / 2.0, 1.0, 2.0 / 3.0, 3.0],
            [2.0, 3.0 / 2.0, 1.0, 4.0],
            [1.0 / 4.0, 1.0 / 3.0, 1.0 / 4.0, 1.0],
        ]
    )

    # -------------------------------------------------------------
    # 6. Eksekusi Perhitungan AHP
    # -------------------------------------------------------------
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

    spk_app = AHPDecisionMaker(ahp_crit, ahp_alts)
    spk_app.solve()