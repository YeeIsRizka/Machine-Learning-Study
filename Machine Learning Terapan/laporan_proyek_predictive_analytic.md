# Laporan Proyek Machine Learning - Rizka Alfadilla

## Domain Proyek

Pendidikan adalah salah satu faktor kunci dalam pengembangan sumber daya manusia. Namun, kinerja akademik siswa seringkali dipengaruhi oleh kebiasaan sehari-hari seperti pola tidur, waktu belajar, dan penggunaan media sosial. Proyek ini bertujuan untuk menganalisis hubungan antara kebiasaan siswa dengan nilai akademik mereka menggunakan  teknik predictive analytics. Hasilnya dapat membantu pendidik dan siswa dalam mengidentifikasi kebiasaan yang perlu dioptimalkan untuk meningkatkan prestasi akademik.

Referensi: 

Orji, F. A., & Vassileva, J. (2022). Machine learning approach for predicting students academic performance and study strategies based on their motivation. _arXiv preprint arXiv:2210.08186_. https://doi.org/10.48550/arXiv.2210.08186

Rabia, M., Mubarak, N., Tallat, H., & Nasir, W. (2017). A study on study habits and academic performance of students. _International Journal of Asian Social Science, 7_(10), 891-897. https://doi.org/10.18488/journal.1.2017.710.891.897

Strecht, P., Cruz, L., Soares, C., Mendes-Moreira, J., & Abreu, R. (2015). A Comparative Study of Classification and Regression Algorithms for Modelling Students' Academic Performance. _International Educational Data Mining Society_.   

Xu, X., Wang, J., Peng, H., & Wu, R. (2019). Prediction of academic performance associated with internet usage behaviors using machine learning algorithms. _Computers in Human Behavior, 98_, 166-173. https://doi.org/10.1016/j.chb.2019.04.015

## Business Understanding

### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, proyek ini bertujuan untuk mengembangkan sistem prediksi performa akademik mahasiswa untuk menjawab permasalahan berikut:
- Bagaimana pengaruh kebiasaan sehari-hari (studi, tidur, media sosial, diet, dan kesehatan mental) terhadap nilai akhir siswa?
- Faktor apa yang paling dominan memengaruhi nilai akademik siswa?


### Goals
Untuk menjawab pertanyaan tersebut, predictive modelling akan dibangun dengan tujuan atau goals sebagai berikut:
- Membuat model machine learning yang dapat memprediksi performa akademik mahasiswa seakurat mungkin berdasarkan kebiasaan sehari-hari.
- Mengidentifikasi faktor dominan yang dapat diintervensi untuk meningkatkan kinerja akademik.

### Solution statements

- Melakukan analisis statistik dan visualisasi data untuk mengidentifikasi pola, outlier, dan korelasi antara variabel kebiasaan siswa dengan performa akademik.
- Mengimplementasikan algoritma Linear Regression, Random Forest, dan Gradient Boosting untuk membangun model prediktif yang akurat dalam mengklasifikasikan tingkat performa akademik siswa.
- Menggunakan metrik evaluasi seperti MAE (Mean Absolute Error) dan R²-score untuk mengukur efektivitas model regresi.
- Melakukan hyperparameter tuning menggunakan GridSearchCV untuk meningkatkan performa model secara signifikan.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Student Habits vs Academic Performance, dataset ini tersedia secara publik di [Kaggle - Student Habits vs Academic Performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance/data). Dataset ini berisi 1.000 catatan siswa dengan lebih dari 15 variabel yang mengukur berbagai aspek kebiasaan sehari-hari dan performa akademik.

### Variabel-variabel pada dataset:

| **No** | **Fitur**                       | **Deskripsi**                               | **Tipe Data** | **Range/Skala**                       |
| ------ | ------------------------------- | ------------------------------------------- | ------------- | ------------------------------------- |
| 1      | `student_id`                    | ID unik untuk setiap siswa                  | Object        | -                                     |
| 2      | `age`                           | Usia siswa                                  | Numerik       | 17 - 24 tahun                         |
| 3      | `gender`                        | Jenis kelamin siswa                         | Kategori      | Male / Female / Other                         |
| 4      | `study_hours_per_day`           | Jam belajar per hari                        | Numerik       | 0 - 24 jam                            |
| 5      | `social_media_hours`            | Jam penggunaan media sosial per hari        | Numerik       | 0 - 24 jam                            |
| 6      | `netflix_hours`                 | Jam menonton Netflix per hari               | Numerik       | 0 - 24 jam                            |
| 7      | `part_time_job`                 | Apakah siswa memiliki pekerjaan paruh waktu | Kategori      | Yes / No                              |
| 8      | `attendance_percentage`         | Persentase kehadiran siswa                  | Numerik       | 0 - 100%                              |
| 9      | `sleep_hours`                   | Jam tidur per hari                          | Numerik       | 0 - 24 jam                            |
| 10     | `diet_quality`                  | Kualitas diet siswa                         | Kategori      | Poor / Fair / Good       |
| 11     | `exercise_frequency`            | Frekuensi olahraga per minggu               | Numerik       | 0 - 7 kali                            |
| 12     | `parental_education_level`      | Tingkat pendidikan orang tua                | Kategori      | High School / Bachelor / Master |
| 13     | `internet_quality`              | Kualitas internet siswa                     | Kategori      | Poor / Average / Good                   |
| 14     | `mental_health_rating`          | Skor kesehatan mental siswa                 | Numerik       | 1 - 10                                |
| 15     | `extracurricular_participation` | Partisipasi kegiatan ekstrakurikuler        | Kategori      | Yes / No                              |
| 16     | `exam_score`                    | Skor ujian akhir (Target)                   | Numerik       | 0 - 100                               |

### Kondisi dataset:

| Variabel                      | Tipe Data | Jumlah Missing | Jumlah Unik | Nilai Unik Contoh                     |
|-------------------------------|-----------|----------------|-------------|---------------------------------------|
| `student_id`                  | object    | 0              | 1000        | [S1000, S1001, S1002, S1003, S1004]  |
| `age`                         | int64     | 0              | 8           | [23, 20, 21, 19, 24]                 |
| `gender`                      | object    | 0              | 3           | [Female, Male, Other]                 |
| `study_hours_per_day`         | float64   | 0              | 78          | [0.0, 6.9, 1.4, 1.0, 5.0]            |
| `social_media_hours`          | float64   | 0              | 60          | [1.2, 2.8, 3.1, 3.9, 4.4]            |
| `netflix_hours`               | float64   | 0              | 51          | [1.1, 2.3, 1.3, 1.0, 0.5]            |
| `part_time_job`               | object    | 0              | 2           | [No, Yes]                             |
| `attendance_percentage`       | float64   | 0              | 320         | [85.0, 97.3, 94.8, 71.0, 90.9]       |
| `sleep_hours`                 | float64   | 0              | 68          | [8.0, 4.6, 9.2, 4.9, 7.4]            |
| `diet_quality`                | object    | 0              | 3           | [Fair, Good, Poor]                    |
| `exercise_frequency`          | int64     | 0              | 7           | [6, 1, 4, 3, 2]                       |
| `parental_education_level`    | object    | 91             | 3           | [Master, High School, Bachelor, nan]  |
| `internet_quality`            | object    | 0              | 3           | [Average, Poor, Good]                 |
| `mental_health_rating`        | int64     | 0              | 10          | [8, 1, 4, 10, 3]                      |
| `extracurricular_participation` | object  | 0              | 2           | [Yes, No]                             |
| `exam_score`                  | float64   | 0              | 480         | [56.2, 100.0, 34.3, 26.8, 66.4]      |

Dataset terdiri dari 1.000 sampel mahasiswa dengan 16 variabel. Tidak ada nilai yang duplikat dan hilang kecuali pada `parental_education_level` yang memiliki 91 missing values. Sebagian besar fitur sudah bersih dan siap untuk analisis.Fitur `parental_education_level` dilakukan imputasi dengan menambahkan nilai "Unknown" untuk mengatasi missing values yang ada.

### Exploratory Data Analysis (EDA):

<div align="center">
  <img src="Extrakulikuler Participation.png" width="400"/>
</div>

68% mahasiswa tidak mengikuti kegiatan ekstrakurikuler Ini menunjukkan keterlibatan non-akademik tergolong rendah dan bisa menjadi indikator keseimbangan antara kegiatan akademik dan sosial.

<div align="center">
  <img src="Part Time Job.png" width="400"/>
</div>

78% mahasiswa tidak memiliki pekerjaan paruh waktu. Hal ini mengindikasikan bahwa sebagian besar mahasiswa fokus pada studi, namun juga memberi peluang untuk menganalisis dampak pekerjaan terhadap performa belajar.

<div align="center">
  <img src="Distribusi Fitur Numerik.png" width="400"/>
</div>

`exam_score` dan `attendance_percentage` cenderung left-skewed, menunjukkan mayoritas siswa memiliki nilai tinggi dan sering masuk. Fitur seperti `study_hours_per_day` dan `sleep_hours` memiliki distribusi mendekati normal, mencerminkan variasi alami. Sementara itu, `netflix_hours` dan `social_media_hours` bersifat right-skewed, menandakan sebagian besar mahasiswa hanya sedikit melakukannya. Sisanya tersebar merata, menunjukkan data yang representatif.

<div align="center">
  <img src="Hubungan Antar Fitur Numerik.png" width="400"/>
</div>

<div align="center">
  <img src="Matriks Korelasi.png" width="400"/>
</div>

Fitur `study_hours_per_day` menunjukkan korelasi sangat kuat terhadap `exam_score` dengan nilai +0.83, menandakan bahwa semakin lama waktu belajar per hari, semakin tinggi kemungkinan nilai ujian mahasiswa. Sebaliknya, `social_media_hours` dan `netflix_hours` berkorelasi negatif (-0.17) dengan `exam_score`, mengindikasikan bahwa penggunaan waktu berlebih pada aktivitas ini mungkin berdampak pada penurunan performa akademik. Fitur lainnya seperti `age`tidak memiliki pengaruh korelatif yang signifikan terhadap skor ujian.

## Data Preparation
Pada tahap ini, dilakukan beberapa proses data preparation agar dataset siap digunakan untuk proses training model machine learning. Proses yang dilakukan adalah sebagai berikut:

### Menghapus Kolom yang Tidak Relevan:
- `student_id`: Kolom ini hanya berfungsi sebagai identitas unik siswa dan tidak memiliki pengaruh langsung terhadap target exam_score.
- `age`: Berdasarkan analisis korelasi, kolom ini memiliki korelasi yang sangat rendah dengan exam_score (-0.01). Oleh karena itu, kolom ini dihapus untuk mengurangi noise dalam data.

### Label Encoding:
- Label encoding diterapkan pada fitur ordinal untuk mempertahankan informasi urutan (ranking).
- Fitur yang diencoding adalah:
  - `diet_quality`: Memiliki skala ordinal (Poor, Fair, Good, Excellent)
  - `parental_education_level`: Memiliki urutan tertentu (None, High School, Bachelor, Master).
  - `internet_quality`: Memiliki skala kualitas (Low, Medium, High).
- Fitur-fitur ini memiliki urutan nilai yang harus dipertahankan, sehingga menggunakan One Hot Encoding akan menyebabkan hilangnya informasi urutan tersebut.

### One Hot Encoding:
- One Hot Encoding diterapkan pada fitur nominal yang tidak memiliki urutan tertentu.
- Fitur yang diencoding adalah:
  - gender: Male, Female, Other.
  - part_time_job: Yes, No.
  - extracurricular_participation: Yes, No.
- Fitur-fitur ini merupakan kategori non-ordinal sehingga tidak ada hubungan urutan antara kategori tersebut. Menggunakan Label Encoding pada fitur ini dapat menyebabkan model mengasumsikan adanya hubungan urutan (`0 < 1 < 2`), yang sebenarnya tidak ada.

### Standarisasi:
- Standarisasi dilakukan pada fitur numerik agar semua fitur berada pada skala yang sama (`mean = 0, std = 1`).
- Fitur yang distandarisasi adalah:
  - `study_hours_per_day`, `social_media_hours`, `netflix_hours`, `attendance_percentage`, `sleep_hours`, `exercise_frequency`, `mental_health_rating`.
- Standarisasi diperlukan untuk memastikan model berbasis gradien (misalnya, Linear Regression dan Gradient Boosting) dapat bekerja optimal. Jika fitur memiliki skala yang berbeda-beda, maka fitur dengan skala besar akan mendominasi model, meskipun pengaruhnya sebenarnya tidak signifikan.

### Split Dataset:
- Dataset dibagi menjadi training set (80%) dan testing set (20%).
  - X (fitur): Semua kolom kecuali `exam_score`.
  - y (target): Kolom `exam_score`.
- Pembagian ini dilakukan untuk memastikan model tidak overfitting terhadap data training. Data testing digunakan untuk mengevaluasi performa model pada data baru yang tidak pernah dilihat sebelumnya.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Pada tahap ini, dilakukan proses pengembangan model machine learning untuk memprediksi exam_score berdasarkan fitur-fitur yang ada. Algoritma yang digunakan adalah Linear Regression, Random Forest Regressor, dan Gradient Boosting Regressor.

### Model yang Digunakan dan Alasan Pemilihan:

| **Model**                       | **Alasan Pemilihan**                                                                                              |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Linear Regression**           | Model sederhana yang dapat digunakan sebagai baseline untuk membandingkan performa model lain.                    |
| **Random Forest Regressor**     | Model berbasis pohon keputusan yang efektif dalam menangani non-linearitas data dan menangani outliers.           |
| **Gradient Boosting Regressor** | Model boosting yang menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting. |

### Kelebihan dan Kekurangan Setiap Algoritma:

| **Model**             | **Kelebihan**                                                      | **Kekurangan**                                                 |
| --------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------- |
| **Linear Regression** | Sederhana, cepat, dan interpretatif.                               | Tidak efektif untuk data non-linear.                           |
| **Random Forest**     | Dapat menangani data non-linear, tidak sensitif terhadap outliers. | Cenderung overfitting jika tidak dituning dengan baik.         |
| **Gradient Boosting** | Sangat akurat, efektif untuk data non-linear.                      | Lebih lambat, rentan terhadap overfitting jika tidak dituning. |

### Parameter yang Dituning dan Alasan Pemilihannya:

### a. Linear Regression:
- `fit_intercept`: Apakah akan menghitung intercept.
- `copy_X`: Menjaga salinan X agar tidak dimodifikasi.
- `positive`: Menjaga koefisien agar tetap positif.
- Alasan: Untuk model linear, parameter-parameter tersebut dapat membantu mengatur koefisien agar lebih stabil dan interpretatif.

### b. Random Forest Regressor:
- `n_estimators`: Jumlah pohon yang digunakan.
- `max_depth`: Kedalaman maksimum pohon untuk menghindari overfitting.
- `min_samples_split`: Jumlah minimum sampel untuk melakukan split.
- `min_samples_leaf`: Jumlah minimum sampel pada daun pohon.  
- Alasan: Tuning pada parameter `n_estimators` dapat meningkatkan akurasi model dengan menambahkan lebih banyak pohon. `max_depth` dan `min_samples_split` mengontrol kompleksitas pohon agar tidak overfitting.

### c. Gradient Boosting Regressor:
- `n_estimators`: Jumlah pohon yang digunakan.
- `max_depth`: Kedalaman maksimum pohon.
- `learning_rate`: Kecepatan pembelajaran untuk memperbarui pohon berikutnya.
- `subsample`: Rasio sampel yang digunakan untuk melatih setiap pohon.
- Alasan: `learning_rate` dapat mengontrol kecepatan belajar agar model tidak overfitting. subsample dapat digunakan untuk mengontrol overfitting dengan mengambil subset data.

### Hasil Hyperparameter Tuning:
Hyperparameter tuning adalah proses mencari kombinasi parameter terbaik untuk setiap model agar dapat meningkatkan performa model. Pada proses ini, digunakan GridSearchCV dengan evaluasi menggunakan metrik Mean Squared Error (MSE).

| Model            | Best Params |
|------------------|-------------|
| Linear Regression | `{'copy_X': True, 'fit_intercept': True, 'positive': False}` |
| Random Forest     | `{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}` |
| Gradient Boosting | `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}` |

## Evaluation
Pada tahap ini, dilakukan evaluasi terhadap model menggunakan beberapa metrik regresi yang sesuai dengan konteks prediksi skor ujian (`exam_score`). Metrik evaluasi yang digunakan adalah Mean Squared Error (MSE), Mean Absolute Error (MAE), dan R² Score (Coefficient of Determination).

### Metrik Evaluasi yang Digunakan:

* **MSE (Mean Squared Error):**

  $$
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
  $$

* **MAE (Mean Absolute Error):**

  $$
  MAE = \frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y_i} |
  $$

* **R² Score:**

  $$
  R^2 = 1 - \frac{\sum (y_i - \hat{y_i})^2}{\sum (y_i - \bar{y})^2}
  $$

| **Metrik**           | **Penjelasan**                                                                                                                                                                   |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **MSE (Mean Squared Error)**  | MSE mengukur rata-rata kuadrat selisih antara nilai aktual dan nilai prediksi. Semakin kecil nilai MSE, semakin baik model tersebut dalam melakukan prediksi.            |
| **MAE (Mean Absolute Error)** | MAE mengukur rata-rata absolut selisih antara nilai aktual dan prediksi. MAE lebih mudah diinterpretasikan karena memiliki satuan yang sama dengan target (`exam_score`). |
| **R² Score**         | R² Score mengukur seberapa baik model dapat menjelaskan variabilitas data. Nilai R² berkisar antara 0 hingga 1. Semakin mendekati 1, semakin baik model tersebut dalam menjelaskan data. |

### Hasil Evaluasi Model:

Berikut adalah hasil evaluasi model berdasarkan metrik MSE, MAE, dan R² Score:

| **Model**             | **Train MSE** | **Test MSE** | **Train MAE** | **Test MAE** | **Train R²** | **Test R²** |
| --------------------- | ------------- | ------------ | ------------- | ------------ | ------------ | ----------- |
| **Linear Regression** | 28.69         | 26.24        | 4.22          | 4.15         | 0.90         | 0.90        |
| **Random Forest**     | 6.60          | 38.39        | 2.07          | 4.94         | 0.98         | 0.85        |
| **Gradient Boosting** | 14.15         | 28.23        | 3.00          | 4.42         | 0.95         | 0.89        |

### Analisis Hasil Evaluasi:

* **Linear Regression:**

  * Train MSE (`28.69`) dan Test MSE (`26.24`) relatif stabil, menunjukkan model tidak overfitting.
  * R² Test (`0.90`) menunjukkan bahwa model dapat menjelaskan **90% variabilitas data**, yang merupakan hasil yang sangat baik.
  * MAE (`4.15`) menunjukkan bahwa rata-rata kesalahan prediksi adalah sekitar **4.15 poin** dari skor ujian aktual.

* **Random Forest:**

  * Train MSE (`6.60`) sangat rendah, tetapi Test MSE (`38.39`) cukup tinggi, menunjukkan adanya **overfitting**.
  * Meskipun Train R² (`0.98`) sangat tinggi, Test R² (`0.85`) menunjukkan bahwa model kurang mampu menangkap pola pada data testing.
  * MAE (`4.94`) juga lebih tinggi daripada Linear Regression dan Gradient Boosting.

* **Gradient Boosting:**

  * Train MSE (`14.15`) lebih rendah daripada Linear Regression, tetapi Test MSE (`28.23`) masih lebih tinggi.
  * R² Test (`0.89`) mendekati hasil Linear Regression (`0.90`), menunjukkan bahwa model ini juga dapat menangkap pola data dengan baik.
  * MAE (`4.42`) berada di antara Linear Regression (`4.15`) dan Random Forest (`4.94`), menunjukkan bahwa model ini **lebih stabil** daripada Random Forest.

### Kesimpulan dan Model Terbaik:

* Berdasarkan hasil evaluasi, model **Linear Regression** menunjukkan hasil yang stabil dan tidak overfitting.
* Meskipun **Random Forest** memiliki Train MSE yang rendah (`6.60`), namun hasil Test MSE (`38.39`) menunjukkan adanya overfitting yang cukup parah.
* **Gradient Boosting** memberikan hasil yang cukup baik (`Test MSE: 28.23`), namun Linear Regression tetap lebih stabil (`Test MSE: 26.24`).
* Oleh karena itu, model terbaik yang dipilih adalah **Linear Regression** karena model ini mampu memberikan hasil yang stabil antara data training dan testing, serta R² yang tinggi (`0.90`).