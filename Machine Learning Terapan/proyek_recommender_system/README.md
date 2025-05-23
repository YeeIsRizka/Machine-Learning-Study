# Laporan Proyek Machine Learning - Rizka Alfadilla

## Project Overview

Minat terhadap board game mengalami pertumbuhan signifikan dalam dua dekade terakhir. Dalam pasar hiburan global, board game kini menempati porsi yang semakin besar seiring munculnya klub-klub lokal dan toko khusus yang tersebar di berbagai negara (Ion et al., 2020; Zalewski et al., 2019). Dengan lebih dari 100.000 judul permainan dan jutaan pengguna terdaftar, situs BoardGameGeek (BGG) menjadi sumber data yang sangat kaya untuk memahami preferensi pengguna terhadap board game (Sahinli et al., 2020).

Namun, meskipun terdapat kebutuhan nyata untuk rekomendasi permainan yang tepat, kebanyakan pengguna masih mengandalkan forum atau ulasan dari tokoh tertentu yang cenderung bias pada judul-judul populer. Pendekatan manual ini tidaklah efisien, tidak dapat diskalakan, dan cenderung mengabaikan permainan yang kurang dikenal namun berkualitas (Sahinli et al., 2020; Zalewski et al., 2019).

Untuk mengatasi tantangan tersebut, proyek ini bertujuan membangun sistem rekomendasi board game berbasis data yang dapat membantu pengguna menemukan permainan yang sesuai dengan selera mereka. Dua pendekatan utama yang diterapkan dalam proyek ini adalah content-based filtering, yang memanfaatkan kemiripan konten (kategori permainan), dan collaborative filtering, yang mengandalkan riwayat interaksi pengguna. Studi-studi terdahulu telah menunjukkan bahwa sistem rekomendasi untuk board game dapat dikembangkan melalui kedua pendekatan diatas, atau menggunakan metode lanjutan seperti klasterisasi dan embedding berbasis deep learning (Ion et al., 2020; Sahinli et al., 2020; Zalewski et al., 2019).

 Penerapan sistem rekomendasi ini tidak hanya meningkatkan pengalaman pengguna dalam menemukan permainan yang sesuai dengan preferensi mereka, tetapi juga membuka peluang bagi pengembang board game untuk menjangkau audiens yang lebih luas. Dengan demikian, sistem ini berpotensi mengurangi dominasi judul-judul populer dan mendorong eksplorasi terhadap permainan yang kurang dikenal, sehingga menciptakan ekosistem board game yang lebih inklusif dan beragam.

### Referensi

Ion, M., Sacharidis, D., & Werthner, H. (2020). Designing a recommender system for board games. *Proceedings of the 35th Annual ACM Symposium on Applied Computing*, 1465–1467. https://doi.org/10.1145/3341105.3375780

Sahinli, C., Debrauwer, F., Brugaletta, L., Martoja, T., Mishra, V., Ruppenthal, Y., & Browne, C. (2020). Recommender system for board games. *Context*, *1*, 3.

Zalewski, J., Ganzha, M., & Paprzycki, M. (2019). Recommender system for board games. *2019 23rd International Conference on System Theory, Control and Computing (ICSTCC)*, 249–254. https://doi.org/10.1109/ICSTCC.2019.8885455


## Business Understanding

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, proyek ini bertujuan untuk mengembangkan sistem rekomendasi board game guna menjawab permasalahan berikut:
- Bagaimana cara merekomendasikan board game yang relevan kepada pengguna berdasarkan kategori atau jenis permainan?
- Bagaimana sistem dapat memberikan rekomendasi yang dipersonalisasi berdasarkan riwayat interaksi dan preferensi pengguna?

### Goals

Untuk menjawab pertanyaan tersebut, sistem rekomendasi akan dibangun dengan tujuan sebagai berikut:
- Mengembangkan sistem rekomendasi yang mampu menyarankan board game berdasarkan informasi deskriptif, seperti kategori permainan.
- Membangun sistem rekomendasi yang mampu menyarankan board game berdasarkan pola interaksi mereka terhadap board game sebelumnya.


### Solution Approach

Untuk mencapai tujuan yang telah ditetapkan, proyek ini mengusulkan dua pendekatan utama dalam membangun sistem rekomendasi board game:

1. **Content-Based Filtering**  
   Pendekatan ini merekomendasikan board game berdasarkan kemiripan konten, khususnya kategori permainan. Setiap board game direpresentasikan dalam bentuk fitur deskriptif (seperti genre atau kategori) yang kemudian diolah menggunakan teknik *TF-IDF vectorization*. Kemiripan antar game dihitung menggunakan *cosine similarity*, sehingga sistem dapat menyarankan game yang memiliki karakteristik serupa dengan game yang disukai oleh pengguna.

2. **Collaborative Filtering**  
   Pendekatan ini memanfaatkan pola interaksi pengguna terhadap berbagai board game. Model *collaborative filtering* dibangun menggunakan arsitektur *neural network embedding* untuk memetakan pengguna dan game ke dalam ruang vektor berdimensi rendah. Dengan mempelajari pola rating yang diberikan oleh pengguna terhadap berbagai game, model ini dapat memprediksi skor preferensi terhadap game yang belum dimainkan, dan menghasilkan rekomendasi yang dipersonalisasi.

Kedua pendekatan ini dirancang untuk saling melengkapi: pendekatan berbasis konten mengatasi masalah *cold start* pada game baru yang belum banyak diulas, sementara pendekatan berbasis kolaborasi mampu menangkap preferensi pengguna secara lebih personal berdasarkan pola historis interaksinya.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah *BoardGameGeek Reviews Dataset*, yang tersedia secara publik di [Kaggle - BoardGameGeek Reviews](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews/data). Dataset ini terdiri dari dua file utama yang digunakan dalam proyek ini:

1. **`games_detailed_info.csv`**  
   Merupakan metadata dari board game, berisi **55** variabel yang mencakup **21.631 entri**. File ini berisi berbagai atribut seperti nama permainan, kategori, jumlah pemain, waktu bermain, tahun rilis, tingkat kesulitan, dan statistik rating lainnya. Informasi ini sangat penting dalam membangun sistem rekomendasi berbasis konten.

2. **`bgg-15m-reviews.csv`**  
   Berisi data ulasan pengguna terhadap berbagai board game, terdiri dari 5 variabel dengan total **15.823.269 entri**. Setiap entri merepresentasikan interaksi pengguna berupa rating terhadap suatu permainan. Data ini digunakan dalam pendekatan collaborative filtering untuk mempelajari pola preferensi pengguna berdasarkan riwayat interaksi mereka.

Kedua file tersebut saling melengkapi, di mana `games_detailed_info.csv` digunakan untuk pendekatan berbasis konten, sementara `bgg-15m-reviews.csv` digunakan untuk pendekatan berbasis kolaborasi. Kombinasi keduanya memungkinkan sistem rekomendasi yang lebih akurat dan adaptif terhadap preferensi pengguna.

### Variabel-variabel pada dataset: `games_detailed_info.csv`

| **No** | **Fitur**                       | **Deskripsi**                                                     |
| ------ | ------------------------------- | ----------------------------------------------------------------- |
| 1      | `type`                          | Jenis entitas (umumnya "boardgame")                               |
| 2      | `id`                            | ID unik board game                                                |
| 3      | `thumbnail`                     | URL gambar kecil dari board game                                  |
| 4      | `image`                         | URL gambar utama dari board game                                  |
| 5      | `primary`                       | Nama utama dari board game                                        |
| 6      | `alternate`                     | Nama alternatif (jika ada)                                        |
| 7      | `description`                   | Deskripsi teks panjang dari game                                  |
| 8      | `yearpublished`                 | Tahun rilis pertama                                               |
| 9      | `minplayers`                    | Jumlah minimum pemain                                             |
| 10     | `maxplayers`                    | Jumlah maksimum pemain                                            |
| 11     | `suggested_num_players`         | Rekomendasi jumlah pemain (string JSON-like)                      |
| 12     | `suggested_playerage`           | Usia pemain yang disarankan                                       |
| 13     | `suggested_language_dependence` | Tingkat ketergantungan terhadap bahasa                            |
| 14     | `playingtime`                   | Estimasi total waktu bermain                                      |
| 15     | `minplaytime`                   | Waktu bermain minimum                                             |
| 16     | `maxplaytime`                   | Waktu bermain maksimum                                            |
| 17     | `minage`                        | Usia minimum pemain                                               |
| 18     | `boardgamecategory`             | Kategori atau genre game                                          |
| 19     | `boardgamemechanic`             | Mekanik permainan                                                 |
| 20     | `boardgamefamily`               | Keluarga game (tema umum atau seri)                               |
| 21     | `boardgameexpansion`            | Nama-nama ekspansi game (jika ada)                                |
| 22     | `boardgameimplementation`       | Implementasi ulang game dari versi sebelumnya                     |
| 23     | `boardgamedesigner`             | Nama desainer game                                                |
| 24     | `boardgameartist`               | Nama ilustrator game                                              |
| 25     | `boardgamepublisher`            | Nama penerbit                                                     |
| 26     | `usersrated`                    | Jumlah pengguna yang memberikan rating                            |
| 27     | `average`                       | Rata-rata rating dari pengguna                                    |
| 28     | `bayesaverage`                  | Rata-rata rating berbobot (Bayesian average)                      |
| 29     | `Board Game Rank`               | Peringkat keseluruhan dalam kategori Board Game                   |
| 30     | `Strategy Game Rank`            | Peringkat di kategori Strategy                                    |
| 31     | `Family Game Rank`              | Peringkat di kategori Family                                      |
| 32     | `stddev`                        | Standar deviasi rating                                            |
| 33     | `median`                        | Nilai tengah rating                                               |
| 34     | `owned`                         | Jumlah pengguna yang memiliki game tersebut                       |
| 35     | `trading`                       | Jumlah pengguna yang menandai game sebagai tersedia untuk ditukar |
| 36     | `wanting`                       | Jumlah pengguna yang ingin memiliki game tersebut                 |
| 37     | `wishing`                       | Jumlah pengguna yang menaruh harapan pada game                    |
| 38     | `numcomments`                   | Jumlah komentar dari pengguna                                     |
| 39     | `numweights`                    | Jumlah rating terhadap tingkat kesulitan                          |
| 40     | `averageweight`                 | Rata-rata tingkat kesulitan game                                  |
| 41     | `boardgameintegration`          | Integrasi ke dalam game lain                                      |
| 42     | `boardgamecompilation`          | Bagian dari kompilasi atau koleksi                                |
| 43     | `Party Game Rank`               | Peringkat dalam kategori Party Game                               |
| 44     | `Abstract Game Rank`            | Peringkat dalam kategori Abstract Game                            |
| 45     | `Thematic Rank`                 | Peringkat dalam kategori Thematic Game                            |
| 46     | `War Game Rank`                 | Peringkat dalam kategori War Game                                 |
| 47     | `Customizable Rank`             | Peringkat dalam kategori Customizable Game                        |
| 48     | `Children's Game Rank`          | Peringkat dalam kategori Children's Game                          |
| 49     | `RPG Item Rank`                 | Peringkat untuk item RPG                                          |
| 50     | `Accessory Rank`                | Peringkat untuk item aksesori game                                |
| 51     | `Video Game Rank`               | Peringkat untuk game video                                        |
| 52     | `Amiga Rank`                    | Peringkat untuk game di platform Amiga                            |
| 53     | `Commodore 64 Rank`             | Peringkat untuk game di platform Commodore 64                     |
| 54     | `Arcade Rank`                   | Peringkat untuk game di platform Arcade                           |
| 55     | `Atari ST Rank`                 | Peringkat untuk game di platform Atari ST                         |


### Variabel-variabel pada dataset: `bgg-15m-reviews.csv`

| **No** | **Fitur** | **Deskripsi**                                                                |
| ------ | --------- | ---------------------------------------------------------------------------- |
| 1      | `user`    | ID atau nama pengguna yang memberikan ulasan dan rating terhadap board game  |
| 2      | `rating`  | Skor atau penilaian numerik yang diberikan oleh pengguna terhadap board game |
| 3      | `comment` | Komentar atau ulasan teks dari pengguna                                      |
| 4      | `ID`      | ID unik dari board game yang diulas (mengacu ke `games_detailed_info.csv`)   |
| 5      | `name`    | Nama board game yang diulas                                                  |

### Kondisi dataset: `games_detailed_info.csv`

| **No** | **Fitur**                       | **Tipe Data** | **Jumlah Data** | **Missing** | **Unik** | **Contoh Nilai Unik**     |
| ------ | ------------------------------- | ------------- | --------------- | ----------- | -------- | ------------------------- |
| 1      | `type`                          | object        | 21631           | 0           | 1        | `boardgame`               |
| 2      | `id`                            | int64         | 21631           | 0           | 21631    | `30549`, `822`, `13`      |
| 3      | `thumbnail`                     | object        | 21631           | 15          | 21615    | `https://...`             |
| 4      | `image`                         | object        | 21631           | 15          | 21615    | `https://...`             |
| 5      | `primary`                       | object        | 21631           | 0           | 21236    | `Pandemic`, `Catan`       |
| 6      | `alternate`                     | object        | 8850            | 12781       | 8836     | `[Pandemia, EPIZOotic]`   |
| 7      | `description`                   | object        | 21630           | 1           | 21615    | `In Pandemic...`          |
| 8      | `yearpublished`                 | int64         | 21631           | 0           | 190      | `2008`, `1995`            |
| 9      | `minplayers`                    | int64         | 21631           | 0           | 11       | `2`, `3`, `1`             |
| 10     | `maxplayers`                    | int64         | 21631           | 0           | 52       | `4`, `5`, `7`             |
| 11     | `suggested_num_players`         | object        | 21631           | 0           | 14955    | `[{'@numplayers': '1'}]`  |
| 12     | `suggested_playerage`           | object        | 21520           | 111         | 3540     | `[{'@value': '2'}]`       |
| 13     | `suggested_language_dependence` | object        | 21559           | 72          | 4853     | `[{'@level': '6'}]`       |
| 14     | `playingtime`                   | int64         | 21631           | 0           | 119      | `45`, `60`, `120`         |
| 15     | `minplaytime`                   | int64         | 21631           | 0           | 89       | `30`, `45`, `15`          |
| 16     | `maxplaytime`                   | int64         | 21631           | 0           | 119      | `120`, `60`, `30`         |
| 17     | `minage`                        | int64         | 21631           | 0           | 21       | `8`, `10`, `14`           |
| 18     | `boardgamecategory`             | object        | 21348           | 283         | 6730     | `['Medical']`, ...        |
| 19     | `boardgamemechanic`             | object        | 20041           | 1590        | 8291     | `['Hand Management']`     |
| 20     | `boardgamefamily`               | object        | 17870           | 3761        | 11285    | `['Components: Map']`     |
| 21     | `boardgameexpansion`            | object        | 5506            | 16125       | 5264     | `['Z-Force Team']`        |
| 22     | `boardgameimplementation`       | object        | 4862            | 16769       | 4247     | `['Legacy: Season 0']`    |
| 23     | `boardgamedesigner`             | object        | 21035           | 596         | 9136     | `['Matt Leacock']`        |
| 24     | `boardgameartist`               | object        | 15724           | 5907        | 9080     | `['Josh Cappel']`         |
| 25     | `boardgamepublisher`            | object        | 21630           | 1           | 11265    | `['Z-Man Games']`         |
| 26     | `usersrated`                    | int64         | 21631           | 0           | 3092     | `109006`, `81582`         |
| 27     | `average`                       | float64       | 21631           | 0           | 20129    | `7.58896`, `7.41837`      |
| 28     | `bayesaverage`                  | float64       | 21631           | 0           | 15735    | `7.48669`, `6.96965`      |
| 29     | `Board Game Rank`               | object        | 21631           | 0           | 21627    | `106`, `191`, `429`       |
| 30     | `Strategy Game Rank`            | float64       | 2337            | 19294       | 2337     | `121.0`, `6.0`            |
| 31     | `Family Game Rank`              | float64       | 2327            | 19304       | 2327     | `18.0`, `131.0`           |
| 32     | `stddev`                        | float64       | 21631           | 0           | 19199    | `1.32857`, `1.42343`      |
| 33     | `median`                        | int64         | 21631           | 0           | 1        | `0`                       |
| 34     | `owned`                         | int64         | 21631           | 0           | 4226     | `168364`, `106956`        |
| 35     | `trading`                       | int64         | 21631           | 0           | 610      | `2508`, `1567`            |
| 36     | `wanting`                       | int64         | 21631           | 0           | 706      | `625`, `1010`             |
| 37     | `wishing`                       | int64         | 21631           | 0           | 1797     | `9344`, `12105`           |
| 38     | `numcomments`                   | int64         | 21631           | 0           | 1644     | `17305`, `14553`          |
| 39     | `numweights`                    | int64         | 21631           | 0           | 778      | `5597`, `7526`            |
| 40     | `averageweight`                 | float64       | 21631           | 0           | 3908     | `2.4063`, `2.3542`        |
| 41     | `boardgameintegration`          | object        | 1731            | 19900       | 1607     | `['Carcassonne: ...']`    |
| 42     | `boardgamecompilation`          | object        | 827             | 20804       | 573      | `['Carcassonne Big Box']` |
| 43     | `Party Game Rank`               | float64       | 647             | 20984       | 647      | `2.0`, `21.0`             |
| 44     | `Abstract Game Rank`            | float64       | 1109            | 20522       | 1109     | `1.0`, `18.0`             |
| 45     | `Thematic Rank`                 | float64       | 1235            | 20396       | 1235     | `1.0`, `214.0`            |
| 46     | `War Game Rank`                 | float64       | 3501            | 18130       | 3501     | `2.0`, `64.0`             |
| 47     | `Customizable Rank`             | float64       | 301             | 21330       | 301      | `18.0`, `3.0`             |
| 48     | `Children's Game Rank`          | object        | 876             | 20755       | 876      | `16.0`, `849.0`           |
| 49     | `RPG Item Rank`                 | float64       | 1               | 21630       | 1        | `347.0`                   |
| 50     | `Accessory Rank`                | float64       | 1               | 21630       | 1        | `140.0`                   |
| 51     | `Video Game Rank`               | float64       | 1               | 21630       | 1        | `5374.0`                  |
| 52     | `Amiga Rank`                    | float64       | 1               | 21630       | 1        | `227.0`                   |
| 53     | `Commodore 64 Rank`             | float64       | 1               | 21630       | 1        | `180.0`                   |
| 54     | `Arcade Rank`                   | float64       | 1               | 21630       | 1        | `170.0`                   |
| 55     | `Atari ST Rank`                 | float64       | 1               | 21630       | 1        | `140.0`                   |

Nilai duplikat tidak ditemukan berdasarkan kolom `id`. Sebagian besar fitur sudah cukup bersih dan siap untuk dianalisis, terdapat beberapa fitur utama yang dapat digunakan seperti `id`, `primary`, `yearpublished`, `minplayers`, `maxplayers`, `playingtime`, `boardgamemechanic`, dan `boardgamecategory`. Untuk mengatasi missing values pada fitur teks seperti `boardgamemechanic` dan `boardgamecategory`, baris dengan nilai kosong pada kolom tersebut harus dihapus karena fitur ini bersifat penting yang dapat digunakan proses pemodelan nanti.

### Kondisi dataset: `bgg-15m-reviews.csv`

| **No** | **Fitur** | **Tipe Data** | **Jumlah Data** | **Missing** | **Unik**  | **Contoh Nilai Unik**                                       |
| ------ | --------- | ------------- | --------------- | ----------- | --------- | ----------------------------------------------------------- |
| 1      | `user`    | object        | 15.823.203      | 66          | 351.048   | `Torsten`, `mitnachtKAUBO-I`, `Mike Mayer`                  |
| 2      | `rating`  | float64       | 15.823.269      | 0           | 10.172    | `10.0`, `9.9`, `9.8`, `9.75`, `9.5`                         |
| 3      | `comment` | object        | 2.995.022       | 12.828.247  | 2.733.414 | `Hands down my favorite new game of BGG CON 2016...`        |
| 4      | `ID`      | int64         | 15.823.269      | 0           | 19.330    | `30549`, `822`, `13`, `68448`, `36218`                      |
| 5      | `name`    | object        | 15.823.269      | 0           | 18.984    | `Pandemic`, `Carcassonne`, `Catan`, `7 Wonders`, `Dominion` |

Sebagian besar data telah bersih, namun terdapat **66 missing values** pada kolom `user`, serta **sekitar 12,8 juta missing values** pada kolom `comment`, yang wajar mengingat tidak semua pengguna menulis komentar saat memberikan rating. Tidak ditemukan nilai duplikat berdasarkan kombinasi `user` dan `ID`. Untuk kebutuhan analisis dan modeling, kolom `comment` dapat diabaikan, sementara kolom `user` dengan nilai kosong dapat dihapus atau diberi label `"anonymous"`.

### Exploratory Data Analysis (EDA):

<div align="center">
  <img src="https://github.com/user-attachments/assets/36a40eb2-cb00-438e-ba70-fde67f5b5d6b" height="300"/>
  <img src="https://github.com/user-attachments/assets/9091cc29-04d9-4762-9dc3-cb259eb72622" height="300"/>
</div>
‎ 

 Terlihat bahwa distribusi bersifat **right-skewed**, dengan puncak-puncak yang sangat tinggi pada nilai **6**, **7**, dan **8**. Ini menandakan bahwa mayoritas pengguna memberikan rating yang cukup tinggi, dan cenderung menghindari nilai rendah.Distribusi ini juga menunjukkan adanya pola **diskrit** yang sangat khas, mengindikasikan bahwa pengguna lebih sering memberikan rating bulat seperti **6.0**, **7.0**, atau **8.0** namun juga ada desimal seperti **6.3** atau **7.6**. Hal ini bisa berdampak pada model prediktif, terutama dalam pemilihan metrik evaluasi dan teknik normalisasi.

‎ 
<div align="center">
  <img src="https://github.com/user-attachments/assets/70d694cf-75ab-4bc0-84c5-19ae975a0908" width="400"/>
</div>
‎ 

Tiga game teratas *Pandemic*, *Carcassonne*, *Catan* memiliki jumlah user yang melakukan rating hampir sama dan jauh melampaui game lain, menunjukkan dominasi dalam hal popularitas dan penetrasi pasar. Popularitas tinggi bisa menjadi indikator kualitas, tetapi juga bisa menutupi keberadaan game lain yang bagus namun kurang terekspos. Game seperti *Pandemic*, *Carcassonne*, dan *Catan* memiliki basis pengguna yang sangat besar dan aktif, sehingga perlu diperhatikan lebih apakah mungkin dapat mempengaruhi sistem rekomendasi secara signifikan atau tidak.

‎ 
<div align="center">
  <img src="https://github.com/user-attachments/assets/d25f929b-dbef-457d-b0c3-b7f448850674" width="400"/>
</div>
‎ 

Dalam dataset `bgg-15m-reviews.csv`, terdapat variasi ekstrem terhadap seberapa sering pengguna pernah melakukan rating. Sebagian besar pengguna hanya pernah melakukan 1–2 rating, sementara sebagian kecil lainnya pernah melakukan ribuan rating. Untuk mengukur dan menangani ketidakseimbangan ini, dilakukan analisis distribusi dan deteksi **outlier** menggunakan metode **Interquartile Range (IQR)**. Hasilnya menunjukkan terdapat **38.214** pengguna yang dianggap sebagai outlier yang mencakup **9.673.842** data, artinya sekitar 61% dari total data review berasal dari pengguna **outlier**.


## Data Preparation
Tahap ini bertujuan untuk mempersiapkan data sebelum digunakan dalam pelatihan model. Beberapa langkah pembersihan dan transformasi dilakukan untuk memastikan kualitas data dan meningkatkan performa sistem rekomendasi yang dibangun.

### Penanganan Outlier:
Distribusi jumlah seberapa sering pengguna melakukan rating menunjukkan adanya **user outlier**, yaitu pengguna yang lebih sering memberikan rating dibandingkan mayoritas pengguna lain. Keberadaan outlier ini dapat mendominasi proses pelatihan, terutama pada pendekatan collaborative filtering yang sangat bergantung pada pola interaksi. Langkah yang dilakukan menghapus seluruh interaksi dari user outlier. 

| Keterangan                          | Jumlah     |
| ----------------------------------- | ---------- |
| Jumlah data awal                    | 15.823.269 |
| Jumlah data setelah outlier dihapus | 6.149.427  |
| Jumlah user setelah pembersihan     | 312.834    |

Sebagian besar user kini memiliki jumlah rating yang lebih merata. Penghapusan data outlier ini bertujuan untuk menjaga **keseimbangan kontribusi antar pengguna** dalam proses pelatihan dan menghindari overfitting terhadap preferensi minoritas ekstrem.

‎ 
<div align="center">
  <img src="https://github.com/user-attachments/assets/d25f929b-dbef-457d-b0c3-b7f448850674" width="400"/>
</div>
‎ 

### Feature Selection:
Langkah selanjutnya adalah **memilih kolom-kolom penting** dari masing-masing dataset sebelum digabungkan untuk mengseleksi data yang hanya ada untuk kedua dataset. Pada tahap ini juga dilakukan penanganan missing value dengan menghapus data yang memiliki missing value tersebut:

* Dari `bgg-15m-reviews.csv` (data interaksi pengguna), dipilih:

  * `user` menjadi `username` (id unik user), terdapat **66** missing value yang diatasi.
  * `rating`.
  * `ID` menjadi `gameId` (id unik game).

* Dari `games_detailed_info.csv` (metadata game), dipilih:

  * `id` menjadi `gameId` (id unik game).
  * `primary` menjadi `gameName`.
  * `boardgamecategory` menjadi `category` (fitur utama yang akan digunakan) terdapat **283** missing value yang diatasi.

Tujuan dari feature selection ini adalah untuk menyederhanakan proses analisis dan memastikan hanya fitur yang relevan yang digunakan pada tahap modeling.

### Data Filtering:
Setelah kolom-kolom penting dipilih melalui proses feature selection, langkah selanjutnya adalah menggabungkan kedua dataset — yaitu data ulasan pengguna dan metadata board game — menggunakan kolom **`gameId`** sebagai kunci utama dari kedua sumber.

Penggabungan dilakukan menggunakan metode **`inner join`** (`how='inner'`) untuk memastikan bahwa hanya data yang memiliki pasangan di kedua tabel yang akan disertakan dalam dataset akhir. Dengan cara ini, setiap interaksi pengguna dipastikan memiliki informasi konten game yang lengkap dan valid.

Hasil penggabungan kemudian dibagi kembali menjadi dua subset sesuai kebutuhan model:

* Subset untuk pendekatan **content-based filtering** (menggunakan `gameId` dan `category`).
* Subset untuk pendekatan **collaborative filtering** (menggunakan `user`, `gameId`, dan `rating`).

Data yang dihasilkan dari gabungan ini telah **bebas dari missing value** dan hanya terdiri dari entri yang **memiliki pasangan valid di kedua dataset**. Dataset inilah yang kemudian digunakan dalam proses pelatihan model rekomendasi.


### TF-IDF Extraction (Content-Based Filtering):
Pada pendekatan *content-based filtering*, sistem rekomendasi dibuat berdasarkan kemiripan konten antar board game. Fitur utama yang mencerminkan karakteristik konten dalam dataset ini adalah kolom `category`, yang berisi daftar kategori atau genre dari setiap game.
* Fitur `boardgamecategory` awalnya berbentuk list (daftar string).
* Daftar tersebut dikonversi menjadi **teks datar** (misalnya: `"Card Game Strategy Fantasy"`).
* Seluruh teks diproses menggunakan `TfidfVectorizer` dari Scikit-learn untuk menghasilkan representasi vektor sparse dari setiap game.
* Proses ini menghasilkan **matriks TF-IDF** berukuran *game × term*, di mana setiap nilai merepresentasikan pentingnya suatu kategori terhadap game tersebut dibandingkan keseluruhan koleksi game.

Representasi vektor dari TF-IDF memungkinkan penggunaan metrik seperti **cosine similarity** untuk mengukur kemiripan antar game, sehingga sistem dapat merekomendasikan game yang secara tematis serupa dengan game sebelumnya.

### Encoded Data (Collaborative Filtering):

Pada pendekatan *collaborative filtering*, sistem rekomendasi mempelajari hubungan antara pengguna dan game berdasarkan riwayat interaksi (rating). Untuk membangun model menggunakan arsitektur neural network dengan embedding layer, diperlukan representasi input dalam bentuk numerik.

* Kolom `user` dan `gameId` pada data memiliki tipe data kategorikal (string dan integer non-urut).
* Keduanya dikonversi menjadi **nilai integer urut (label encoding)** menggunakan `LabelEncoder` dari Scikit-learn.
* Hasil encoding disimpan sebagai:

  * `user`: representasi numerik dari pengguna.
  * `game`: representasi numerik dari game.

* Contoh:

  * `username = "TomVasel"` → `user = 281`
  * `gameId = 30549` → `game = 1573`

Neural network embedding layer hanya menerima input berupa indeks integer, bukan string atau nilai acak. Label encoding memungkinkan setiap user dan game dipetakan ke dalam **vektor embedding berdimensi rendah**, yang akan dipelajari selama pelatihan. Pendekatan ini memungkinkan model mengenali **pola laten** dalam preferensi pengguna dan karakteristik game tanpa menggunakan fitur konten eksplisit.

### Label Normalization (Collaborative Filtering):

Model collaborative filtering dibangun menggunakan arsitektur neural network yang memprediksi **nilai rating** sebagai output. Model ini menggunakan fungsi aktivasi **sigmoid** di lapisan output, yang secara alami membatasi nilai keluaran pada rentang **\[0, 1]**.

* Nilai rating asli dari pengguna berada dalam rentang **1 hingga 10**.
* Untuk menyesuaikan dengan rentang output sigmoid, nilai rating dinormalisasi menggunakan metode **min-max scaling**.
* Transformasi dilakukan menggunakan rumus:

$$
r_{\text{scaled}} = \frac{r - 1}{10 - 1}
$$

Fungsi aktivasi sigmoid hanya dapat memetakan nilai ke rentang \[0, 1], sehingga nilai rating perlu dinormalisasi ke skala yang sama. Tanpa scaling, model tidak dapat belajar dengan benar karena prediksi akan selalu dibatasi oleh output sigmoid. Scaling juga mempercepat konvergensi saat training dan membantu menstabilkan proses pembelajaran.

### Data Splitting (Collaborative Filtering):
Setelah semua tahap pembersihan dan transformasi data selesai dilakukan, dataset disiapkan untuk pelatihan model dengan cara membaginya ke dalam dua subset:

* **Training set (80%)**: digunakan untuk melatih model, yaitu mempelajari pola hubungan antara pengguna dan game dari data historis.
* **Validation set (20%)**: digunakan untuk mengevaluasi kinerja model terhadap data yang belum pernah dilihat sebelumnya, guna mengukur generalisasi model.

Pembagian dilakukan secara acak menggunakan fungsi `train_test_split` dari Scikit-learn. Parameter `random_state` digunakan untuk memastikan hasil split **konsisten dan reprodusibel** setiap kali kode dijalankan. Fitur (`user_index`, `game_index`) dan target (`rating_scaled`) dipisahkan sebelum pemisahan data dilakukan.

## Modeling and Result

Tahap ini bertujuan untuk membangun dan mengevaluasi sistem rekomendasi board game yang mampu membantu pengguna menemukan permainan yang sesuai dengan preferensi mereka. Dua pendekatan digunakan dalam proyek ini: **content-based filtering** dan **collaborative filtering**, yang masing-masing menyelesaikan permasalahan dari sudut yang berbeda.

### 1. Content-Based Filtering

#### **Model dan Cara Kerja**

Content-based filtering bekerja dengan **menganalisis fitur deskriptif dari item**, dalam hal ini kategori board game, untuk menemukan kesamaan antar game. Prosesnya sebagai berikut:

1. Kolom `category` dikonversi dari list ke format string.
2. Teks kategori diubah menjadi representasi numerik menggunakan **TF-IDF Vectorizer**, menghasilkan matriks sparse yang menggambarkan pentingnya setiap istilah dalam konteks game.
3. Kemiripan antar game dihitung menggunakan **cosine similarity**, yaitu metrik yang mengukur sudut antar vektor di ruang fitur.
4. Untuk setiap input game, sistem mengambil Top-N game dengan skor kemiripan tertinggi.

#### **Top-5 Recommendation**

Input: *Hannibal & Hamilcar*  
Kategori: `['Ancient', 'Political', 'Wargame']`

Output:

| No | Game yang Direkomendasikan             | Kategori                             |
| -- | -------------------------------------- | ------------------------------------ |
| 1  | Pericles: The Peloponnesian Wars       | \['Ancient', 'Political', 'Wargame'] |
| 2  | Hannibal: Rome vs. Carthage            | \['Ancient', 'Political', 'Wargame'] |
| 3  | Imperium Romanum: The Clash of Legions | \['Ancient', 'Political', 'Wargame'] |
| 4  | Nero                                   | \['Ancient', 'Political', 'Wargame'] |
| 5  | Pax Romana                             | \['Ancient', 'Political', 'Wargame'] |

Rekomendasi ini sangat relevan secara tematis, menunjukkan efektivitas metode dalam memahami konten.

#### **Kelebihan**

* **Independen dari pengguna lain**: sangat efektif untuk pengguna baru (*cold-start*).
* **Mudah dijelaskan**: alasan kemiripan bisa dilacak melalui fitur seperti kategori.

#### **Kekurangan**

* **Terbatas pada informasi konten**: tidak memperhitungkan ulasan atau rating.
* **Tidak menangkap preferensi pribadi pengguna** yang tidak tercermin dalam konten.

### 2. Collaborative Filtering

#### **Model dan Cara Kerja**

Pendekatan ini menggunakan data interaksi (rating) antar pengguna dan game untuk mempelajari **relasi laten** di antara keduanya, tanpa memerlukan informasi konten.

1. Kolom `user` dan `gameId` diencoding menjadi indeks integer.
2. Setiap entitas (user dan game) direpresentasikan sebagai **vektor embedding berdimensi rendah** (misalnya dimensi 50).
3. Embedding user dan game digabung (concatenate), kemudian diproses melalui jaringan fully connected (dense layer).
4. Output model berupa nilai rating terprediksi dalam rentang \[0, 1] (setelah sigmoid).
5. Fungsi loss yang digunakan adalah **Mean Squared Error (MSE)** terhadap rating yang telah dinormalisasi.

####  **Spesifikasi Arsitektur**

| **Komponen**          | **Spesifikasi**                     | **Alasan Pemilihan**                                                                                                                     |
| --------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Input Layer**       | `user`, `game` (sebagai indeks)     | Kedua entitas direpresentasikan sebagai indeks numerik agar dapat dipetakan ke dalam vektor embedding.                                   |
| **Embedding Dimensi** | 50                                  | Dimensi 50 cukup untuk menangkap pola laten tanpa overfitting; lebih besar dari 20 (standar awal), namun masih efisien secara komputasi. |
| **Interaksi Layer**   | Dot product antara user-game vector | Pendekatan GMF (Generalized Matrix Factorization) yang sederhana namun efektif untuk sistem rekomendasi berbasis interaksi.              |
| **Bias Layer**        | Embedding bias user dan game        | Membantu model mengakomodasi perbedaan global antar user dan antar game (misalnya user yang selalu memberi rating tinggi/rendah).        |
| **Output Layer**      | 1 neuron, aktivasi sigmoid          | Karena target rating telah dinormalisasi ke rentang \[0, 1], sigmoid cocok digunakan sebagai aktivasi output regresi probabilistik.      |
| **Loss Function**     | Mean Squared Error (MSE)            | Metrik klasik untuk regresi, efektif untuk mengukur jarak antara rating prediksi dan aktual dalam skenario skala kontinu.                |
| **Optimizer**         | Adam (learning rate 0.001)          | Optimizer adaptif yang cepat dan stabil untuk model deep learning tanpa banyak tuning manual.                                            |
| **Epoch**             | 50                                  | Jumlah epoch cukup untuk mencapai konvergensi pada dataset besar tanpa overfitting. Telah diuji dengan validasi dan hasilnya stabil.     |
| **Batch Size**        | 512                                 | Ukuran batch ini cocok untuk menyeimbangkan kecepatan pelatihan dan efisiensi memori, terutama saat menggunakan hardware GPU/TPU.        |

#### **Top-10 Recommendation**

User: *josephcasey*   

**Game dengan rating tinggi dari user:**

| Game                                 | Category                                                                                  |
|------------------------------------|-------------------------------------------------------------------------------------------|
| The Resistance: Avalon              | Bluffing, Card Game, Deduction, Fantasy, Medieval, Negotiation, Party Game, Spies/Secret Agents |
| Star Wars: Imperial Assault         | Adventure, Exploration, Fighting, Miniatures, Movies / TV / Radio theme, Science Fiction, Wargame |
| Architects of the West Kingdom      | City Building, Medieval                                                                   |
| The Lord of the Rings: Journeys in Middle-Earth | Adventure, Fantasy, Fighting, Miniatures, Novel-based                          |
| Heroes of Land, Air & Sea            | Exploration, Fantasy, Fighting, Miniatures, Wargame                                       |

**Game yang direkomendasikan:**

| Game                             | Category                                                                                             |
|---------------------------------|----------------------------------------------------------------------------------------------------|
| Terraforming Mars                | Economic, Environmental, Industry / Manufacturing, Science Fiction, Space Exploration, Territory Building |
| Gloomhaven                     | Adventure, Exploration, Fantasy, Fighting, Miniatures                                              |
| Pandemic Legacy: Season 1       | Environmental, Medical                                                                             |
| Terra Mystica                  | Civilization, Economic, Fantasy, Territory Building                                                |
| Arkham Horror: The Card Game    | Adventure, Card Game, Collectible Components, Fantasy, Horror, Novel-based                         |
| Spirit Island                  | Age of Reason, Environmental, Fantasy, Fighting, Mythology, Renaissance, Territory Building       |
| Caverna: The Cave Farmers       | Animals, Economic, Fantasy, Farming                                                               |
| Star Wars: Rebellion            | Civil War, Fighting, Miniatures, Movies / TV / Radio theme, Science Fiction, Wargame               |
| Brass: Birmingham               | Economic, Industry / Manufacturing, Post-Napoleonic, Transportation                               |
| Gaia Project                   | Economic, Science Fiction, Space Exploration, Territory Building                                   |


Model mampu menyarankan game **yang belum pernah dimainkan**, namun relevan dengan pola rating historis pengguna.

#### **Kelebihan**

* Menangkap **preferensi pengguna secara laten** tanpa melihat isi game.
* Mampu merekomendasikan game yang secara konten tidak mirip, tetapi disukai oleh pengguna serupa.

#### **Kekurangan**

* **Cold-start problem**: performa menurun untuk pengguna atau game baru tanpa riwayat.
* Butuh dataset besar dan distribusi interaksi yang cukup merata untuk hasil yang optimal.

### Kelebihan dan Kekurangan Setiap Model:
| Model                  | Kelebihan                                                                                                                                           | Kekurangan                                                                                                           |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Content-Based Filtering | - Independen dari pengguna lain (baik untuk *cold-start*)<br>- Mudah dijelaskan lewat fitur konten<br>- Tidak butuh data dari pengguna lain<br>- Konsisten dalam rekomendasi jika fitur konten tetap | - Terbatas pada informasi konten<br>- Tidak menangkap preferensi pengguna tersembunyi<br>- Kurang inovatif (cenderung menyarankan item serupa) |
| Collaborative Filtering | - Menangkap preferensi pengguna secara laten<br>- Bisa merekomendasikan game yang tidak mirip kontennya<br>- Menyesuaikan dengan tren komunitas<br>- Cocok untuk dataset besar dengan banyak interaksi | - Mengalami *cold-start problem* untuk pengguna atau item baru<br>- Butuh dataset besar dan interaksi merata<br>- Sulit dijelaskan alasan rekomendasinya |

## Evaluation
Pada tahap ini, dilakukan evaluasi terhadap sistem rekomendasi yang telah dibangun menggunakan dua pendekatan berbeda, yaitu **content-based filtering** dan **collaborative filtering**. Masing-masing pendekatan dievaluasi dengan metrik yang sesuai:

* Untuk **content-based filtering**, digunakan metrik relevansi untuk mengukur kualitas top-N rekomendasi.
* Untuk **collaborative filtering**, digunakan metrik regresi karena model memprediksi nilai rating.

### Metrik Evaluasi (Content-Based Filtering):

**Precision:**

![Precision](https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}\textbf{$$\text{Precision}=\frac{|\text{Relevant&space;Items}\cap\text{Recommended&space;Items}|}{|\text{Recommended&space;Items}|}$$})

**Recall:**

![Recall](https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}$$\text{Recall}=\frac{|\text{Relevant&space;Items}\cap\text{Recommended&space;Items}|}{|\text{Relevant&space;Items}|}$$)

**F1-Score:**

![F1-Score](https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}$$\text{F1}=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}&plus;\text{Recall}}$$)

**MAP\@K (Mean Average Precision at K):**

![MAP\@K (Mean Average Precision at K)](https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}$$\text{MAP@K}=\frac{1}{|U|}\sum_{u\in&space;U}\frac{1}{\min(K,|R_u|)}\sum_{k=1}^{K}P_u(k)\cdot\text{rel}_u(k)$$)

**Intra-List Similarity:**

![Intra-List Similarity](https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}$$\text{ILS}(R)=\frac{2}{|R|(|R|-1)}\sum_{i=1}^{|R|}\sum_{j=i&plus;1}^{|R|}\text{sim}(i,j)$$)

**Coverage:**

![Coverage](https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}$$\text{Coverage}=\frac{|\cup_{u\in&space;U}R_u|}{|I|}$$)

| **Metrik**                | **Penjelasan**                                                                                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Precision**             | Proporsi rekomendasi yang relevan dari semua item yang direkomendasikan.                                                                                   |
| **Recall**                | Proporsi item relevan yang berhasil direkomendasikan dari semua item relevan yang tersedia.                                                                |
| **F1-Score**              | Harmonik rata-rata precision dan recall. Berguna saat ingin menyeimbangkan keduanya.                                                                       |
| **MAP\@K**                | Rata-rata presisi kumulatif atas Top-K rekomendasi. Semakin tinggi nilainya, semakin baik rekomendasi teratas.                                             |
| **Intra-list Similarity** | Mengukur kesamaan antar item dalam satu daftar rekomendasi. Nilai tinggi berarti rekomendasi lebih homogen.                                                |
| **Coverage**              | Proporsi item yang direkomendasikan dibandingkan keseluruhan item yang tersedia. Nilai rendah bisa berarti sistem hanya mengekspos game yang itu-itu saja. |

### Hasil Evaluasi (Content-Based Filtering):
Berikut adalah hasil evaluasi model berdasarkan metrik Precission, Recall, F1-Score, Map\@K, Intra-list Similarity dan Coverage:

| **Metrik**                | **Nilai** |
| ------------------------- | --------- |
| **Precision**             | 0.620     |
| **Recall**                | 0.313     |
| **F1-Score**              | 0.261     |
| **MAP\@K**                | 0.746     |
| **Intra-list Similarity** | 0.934     |
| **Coverage**              | 0.034     |
| **Samples Evaluated**     | 159       |

**Precision** sebesar `0.620` menunjukkan bahwa lebih dari separuh item yang direkomendasikan memang relevan — ini merupakan indikator bahwa sistem memberikan rekomendasi yang cukup akurat.

**Recall** `0.313` masih tergolong rendah, artinya banyak item relevan yang belum berhasil direkomendasikan. Hal ini umum pada content-based karena cakupan sistem terbatas pada konten yang mirip.

**F1-Score** `0.261` mengonfirmasi ketidakseimbangan antara precision dan recall.

**MAP\@K** yang tinggi (`0.746`) mengindikasikan bahwa item yang relevan cenderung muncul di posisi atas dalam daftar rekomendasi, yang sangat baik untuk pengalaman pengguna.

**Intra-list Similarity** `0.934` menunjukkan bahwa item-item dalam satu daftar sangat mirip, yang membuat rekomendasi sangat konsisten, meskipun dapat mengurangi keragaman.

**Coverage** sangat rendah (`0.034`), artinya sistem hanya menjelajahi sebagian kecil dari keseluruhan game. Ini adalah kelemahan umum pada content-based filtering karena model hanya bisa merekomendasikan game yang mirip dengan apa yang sudah dikenal pengguna.

### Metrik Evaluasi (Collaborative Filtering):

**Mean Squared Error (MSE):**

![MSE](https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}$$\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$$)

**Mean Absolute Error (MAE):**

![MAE](https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}$$\text{MAE}=\frac{1}{n}\sum_{i=1}^{n}\left|y_i-\hat{y}_i\right|$$)

**Root Mean Squared Error (RMSE):**

![RMSE](https://latex.codecogs.com/png.image?\huge&space;\dpi{110}\bg{white}$$\text{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}$$)

| **Metrik** | **Penjelasan**                                                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **MSE**    | Mengukur rata-rata kuadrat selisih antara nilai aktual dan prediksi. Nilai lebih kecil menandakan model lebih akurat.                |
| **MAE**    | Rata-rata selisih absolut antara prediksi dan nilai aktual. Lebih mudah diinterpretasikan karena satuannya sama dengan rating.       |
| **RMSE**   | Akar dari MSE, memberi penalti lebih besar pada kesalahan besar. Sering digunakan sebagai metrik utama untuk evaluasi model regresi. |

### Hasil Evaluasi (Content-Based Filtering):
Berikut adalah hasil train model berdasarkan metrik MSE, MAE dan RMSE:

‎ 
<div align="center">
  <img src="https://github.com/user-attachments/assets/70d694cf-75ab-4bc0-84c5-19ae975a0908" width="400"/>
</div>
‎ 

Pada awal pelatihan, terjadi lonjakan loss hingga mencapai puncaknya sekitar epoch ke-2 (train loss \~0.178, val loss \~0.171), yang merupakan dampak wajar dari inisialisasi parameter model. Setelah itu, loss menurun secara progresif dan mencapai nilai konvergen setelah sekitar epoch ke-20.

Di akhir pelatihan:

* **Train Loss**: \~0.102
* **Validation Loss**: \~0.106

Perbedaan hanya sekitar **0.004 poin**, menunjukkan bahwa model **tidak mengalami overfitting**, karena performa di validation set hampir setara dengan training set.

Tren MSE menurun signifikan setelah fluktuasi awal, menunjukkan bahwa model mampu meminimalkan error besar secara bertahap.

Nilai akhir:

* **Train MSE**: \~0.061
* **Validation MSE**: \~0.068

Perbedaan kecil (\~0.007) menunjukkan bahwa model tidak terlalu “menghapal” data latih dan dapat beradaptasi dengan baik pada data baru.

MAE juga menunjukkan tren penurunan yang stabil. Setelah sempat fluktuatif di awal (train MAE sempat menyentuh \~0.243), nilai akhirnya stabil.

Nilai akhir:

* **Train MAE**: \~0.197
* **Validation MAE**: \~0.219

Dengan perbedaan sekitar 0.022, ini menunjukkan bahwa model memiliki **kemampuan generalisasi yang baik namun tidak agresif mengorbankan error rata-rata demi error besar**.

Dari grafik, RMSE mencapai puncak di awal pelatihan (epoch 2-3) lalu stabil menurun. Setelah epoch ke-20, RMSE menjadi relatif konstan.

Nilai akhir:

* **Train RMSE**: \~0.248
* **Validation RMSE**: \~0.260

Kesenjangan RMSE yang kecil mengindikasikan **model tidak high variance**, dan kesalahan prediksi masih dalam batas yang sangat wajar.

---

### Apakah model menjawab *problem statement*?

Ya, model yang dikembangkan dalam proyek ini berhasil menjawab kedua *problem statement* secara efektif. Pada sisi **content-based filtering**, model mampu merekomendasikan board game berdasarkan kemiripan konten, khususnya kategori permainan. Misalnya, pengguna yang menyukai game bertema “Wargame” atau “Ancient” akan mendapatkan rekomendasi dari game lain dengan karakteristik konten serupa. Rekomendasi yang dihasilkan konsisten secara tematis dan relevan, seperti terlihat pada daftar rekomendasi untuk game *Hannibal & Hamilcar* yang seluruhnya berbagi kategori utama yang sama.

Sementara itu, pada pendekatan **collaborative filtering**, model memanfaatkan data rating dari pengguna untuk mempelajari pola preferensi dan memberikan rekomendasi yang dipersonalisasi. Hal ini menjawab problem statement kedua tentang bagaimana memberikan rekomendasi yang sesuai dengan riwayat interaksi pengguna. Model berhasil merekomendasikan game baru yang belum pernah dimainkan pengguna tetapi relevan secara preferensial, seperti ditunjukkan pada rekomendasi untuk pengguna *josephcasey* yang cenderung menyukai game strategis dengan elemen petualangan dan miniatur.

---

### Apakah model berhasil mencapai *goals*?

Secara keseluruhan, kedua *goals* proyek berhasil dicapai dengan hasil yang memuaskan:

* **Goal 1: Membangun sistem rekomendasi berdasarkan konten** berhasil direalisasikan melalui pendekatan content-based filtering berbasis TF-IDF dan cosine similarity. Evaluasi metrik seperti Precision (0.620) dan MAP\@K (0.746) membuktikan bahwa sistem dapat memberikan rekomendasi yang relevan dan bernilai tinggi di posisi atas.

* **Goal 2: Membangun sistem rekomendasi berbasis riwayat pengguna** tercapai dengan penerapan model embedding neural network. Evaluasi menggunakan MSE, MAE, dan RMSE menunjukkan performa yang baik dan stabil:

  * MSE: \~0.068 (val)
  * MAE: \~0.219 (val)
  * RMSE: \~0.260 (val)

Kedua sistem menunjukkan hasil yang mendekati ground truth dan mampu melakukan generalisasi tanpa overfitting, membuktikan keandalan pendekatan dan efektivitas proses pelatihan.

---

### Apakah solusi statement berdampak?

Solusi yang dibangun memiliki dampak yang nyata terhadap proses rekomendasi dan pengalaman pengguna. Dengan menggabungkan kekuatan content-based dan collaborative filtering, sistem yang dihasilkan mampu:

* Menyediakan rekomendasi personal yang akurat meskipun dengan data terbatas pada pengguna baru (melalui pendekatan konten),
* Menangkap preferensi tersembunyi dari pengguna aktif (melalui pendekatan kolaboratif),
* Memungkinkan eksplorasi game yang kurang populer namun berkualitas, sehingga mendukung ekosistem board game yang lebih inklusif.

Analisis seperti Intra-List Similarity (0.934) menunjukkan bahwa rekomendasi content-based sangat tematik dan konsisten, sementara collaborative filtering menyeimbangkan relevansi personal dengan keberagaman. Meski **coverage** pada pendekatan berbasis konten masih rendah (0.034), sistem tetap menunjukkan potensi besar untuk dikembangkan lebih lanjut.

---

### Kesimpulan

Proyek ini telah berhasil membangun sistem rekomendasi board game yang efektif dan dapat diandalkan, baik dalam memahami konten game maupun preferensi pengguna. Dengan menggabungkan pendekatan content-based dan collaborative filtering, sistem mampu mengatasi berbagai tantangan, mulai dari *cold-start* hingga personalisasi mendalam. Evaluasi metrik menunjukkan performa model yang tinggi dan stabil, serta menunjukkan bahwa rekomendasi yang dihasilkan tidak hanya relevan secara teknis, tetapi juga bermakna secara praktis.

Dengan demikian, sistem ini berpotensi memberikan manfaat besar bagi pengguna yang ingin menemukan game sesuai preferensi mereka, sekaligus membantu pengembang game menjangkau audiens yang lebih luas melalui pendekatan berbasis data. Proyek ini menunjukkan bahwa machine learning dapat digunakan secara nyata untuk **meningkatkan pengalaman pengguna dalam domain hiburan**, khususnya dunia board game.