# Narasi Presentasi — CASA (Climate-Adaptive Spatial Attention)
**Seminar Doctoral · Ehime University · April 2026**
**Presenter: Arif Nurwahid**

---

## Slide 1: Judul

> Selamat siang, terima kasih sudah hadir hari ini.
>
> Nama saya Arif Nurwahid, di bawah bimbingan Professor Matsuura. Hari ini saya akan mempresentasikan progress saya tentang modul CASA — Climate-Adaptive Spatial Attention — sebuah komponen baru untuk peramalan curah hujan spatiotemporal di seluruh provinsi Indonesia.
>
> Ide utamanya: daripada menggunakan bobot spasial yang sama selama 20 tahun, CASA membuat bobot tersebut berubah setiap bulan, menyesuaikan kondisi iklim seperti ENSO dan IOD.

---

## Slide 2: Section — Why Static Spatial Weights Fail

*(Halaman pemisah bagian — transisi langsung.)*

---

## Slide 3: One Assumption That Does Not Hold Physically

> Saya mulai dari masalahnya. Di model LSTM-GSTARX kita saat ini, spatial lag dihitung menggunakan matriks bobot statis — W-static — yang dibuat sekali dari jarak Haversine antar pusat provinsi, lalu tidak berubah selama periode 2005 sampai 2024.
>
> Ini mengasumsikan hubungan spasial antar provinsi tidak pernah berubah. Tapi kenyataannya berubah, karena tiga alasan.
>
> Pertama, ENSO. Saat El Niño, Sirkulasi Walker melemah. Provinsi yang biasanya punya pola curah hujan mirip bisa tiba-tiba berperilaku sebaliknya.
>
> Kedua, Indian Ocean Dipole. IOD positif mengeringkan Sumatera dan Jawa, tapi Papua di sisi timur tidak terpengaruh. Jadi provinsi yang biasanya bergerak bersama tiba-tiba terputus.
>
> Ketiga, transisi monsun. Arah aliran kelembapan utama berbalik antar musim, yang mengubah provinsi mana mempengaruhi provinsi mana.
>
> Yang saya usulkan: mengganti W statis dengan W-sub-t yang dinamis — campuran bobot geografi dan bobot yang dipelajari, dan campuran ini menyesuaikan setiap bulan.

---

## Slide 4: Section — Background

*(Halaman pemisah bagian.)*

---

## Slide 5: Base Model — From GSTAR to LSTM-GSTARX

> Sebelum saya jelaskan CASA, saya perlihatkan dulu model dasar tempat CASA berada.
>
> Kita mulai dari GSTAR — model linear yang menggunakan bobot geografi untuk menangkap hubungan spasial. Lalu kita tambahkan indeks iklim sebagai input tambahan, jadi GSTARX. Lalu kita ganti bagian linear dengan LSTM untuk menangkap pola nonlinear — jadilah LSTM-GSTARX.
>
> Semua model ini menggunakan matriks bobot statis yang sama. CASA adalah langkah keempat: mengganti matriks statis itu dengan yang dinamis. Bagian lainnya — LSTM, cara training — semuanya tetap sama persis. CASA hanya mengubah cara bobot spasial dihitung.

---

## Slide 6: What is a Graph Neural Network?

> Sekarang sedikit latar belakang tentang graph neural network, karena dari sinilah ide attention berasal.
>
> Graph itu sederhana: node yang dihubungkan oleh edge. Dalam kasus kita, 38 provinsi adalah node, dan edge menggambarkan kedekatan lokasi.
>
> GNN memperbarui setiap node dengan mengumpulkan informasi dari tetangganya. Rumusnya sederhana: representasi baru node i adalah hasil agregasi dari representasi tetangga-tetangganya.
>
> Graph Convolutional Network — GCN — dari Kipf dan Welling menggunakan bobot tetap berdasarkan degree (jumlah koneksi) node. Tapi artinya semua tetangga diperlakukan sama. Untuk curah hujan, ini bermasalah: saat El Niño, provinsi yang jauh tapi terhubung lewat telekoneksi atmosfer bisa lebih penting dari yang dekat. Kita butuh bobot yang bisa belajar tetangga mana yang benar-benar berpengaruh.

---

## Slide 7: GAT and GATv2

> GAT — Graph Attention Network — menyelesaikan masalah ini dengan menambahkan bobot attention yang bisa dipelajari. Model belajar seberapa penting tiap tetangga, menggunakan skor attention yang melihat fitur kedua node.
>
> Skor ini dihitung dengan menggabungkan fitur node i dan j — tanda garis vertikal di sini artinya concatenation, yaitu menyambung dua vektor jadi satu — lalu dilewatkan ke neural network kecil dengan LeakyReLU.
>
> Tapi ada masalah tersembunyi yang ditemukan Brody dan rekan-rekannya tahun 2022. Urutan pentingnya tetangga di GAT ternyata tidak berubah tergantung siapa yang bertanya. Saya akan buktikan ini di slide selanjutnya.
>
> Solusi mereka, GATv2, menerapkan nonlinearitas setelah concatenation, bukan sebelumnya. Dengan begitu model bisa menangkap interaksi nyata antara kedua node.
>
> CASA dibangun di atas GATv2 dengan tiga perubahan: kita melihat semua pasangan provinsi bukan hanya tetangga, kita memasukkan informasi iklim, dan kita mencampur hasilnya dengan bobot geografi.

---

## Slide 8: Related Work

> Sebelum membahas CASA, saya tempatkan dulu konteksnya dengan metode lain yang juga mencoba mempelajari struktur spasial.
>
> Graph WaveNet dari Wu dkk. mempelajari matriks adjacency adaptif melalui node embedding. Ide bagus, tapi matriksnya dihitung sekali dan tidak berubah — tidak menyesuaikan setiap bulan. Dan tidak ada informasi geografi di dalamnya.
>
> AGCRN dari Bai dkk. lebih fleksibel — membangun struktur graf dari data. Tapi lagi-lagi, grafnya tetap setelah training. Tidak menyesuaikan kondisi iklim yang berbeda.
>
> WST-ANet menggunakan dekomposisi wavelet dengan spatial attention. Ada sedikit conditioning terhadap iklim, tapi tidak ada basis geografi dan tidak ada mekanisme untuk mencampur antara geografi dan bobot yang dipelajari.
>
> Tabel perbandingan nanti akan menunjukkan bahwa CASA adalah yang pertama menggabungkan semua ini: bobot yang berubah tiap waktu, basis geografi, conditioning iklim, dan blending gate yang bisa dipelajari.

---

## Slide 9: Section — CASA Architecture

*(Halaman pemisah bagian.)*

---

## Slide 10: Three Stages Overview

> Ini arsitektur CASA. Tiga tahap, semuanya berjalan di setiap time step.
>
> Kita ambil curah hujan bulan sebelumnya ditambah indeks iklim. Stage 1 memampatkan keadaan iklim ke dalam vektor konteks 16 dimensi. Ini masuk ke Stage 2a yang menghitung skor attention untuk semua pasangan provinsi, dan Stage 2b yang menghitung blending gate — satu angka yang memutuskan seberapa besar kita percaya geografi dibanding pola yang dipelajari.
>
> Stage 3 mencampur bobot geografi dan bobot yang dipelajari, lalu mengalikannya dengan vektor curah hujan untuk mendapat spatial lag. Ini masuk ke LSTM, yang tidak diubah sama sekali.

---

## Slide 11: Stage 1 — Climate Context Vector (Rumus)

> Stage 1 menghitung vektor konteks iklim c-sub-t. Kita ambil empat input: rata-rata curah hujan, standar deviasi spasial, Niño 3.4, dan DMI. Kalikan dengan matriks yang bisa dipelajari, tambah bias, dan terapkan tanh.
>
> Rata-rata curah hujan memberi tahu kita: apakah Indonesia secara keseluruhan basah atau kering? Standar deviasi memberi tahu: apakah curah hujannya merata atau terpusat di beberapa daerah? Dan dua indeks iklim langsung menggambarkan kondisi Samudera Pasifik dan Hindia.

---

## Slide 12: What Does c_t Represent?

> Apa sebenarnya arti vektor ini? Bayangkan sebagai sidik jari iklim untuk bulan ini. Isinya 16 angka, dan masing-masing belajar merespons kombinasi tertentu dari empat input tadi.
>
> Kenapa pakai tanh? Supaya semua nilainya tetap antara minus satu dan plus satu, jadi tidak ada satu kejadian iklim ekstrem yang mendominasi. Dan kenapa hanya 4 input? Kalau pakai semua 38 provinsi, parameternya terlalu banyak. Statistik ringkasan sudah cukup menangkap pola keseluruhan. Detail per provinsi masuk nanti di Stage 2.
>
> Satu hal penting: vektor konteks ini dipakai dua kali — di skor attention dan di blending gate. Jadi vektor ini menentukan baik provinsi mana yang saling mempengaruhi, maupun seberapa besar kita percaya hasilnya.

---

## Slide 13: Activation Functions

> Sebelum Stage 2, saya review singkat tiga fungsi yang kita pakai.
>
> Softmax mengambil skor mentah dan mengubahnya jadi pecahan yang jumlahnya tepat 1. Skor lebih tinggi berarti bagian lebih besar. Kita pakai ini untuk mengubah skor attention jadi bobot yang valid.
>
> Sigmoid memampatkan angka apapun ke rentang antara 0 dan 1. Kita pakai ini untuk blending gate — alpha-t.
>
> LeakyReLU melewatkan nilai positif langsung, dan membiarkan nilai negatif lewat dengan kemiringan kecil 0.2. Ini menghindari masalah "neuron mati" di mana neuron berhenti belajar sama sekali. Kita pakai ini dalam perhitungan attention, mengikuti standar dari paper GAT.

---

## Slide 14: Stage 2a — Attention Scores

> Stage 2a menghitung seberapa besar setiap provinsi harus mempengaruhi provinsi lainnya bulan ini.
>
> Untuk setiap pasangan provinsi i dan j, kita buat vektor input: curah hujan kedua provinsi, selisih di antara keduanya, dan 16 angka konteks iklim. Total 19 input. Ini melewati matriks bobot, LeakyReLU, dan proyeksi untuk menghasilkan satu skor.
>
> Kita lakukan ini untuk semua 38 kuadrat — 1.444 — pasangan. Lalu softmax per baris menormalisasi skornya supaya setiap baris jumlahnya 1. Hasilnya adalah matriks attention yang dipelajari.

---

## Slide 15: Stages 2b & 3 — Gate dan Final Weights

> Stage 2b sederhana tapi penting. Menghitung alpha-t — satu angka antara 0 dan 1 — pakai sigmoid yang diterapkan pada kombinasi linear dari vektor konteks.
>
> Kalau alpha-t dekat 1, geografi yang mendominasi. Kalau dekat 0, bobot yang dipelajari yang mendominasi. Tabel menunjukkan apa yang kita harapkan untuk kondisi iklim berbeda. Saat bulan normal, kita harap alpha-t dekat 1. Saat El Niño kuat, di bawah 0.3.
>
> Saya tekankan: ini nilai yang diharapkan berdasarkan desain, bukan hasil pengukuran. Kita akan verifikasi di eksperimen nanti.
>
> Stage 3 mencampur: W-sub-t adalah alpha-t kali matriks geografi ditambah satu-minus-alpha-t kali matriks yang dipelajari. Lalu kita kalikan dengan vektor curah hujan.

---

## Slide 16: Forward Pass Algorithm

> Slide ini menunjukkan seluruh forward pass sebagai pseudocode. Merangkum semua yang sudah kita bahas dalam satu algoritma. Yang penting diperhatikan: seluruh perhitungan ini menggantikan hanya satu baris di model asli — perhitungan spatial lag statis.

---

## Slide 17: Section — Mathematical Proofs

*(Halaman pemisah bagian.)*

---

## Slide 18: Proof A — GAT Menghasilkan Attention Statis

> Bukti ini menjelaskan keterbatasan mendasar GAT yang memotivasi pilihan kita menggunakan GATv2.
>
> Bayangkan provinsi i sedang bertanya: "provinsi mana yang paling penting buat saya sekarang?" Skor attention e-ij mengukur seberapa penting provinsi j bagi provinsi i.
>
> Masalah GAT: jawabannya tidak bergantung pada siapa yang bertanya. Kalau Jawa menganggap Sulawesi lebih penting dari Papua, maka setiap provinsi berpikiran sama — termasuk Papua sendiri. Saat El Niño, ini tidak masuk akal. Papua seharusnya peduli pada provinsi yang berbeda dari Jawa.
>
> Buktinya singkat. Karena LeakyReLU bekerja per elemen, skor GAT terpecah jadi dua bagian: g-dari-i yang hanya bergantung pada provinsi yang bertanya, dan h-dari-j yang hanya bergantung pada provinsi yang dievaluasi. Saat kita membandingkan dua tetangga, bagian g-dari-i-nya sama dan saling menghapus. Jadi urutannya hanya bergantung pada h-dari-j — tidak peduli siapa yang bertanya.
>
> Itulah kenapa CASA menggunakan pendekatan GATv2, di mana matriks bobot melihat kedua node bersamaan sebelum nonlinearitas diterapkan.

---

## Slide 19: Proofs B-D — Jaminan Matematis

> CASA punya tiga jaminan matematis.
>
> Proof B: matriks bobot W-sub-t bersifat row-stochastic. Setiap baris jumlahnya 1, jadi spatial lag adalah rata-rata berbobot yang valid. Outputnya tetap dalam skala yang sama dengan data curah hujan.
>
> Proof C: CASA hanya bisa menyamai atau mengalahkan model statis. Kalau ternyata tidak ada pola baru yang berguna untuk dipelajari, optimizer akan mendorong bias ke nilai sangat besar, alpha menuju 1, dan kita kembali ke bobot geografi asli. Secara teori training error pasti tidak lebih buruk. Untuk test error, karena parameter tambahan CASA sangat kecil dibanding total model, risiko overfitting-nya rendah — tapi tetap akan kita verifikasi lewat eksperimen.
>
> Proof D: bobot tidak bisa bergeser terlalu jauh dari geografi. Jarak antara W-sub-t dan matriks geografi dikontrol langsung oleh alpha-t. Alpha-t lebih tinggi berarti lebih dekat ke geografi. Menariknya, ini berarti alpha-t berfungsi ganda: sebagai pengatur campuran sekaligus sebagai pembatas otomatis — kita tidak perlu menambahkan regularisasi L2 secara terpisah. Dan batasannya menyesuaikan kondisi: ketat saat normal, longgar saat El Niño.

---

## Slide 20: Section — Positioning

*(Halaman pemisah bagian.)*

---

## Slide 21: No Prior Method Combines All of These

> Tabel ini membandingkan CASA dengan metode yang sudah saya perkenalkan tadi. Bisa dilihat Graph WaveNet dan AGCRN mempelajari bobot adaptif tapi tidak per time step, dan tidak punya basis geografi. GAT dan GATv2 dinamis tapi tidak menggunakan informasi iklim dan tidak punya titik awal geografi. WST-ANet punya sedikit conditioning iklim tapi tidak punya blending gate.
>
> CASA adalah yang pertama menggabungkan semua ini dalam satu modul, khususnya untuk peramalan curah hujan.

---

## Slide 22: Section — Validation Plan

*(Halaman pemisah bagian.)*

---

## Slide 23: Why Validation? And What Will We Test?

> CASA masih berupa usulan saat ini. Bukti matematisnya menunjukkan rumusnya benar, tapi kita belum mengujinya di data nyata. Ada empat hal yang perlu dicek.
>
> Pertama: apakah CASA mengalahkan model baseline? Kita akan bandingkan seluruh rantai — dari GSTAR ke GSTARX ke LSTM-GSTARX ke LSTM-GSTARX-CASA.
>
> Kedua: komponen mana yang benar-benar membantu? Kita akan jalankan eksperimen ablasi — menghilangkan basis geografi, menghilangkan input iklim, membekukan gate, membatasi hanya ke tetangga. Kalau setiap penghilangan menurunkan performa, itu mengonfirmasi setiap pilihan desain berkontribusi.
>
> Ketiga: apakah alpha-t berperilaku sesuai harapan secara fisik? Kita akan plot nilainya sepanjang waktu bersama Niño 3.4, dan membandingkan nilai alpha-t di periode El Niño, La Niña, dan Neutral menggunakan Kruskal-Wallis test — ini tes standar untuk mengecek apakah tiga kelompok atau lebih berasal dari distribusi yang sama. Kita juga akan melihat peta panas W-sub-t di berbagai rezim iklim.
>
> Keempat: apakah perbedaannya nyata, bukan kebetulan? Kita akan pakai Diebold-Mariano test — tes standar yang membandingkan error prediksi dari dua model peramalan dan menentukan apakah salah satunya benar-benar lebih baik. Ini diterapkan pada nilai RMSE dari lima-fold walk-forward cross-validation.

---

## Slide 24: Thank You

> Itu progress report saya. Ringkasannya: CASA adalah modul ringan yang membuat bobot spasial merespons kondisi iklim. Tiga tahap — konteks, attention, dan blending — dengan jaminan matematis bahwa bobotnya valid, tidak bisa lebih buruk dari model statis, dan tetap dekat geografi saat data menunjukkan demikian.
>
> Langkah selanjutnya: menulis kode CASA, menghubungkannya ke pipeline cross-validation, dan menjalankan eksperimen validasi.
>
> Terima kasih. Saya siap menerima pertanyaan.

---

## Slide 25: References

*(Tampilkan jika perlu. Daftar referensi lengkap ada di layar.)*
