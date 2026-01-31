# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang
Kebutuhan energi listrik yang terus meningkat seiring dengan pertumbuhan ekonomi dan populasi menuntut penyediaan pasokan listrik yang andal, efisien, dan ramah lingkungan. PT PLN Indonesia Power sebagai salah satu penyedia energi utama di Indonesia terus berupaya mengoptimalkan kinerja unit pembangkitnya. Salah satu jenis pembangkit yang memiliki peran vital dalam sistem kelistrikan adalah Pembangkit Listrik Tenaga Gas dan Uap (PLTGU).

UPPLTGU Keramasan, khususnya Unit 1, beroperasi menggunakan Turbin Gas sebagai *prime mover* utama. Efisiensi operasi turbin gas sangat bergantung pada kualitas proses pembakaran yang terjadi di ruang bakar (*combustion chamber*). Proses pembakaran yang tidak sempurna tidak hanya menurunkan efisiensi termal dan meningkatkan konsumsi bahan bakar (*heat rate*), tetapi juga berdampak pada peningkatan emisi gas buang.

Di sisi lain, isu perubahan iklim global mendorong industri pembangkitan untuk lebih peduli terhadap dampak lingkungan, khususnya terkait emisi Gas Rumah Kaca (GRK) seperti Karbon Dioksida ($CO_2$). Estimasi *Carbon Footprint* atau jejak karbon menjadi indikator penting untuk mengevaluasi kinerja lingkungan suatu pembangkit. Oleh karena itu, analisis mendalam mengenai proses pembakaran pada kondisi operasi *Base Load* dan estimasi jejak karbon yang dihasilkannya menjadi sangat relevan untuk dilakukan guna mendukung operasional yang efisien dan berkelanjutan.

## 1.2 Rumusan Masalah
Berdasarkan latar belakang di atas, rumusan masalah dalam kerja praktik ini adalah:
1.  Bagaimana karakteristik dan efisiensi proses pembakaran pada Turbin Gas Unit 1 UPPLTGU Keramasan dalam kondisi operasi *Base Load*?
2.  Berapa besar estimasi *Carbon Footprint* (emisi $CO_2$) yang dihasilkan dari operasional Turbin Gas Unit 1 pada kondisi tersebut?
3.  Faktor-faktor apa saja yang mempengaruhi performa pembakaran dan besaran emisi yang dihasilkan?

## 1.3 Tujuan Penelitian
Tujuan dari pelaksanaan kerja praktik dan penyusunan laporan ini adalah:
1.  Menganalisis proses pembakaran dan menghitung efisiensi termal Turbin Gas Unit 1 UPPLTGU Keramasan pada beban dasar.
2.  Mengestimasi jumlah emisi $CO_2$ (*Carbon Footprint*) berdasarkan konsumsi bahan bakar menggunakan standar perhitungan yang berlaku (IPCC).
3.  Memberikan gambaran kinerja unit pembangkit ditinjau dari aspek efisiensi energi dan dampak lingkungan.

## 1.4 Batasan Masalah
Agar pembahasan lebih terarah, penulisan laporan ini dibatasi pada:
1.  Objek penelitian adalah Turbin Gas (Gas Turbine) Unit 1 di UPPLTGU Keramasan.
2.  Data yang dianalisis adalah data operasi pada kondisi *Base Load* (beban dasar) selama periode kerja praktik.
3.  Perhitungan emisi difokuskan pada emisi $CO_2$ dari pembakaran bahan bakar gas alam, tidak mencakup emisi siklus hidup (*Life Cycle Assessment*) secara keseluruhan.
4.  Analisis dilakukan berdasarkan data sekunder dari *log sheet* harian dan spesifikasi teknis peralatan.

## 1.5 Manfaat Penelitian
**Bagi Perusahaan:**
Memberikan masukan dan evaluasi independen mengenai kinerja unit pembangkit, khususnya terkait efisiensi pembakaran dan status emisi karbon terkini.

**Bagi Mahasiswa:**
Mengaplikasikan ilmu termodinamika dan konversi energi yang diperoleh di bangku perkuliahan ke dalam kasus nyata di industri pembangkitan listrik.

**Bagi Institusi Pendidikan:**
Menambah referensi kepustakaan mengenai analisis performa PLTGU dan isu lingkungan di sektor energi.

# BAB II
# DASAR TEORI

## 2.1 Pembangkit Listrik Tenaga Gas dan Uap (PLTGU)
Pembangkit Listrik Tenaga Gas dan Uap (PLTGU) adalah jenis pembangkit listrik yang menggabungkan dua siklus termodinamika utama, yaitu siklus Brayton (siklus turbin gas) dan siklus Rankine (siklus uap). Penggabungan ini dikenal sebagai *combined cycle*. Tujuan utama dari kombinasi ini adalah untuk meningkatkan efisiensi termal pembangkit secara keseluruhan [1].

Pada PLTGU, gas buang panas (*exhaust gas*) yang dihasilkan dari operasi turbin gas tidak langsung dibuang ke atmosfer, melainkan dimanfaatkan kembali untuk memanaskan air di dalam *Heat Recovery Steam Generator* (HRSG). Uap bertekanan tinggi yang dihasilkan dari HRSG kemudian digunakan untuk memutar turbin uap yang terkopel dengan generator untuk menghasilkan listrik tambahan. Dengan memanfaatkan energi panas yang terbuang ini, efisiensi PLTGU bisa mencapai kisaran 50% hingga 60%, jauh lebih tinggi dibandingkan pembangkit siklus tunggal (*simple cycle*) yang biasanya hanya berkisar 30-40% [2].

## 2.2 Turbin Gas (*Gas Turbine*)
Turbin gas adalah mesin rotasi yang mengekstrak energi dari aliran gas pembakaran. Prinsip kerjanya didasarkan pada siklus Brayton yang terdiri dari tiga komponen utama [3]:

1.  **Kompresor (*Compressor*)**: Berfungsi untuk menghisap udara dari lingkungan dan menaikkan tekanannya. Udara yang terkompresi ini kemudian diarahkan ke ruang bakar. Rasio tekanan (*pressure ratio*) pada kompresor sangat mempengaruhi efisiensi turbin gas.
2.  **Ruang Bakar (*Combustion Chamber*)**: Di sinilah terjadi pencampuran udara bertekanan dengan bahan bakar (gas alam atau minyak). Campuran ini kemudian dibakar untuk menghasilkan gas panas bertekanan tinggi dan berkecepatan tinggi. Proses pembakaran ini harus berlangsung stabil untuk menjaga efisiensi dan keamanan operasi [3].
3.  **Turbin (*Turbine*)**: Gas panas hasil pembakaran kemudian diekspansikan melalui sudu-sudu turbin, mengubah energi kinetik dan termal gas menjadi energi mekanik putaran poros. Sebagian energi mekanik ini digunakan untuk memutar kompresor (sekitar 50-60%), dan sisanya digunakan untuk memutar generator listrik atau beban mekanik lainnya.

## 2.3 Proses Pembakaran
Pembakaran adalah reaksi kimia eksotermik antara bahan bakar dan oksidan (biasanya oksigen di udara) yang menghasilkan energi panas. Pada turbin gas, pembakaran ideal (stoikiometri) sangat jarang terjadi secara sempurna di seluruh bagian ruang bakar karena kebutuhan untuk menjaga material sudu turbin dari temperatur yang terlalu ekstrem (*Turbine Inlet Temperature* - TIT). Oleh karena itu, udara yang dimasukkan biasanya berlebih (*excess air*) untuk pendinginan [4].

Reaksi umum pembakaran gas alam (Metana - CH₄) adalah:
$$CH_4 + 2O_2 \rightarrow CO_2 + 2H_2O + Energi$$

Faktor-faktor yang mempengaruhi keberhasilan pembakaran dalam turbin gas meliputi:
*   **Rasio Udara-Bahan Bakar (*Air-Fuel Ratio*)**: Menentukan temperatur api dan stabilitas pembakaran.
*   **Pencampuran (*Mixing*)**: Homogenitas campuran udara dan bahan bakar sangat penting untuk pembakaran yang efisien dan minim emisi.
*   **Waktu Tinggal (*Residence Time*)**: Waktu yang cukup dibutuhkan agar reaksi pembakaran tuntas sebelum gas keluar dari ruang bakar.

## 2.4 Operasi *Base Load*
Operasi *Base Load* (Beban Dasar) merujuk pada mode operasi pembangkit listrik di mana unit beroperasi secara terus-menerus pada kapasitas maksimum atau mendekati maksimum untuk memenuhi permintaan listrik dasar yang konstan sepanjang waktu [2].

Unit pembangkit yang beroperasi sebagai *base load* biasanya adalah unit yang memiliki biaya produksi listrik per kWh yang paling rendah (efisiensi tinggi) dan keandalan yang tinggi. PLTGU sering dijadikan pembangkit *base load* karena efisiensinya yang tinggi, meskipun fleksibilitasnya juga memungkinkan untuk operasi *load following*. Dalam konteks analisis data, kondisi *base load* memberikan data operasi yang relatif stabil (steady state), sehingga memudahkan analisis parameter kinerja seperti *Heat Rate* dan efisiensi termal tanpa gangguan fluktuasi beban yang signifikan.

## 2.5 Estimasi *Carbon Footprint*
*Carbon Footprint* atau jejak karbon adalah total emisi gas rumah kaca (GRK) yang dihasilkan secara langsung maupun tidak langsung oleh suatu individu, organisasi, peristiwa, atau produk. Dalam konteks pembangkit listrik termal, emisi utama yang menjadi perhatian adalah Karbon Dioksida ($CO_2$) [5].

Perhitungan emisi $CO_2$ dari pembakaran bahan bakar fosil umumnya mengikuti pedoman dari *Intergovernmental Panel on Climate Change* (IPCC). Rumus umum untuk estimasi emisi dari pembakaran stasioner adalah:

$$Emisi = Konsumsi Bahan Bakar \times Faktor Emisi$$

Dimana:
*   **Konsumsi Bahan Bakar**: Jumlah bahan bakar yang dibakar (dalam satuan massa atau volume, atau satuan energi seperti TJ).
*   **Faktor Emisi**: Koefisien yang menyatakan jumlah emisi yang dilepaskan per satuan bahan bakar yang dikonsumsi (misalnya kg $CO_2$/TJ atau kg $CO_2$/m³ gas) [6].

Emisi spesifik ($kg CO_2 / kWh$) sering digunakan sebagai indikator kinerja lingkungan, yang menunjukkan seberapa banyak polusi karbon yang dihasilkan untuk setiap satuan energi listrik yang diproduksi. Semakin tinggi efisiensi pembangkit, semakin rendah konsumsi bahan bakar untuk output daya yang sama, sehingga menurunkan emisi spesifik [5].

# BAB III
# METODOLOGI PENELITIAN

## 3.1 Tempat dan Waktu Pelaksanaan
Penelitian dan pengambilan data untuk kerja praktik ini dilaksanakan di Unit 1 UPPLTGU Keramasan PT PLN Indonesia Power. Waktu pelaksanaan kerja praktik dimulai dari tanggal [Tanggal Mulai] sampai dengan [Tanggal Selesai].

## 3.2 Alat dan Bahan
### 3.2.1 Alat
Perangkat keras dan lunak yang digunakan untuk mendukung penyusunan laporan ini meliputi:
1.  **Perangkat Keras (*Hardware*)**: Laptop dengan spesifikasi yang memadai untuk pengolahan data.
2.  **Perangkat Lunak (*Software*)**:
    *   Microsoft Office (Word, Excel) untuk penyusunan laporan dan pengolahan data awal.
    *   **Python (Google Colab / Jupyter Notebook)**: Digunakan untuk analisis data lanjutan dan pemodelan statistik/Machine Learning (*XGBoost*) guna melihat korelasi parameter operasi.
    *   Peramban web (*Browser*) untuk mencari referensi literatur.

### 3.2.2 Bahan
Bahan penelitian utama berupa data sekunder, yaitu data operasional harian (*log sheet*) unit Turbin Gas pada kondisi beban dasar (*base load*), spesifikasi teknis peralatan, dan buku manual operasi.

## 3.3 Metode Pengumpulan Data
Metode yang digunakan dalam pengumpulan data adalah:
1.  **Studi Literatur**: Mencari landasan teori dari buku, jurnal, dan dokumen standar perusahaan terkait sistem PLTGU, turbin gas, dan perhitungan emisi.
2.  **Observasi Lapangan**: Mengamati langsung peralatan utama dan sistem pendukung di PLTGU Keramasan (jika memungkinkan) serta melihat proses pencatatan data di *Control Room*.
3.  **Wawancara**: Melakukan diskusi dengan pembimbing lapangan dan operator untuk memahami kondisi real di lapangan.
4.  **Pengambilan Data Operasi**: Mengumpulkan data parameter kunci seperti beban generator (MW), aliran bahan bakar (*fuel flow*), temperatur gas buang (*exhaust temperature*), dan temperatur udara masuk kompresor.

## 3.4 Prosedur Penelitian
Tahapan pelaksanaan penelitian digambarkan dalam alur berikut:

1.  **Identifikasi Masalah**: Merumuskan tujuan analisis efisiensi pembakaran dan estimasi jejak karbon.
2.  **Studi Pustaka**: Mempelajari termodinamika siklus Brayton dan faktor emisi bahan bakar.
3.  **Pengumpulan Data**: Mengambil data operasi Unit 1 pada kondisi *Base Load*.
4.  **Pengolahan Data**:
    *   Menghitung efisiensi termal turbin gas.
    *   Menghitung estimasi emisi $CO_2$ menggunakan metode perhitungan stoikiometri atau faktor emisi IPCC.
    *   (Opsional) Melakukan pemodelan prediksi menggunakan algoritma *XGBoost* untuk melihat sensitivitas parameter operasi terhadap emisi/efisiensi.
5.  **Analisis dan Pembahasan**: Membandingkan hasil perhitungan dengan standar desain atau kriteria performa (`commissioning data`). Mengidentifikasi faktor penyebab deviasi jika ada.
6.  **Kesimpulan dan Saran**: Menarik kesimpulan dari hasil analisis dan memberikan rekomendasi untuk operasi yang lebih efisien dan ramah lingkungan.

# BAB IV
# HASIL DAN PEMBAHASAN

## 4.1 Data Operasi *Base Load*
Data operasi diambil dari *log sheet* harian Unit 1 UPPLTGU Keramasan pada periode pengamatan. Data difokuskan pada kondisi beban dasar (*Base Load*) untuk mendapatkan perhitungan yang stabil.

*(Tempatkan Tabel Data Operasi Di Sini: Waktu, Beban (MW), Fuel Flow (kg/s), Exhaust Temp (C), dll)*

## 4.2 Perhitungan Efisiensi Termal
Efisiensi termal turbin gas dihitung dengan membandingkan energi listrik yang dihasilkan dengan energi input dari bahan bakar.

$$ \eta_{th} = \frac{P_{out}}{\dot{m}_f \times LHV} \times 100\% $$

Dimana:
*   $P_{out}$ = Daya Output Generator (MW)
*   $\dot{m}_f$ = Laju Aliran Massa Bahan Bakar (kg/s)
*   $LHV$ = *Lower Heating Value* Gas Alam (kJ/kg)

*(Masukkan Hasil Perhitungan Rata-rata Di Sini)*

## 4.3 Estimasi *Carbon Footprint*
Emisi $CO_2$ diestimasi berdasarkan konsumsi bahan bakar dan faktor emisi gas alam sesuai standar IPCC atau spesifikasi gas.

$$ Emisi_{CO2} = Konsumsi_{Gas} \times EF_{Gas} $$

*(Masukkan Hasil Perhitungan Total Emisi dan Emisi Spesifik (kgCO2/kWh) Di Sini)*

## 4.4 Analisis
Berdasarkan hasil perhitungan di atas, dapat dilakukan analisis sebagai berikut:
1.  **Analisis Efisiensi**: Apakah efisiensi saat ini mengalami penurunan dibandingkan data komisioning/desain? Jika ya, apa faktor penyebabnya (misalnya: *fouling* pada kompresor, degradasi ruang bakar, tingginya temperatur udara masuk)?
2.  **Analisis Emisi**: Bagaimana tingkat emisi spesifik unit ini? Apakah masih dalam ambang batas regulasi lingkungan?
3.  **Korelasi Operasi**: (Jika menggunakan XGBoost/Python) Bagaimana pengaruh beban operasi terhadap efisiensi dan emisi?

# BAB V
# PENUTUP

## 5.1 Kesimpulan
Berdasarkan analisis yang telah dilakukan, dapat disimpulkan bahwa:
1.  Efisiensi termal rata-rata Unit 1 pada kondisi *Base Load* adalah ... %.
2.  Estimasi *Carbon Footprint* yang dihasilkan adalah ... ton $CO_2$/hari dengan emisi spesifik sebesar ... kg $CO_2$/kWh.
3.  ...(Tambahkan kesimpulan lain terkait performa/operasi).

## 5.2 Saran
1.  Disarankan untuk melakukan *compressor washing* secara berkala untuk menjaga efisiensi.
2.  Optimasi pola operasi diperlukan untuk menekan angka emisi.
3.  Penelitian selanjutnya dapat memperluas cakupan ke analisis ekonomi (*carbon tax*).

## Daftar Pustaka
[1] TUGAS AKHIR – TM141585 ANALISIS PERFORMA PLTGU PLN SEKTOR.
[2] ANALISIS PENGARUH PERUBAHAN BEBAN TERHADAP GENERATOR GT (GAS TURBINE) UNIT 1 DI PLTGU KERAMASAN.
[3] REKAYASA MEKANIKA Vol.5 No.2 │Oktober 2021 MAINTENANCE PADA COMBUSTION SECTION TURBIN GAS UNIT 2 PLTGU.
[4] Novel ammonia gas turbine – Carbon dioxide cycle combined system using thermochemical fuel reforming and rich-quench-lean combustor.
[5] Carbon capture considerations for combined cycle gas turbine.
[6] On the cost of zero carbon electricity A techno-economic analysis of combined cycle gas turbines with post-combustion CO2 capture.


