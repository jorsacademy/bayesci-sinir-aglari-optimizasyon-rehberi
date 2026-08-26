# Uygulamalı Notebooklar

## BNN ile Talep Belirsizliği ve Stok/Üretim Optimizasyonu

Ana örnek:

[`bnn_talep_stok_optimizasyonu.ipynb`](./bnn_talep_stok_optimizasyonu.ipynb)

Bu notebook aşağıdaki uçtan uca akışı gösterir:

```text
Sentetik / gerçek talep verisi
        ↓
Pyro ile Bayesçi Sinir Ağı
        ↓
Variational Inference (SVI)
        ↓
Posterior predictive talep senaryoları
        ↓
Sample Average Approximation (SAA)
        ↓
Pyomo + HiGHS
        ↓
Optimal üretim miktarı
        ↓
Deterministik karar ile maliyet / stokout / CVaR karşılaştırması
```

## Kurulum

Repo kök dizininde:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Daha sonra `notebooks/bnn_talep_stok_optimizasyonu.ipynb` dosyasını açın ve hücreleri sırayla çalıştırın.

## Kullanılan kütüphaneler

- `torch`: sinir ağı altyapısı
- `pyro-ppl`: Bayesçi sinir ağı ve SVI
- `pyomo`: matematiksel programlama
- `highspy`: HiGHS LP/MIP çözücüsü
- `numpy`, `pandas`, `matplotlib`: veri işleme ve görselleştirme

## Endüstri mühendisliği açısından ne gösteriyor?

Notebookta BNN doğrudan optimizasyon çözücüsü değildir. BNN, belirsiz talep için posterior predictive dağılım üretir. Bu dağılımdan elde edilen senaryolar Pyomo modeline aktarılır ve karar değişkeni stokastik programlama ile seçilir.

Bu mimari talep yerine aşağıdaki belirsiz büyüklüklere de uygulanabilir:

- işlem süreleri → çizelgeleme,
- lead time → tedarik zinciri,
- arıza / kalan faydalı ömür → bakım planlama,
- taşıma süreleri → araç rotalama,
- enerji tüketimi / kalite → proses optimizasyonu,
- pahalı simülasyon çıktıları → simulation optimization.

## Not

Örnek öğretim amaçlıdır ve sentetik veri kullanır. Gerçek karar sistemlerinde posterior kalibrasyonu, out-of-distribution davranışı, senaryo sayısı duyarlılığı, baseline modeller ve karar kalitesi metrikleri ayrıca doğrulanmalıdır.
