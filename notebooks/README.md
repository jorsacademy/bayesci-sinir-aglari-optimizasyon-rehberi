# Uygulamalı Notebooklar

Bu klasör, Bayesçi sinir ağlarının endüstri mühendisliği ve yöneylem araştırması problemlerine nasıl bağlanabileceğini üç farklı karar mimarisi üzerinden gösterir.

## 1. BNN ile Talep Belirsizliği ve Stok/Üretim Optimizasyonu

[`bnn_talep_stok_optimizasyonu.ipynb`](./bnn_talep_stok_optimizasyonu.ipynb)

Akış:

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

Bu notebook BNN'nin **senaryo üreticisi**, Pyomo'nun ise **karar çözücüsü** olduğu temel mimariyi gösterir.

## 2. BNN + CVaR + Chance Constraint ile Risk-Duyarlı Üretim Planlama

[`bnn_cvar_chance_constraint_uretim.ipynb`](./bnn_cvar_chance_constraint_uretim.ipynb)

Aynı posterior predictive talep dağılımı altında üç karar yaklaşımını karşılaştırır:

- beklenen maliyet / SAA,
- CVaR95 optimizasyonu,
- %95 hizmet seviyesi sağlayan chance-constraint yaklaşımı.

Temel fikir:

```text
BNN posterioru
      ↓
Talep senaryoları
      ↓
Risk tercihi
  ↙    ↓     ↘
SAA   CVaR   Chance Constraint
  ↘    ↓     ↙
   Pyomo + HiGHS
        ↓
Karar kalitesi karşılaştırması
```

Bu örnek, **uncertainty modeling** ile **risk preference** kavramlarının birbirinden ayrı olduğunu gösterir.

## 3. Pyro BNN Surrogate + BoTorch ile Bayesçi Optimizasyon

[`bnn_botorch_bayesian_optimizasyon.ipynb`](./bnn_botorch_bayesian_optimizasyon.ipynb)

Pahalı black-box proses veya simülasyon fonksiyonları için:

```text
Başlangıç deneyleri
       ↓
Pyro BNN surrogate
       ↓
Posterior predictive samples
       ↓
BoTorch EnsembleModel / Posterior
       ↓
qLogExpectedImprovement
       ↓
Yeni deney noktası
       ↓
Sistemi değerlendir ve döngüyü tekrarla
```

Örnek iki proses parametresi üzerinde çalışır. Aynı yapı aşağıdakilere uyarlanabilir:

- ayrık olay simülasyonu,
- dijital ikiz,
- gerçek fabrika deneyleri,
- FEA / CFD,
- enerji sistemi simülasyonu,
- kalite ve proses optimizasyonu.

BoTorch'un Monte Carlo acquisition fonksiyonları surrogate modelin GP olmasını zorunlu kılmaz; örneklenebilir bir posterior sağlayan custom modellerle de çalışabilir. Bu notebook Pyro posterior predictive örneklerini `EnsembleModel` üzerinden BoTorch'a bağlar.

## Kurulum

Repo kök dizininde:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Daha sonra `notebooks/` klasöründeki notebookları açıp hücreleri sırayla çalıştırın.

## Kullanılan kütüphaneler

- `torch`: sinir ağı altyapısı
- `pyro-ppl`: Bayesçi sinir ağı ve SVI
- `botorch`: Bayesçi optimizasyon ve acquisition fonksiyonları
- `pyomo`: matematiksel programlama
- `highspy`: HiGHS LP/MIP çözücüsü
- `numpy`, `pandas`, `matplotlib`: veri işleme ve görselleştirme

## Endüstri mühendisliği açısından genel mimari

BNN çoğu örnekte doğrudan optimizasyon çözücüsü değildir. Üç temel rol üstlenir:

1. **Belirsiz parametre modeli:** talep, işlem süresi, lead time, arıza riski vb.
2. **Senaryo üreticisi:** stochastic programming, CVaR veya chance constraints için posterior örnekleri.
3. **Surrogate model:** pahalı simülasyon veya fiziksel deneylerin Bayesçi optimizasyonu.

Bu mimariler şu belirsiz büyüklüklere uygulanabilir:

- işlem süreleri → çizelgeleme,
- lead time → tedarik zinciri,
- arıza / kalan faydalı ömür → bakım planlama,
- taşıma süreleri → araç rotalama,
- enerji tüketimi / kalite → proses optimizasyonu,
- pahalı simülasyon çıktıları → simulation optimization.

## BNN aileleri

Repo kapsamında henüz notebooka dönüşmemiş BNN türleri ve önerilen öncelik sırası için kök dizindeki [`BNN_AILELERI.md`](../BNN_AILELERI.md) dosyasına bakın.

## Not

Örnekler öğretim amaçlıdır ve sentetik veri kullanır. Gerçek karar sistemlerinde posterior kalibrasyonu, out-of-distribution davranışı, senaryo sayısı duyarlılığı, baseline modeller, computation budget ve downstream karar kalitesi ayrıca doğrulanmalıdır.
