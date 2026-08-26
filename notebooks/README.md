# Uygulamalı Notebooklar

Bu klasör, Bayesçi sinir ağlarının ve yakın Bayesçi belirsizlik modellerinin endüstri mühendisliği / yöneylem araştırması kararlarına nasıl bağlanabileceğini beş farklı mimari üzerinden gösterir.

## 1. BNN ile Talep Belirsizliği ve Stok/Üretim Optimizasyonu

[`bnn_talep_stok_optimizasyonu.ipynb`](./bnn_talep_stok_optimizasyonu.ipynb)

```text
Talep verisi → Pyro BNN → posterior predictive senaryolar → SAA → Pyomo + HiGHS → üretim kararı
```

BNN'nin senaryo üreticisi, matematiksel programlamanın ise karar çözücüsü olduğu temel mimaridir.

## 2. BNN + CVaR + Chance Constraint

[`bnn_cvar_chance_constraint_uretim.ipynb`](./bnn_cvar_chance_constraint_uretim.ipynb)

Aynı posterior predictive talep dağılımı altında:

- beklenen maliyet / SAA,
- CVaR95,
- %95 hizmet seviyesi / chance constraint

yaklaşımlarını karşılaştırır. `uncertainty model` ile `risk preference` kavramlarının ayrı olduğunu gösterir.

## 3. Pyro BNN Surrogate + BoTorch ile Bayesçi Optimizasyon

[`bnn_botorch_bayesian_optimizasyon.ipynb`](./bnn_botorch_bayesian_optimizasyon.ipynb)

```text
Başlangıç deneyleri → Pyro BNN surrogate → posterior samples → BoTorch → acquisition → yeni deney
```

Pahalı fiziksel deneyler, ayrık olay simülasyonları, dijital ikiz, FEA/CFD ve proses optimizasyonu için BNN'nin surrogate rolünü gösterir.

BoTorch'un Monte Carlo acquisition fonksiyonları surrogate modelin GP olmasını zorunlu kılmaz; örneklenebilir bir posterior sağlayan custom modellerle de çalışabilir.

## 4. Heteroskedastik BNN + Aleatorik/Epistemik Ayrıştırma

[`heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb`](./heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb)

Talep varyansının girdiye göre değiştiği durumda BNN aynı anda:

- \(\mu_w(x)\): koşullu ortalama,
- \(\sigma_w(x)\): girdiye bağlı gözlem gürültüsü

öğrenir. Toplam varyans yasasıyla:

\[
Var(Y\mid x,D)=E_w[\sigma_w^2(x)]+Var_w[\mu_w(x)]
\]

ayrıştırması yapılır ve posterior predictive senaryolar CVaR/chance-constraint kararlarına bağlanır.

Notebook ayrıca, posterior ortalama fonksiyonunu kullanıp gözlem gürültüsünü yok saymanın riskli bölgelerde kapasiteyi düşük tahmin edebileceğini açıkça gösterir.

## 5. Bayesian Last Layer / Neural Linear + BoTorch

[`bayesian_last_layer_botorch.ipynb`](./bayesian_last_layer_botorch.ipynb)

Bu notebook **tam BNN değildir**. Gizli katmanlar deterministik olarak özellik öğrenir; yalnız son lineer katmanın ağırlıkları Bayesçi modellenir:

```text
Deney verisi
    ↓
deterministik backbone
    ↓
φ(x): öğrenilmiş özellikler
    ↓
Bayesçi lineer son katman
β | D ~ N(m, Σ)
    ↓
posterior ağırlık örnekleri
    ↓
BoTorch EnsembleModel
    ↓
qLogExpectedImprovement
    ↓
yeni deney noktası
```

Gaussian prior + Gaussian likelihood altında son katman posterioru kapalı formda hesaplanabildiğinden full BNN'ye göre çok daha hafiftir. Özellikle online/sequential optimization için pragmatik bir alternatiftir.

Önemli sınırlama: backbone ağırlıklarının belirsizliği posteriora taşınmaz. Bu nedenle BLL'nin epistemik belirsizliği full BNN ile aynı değildir ve OOD bölgelerinde aşırı güven sorunu görülebilir.

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

## Endüstri mühendisliği açısından genel roller

Bu repoda BNN/Bayesian surrogate doğrudan optimizasyon çözücüsü olmaktan çok şu rolleri üstlenir:

1. **Belirsiz parametre modeli:** talep, işlem süresi, lead time, arıza riski vb.
2. **Senaryo üreticisi:** stochastic programming, CVaR ve chance constraints.
3. **Surrogate model:** pahalı simülasyon veya fiziksel deneylerin Bayesçi optimizasyonu.
4. **Belirsizlik ayrıştırıcı:** aleatorik ve epistemik bileşenleri karar modeline taşıma.
5. **Hafif online surrogate:** Bayesian Last Layer ile düşük maliyetli posterior güncellemesi.

Bu mimariler işlem süresi, lead time, arıza/RUL, taşıma süresi, enerji tüketimi, kalite, talep ve pahalı simülasyon çıktıları gibi IE/OR büyüklüklerine uyarlanabilir.

## BNN aileleri ve teorik dayanak

Henüz notebooka dönüşmemiş yöntemler için [`BNN_AILELERI.md`](../BNN_AILELERI.md), matematiksel dayanak ve sınırlar için [`TEORIK_DAYANAK.md`](../TEORIK_DAYANAK.md) dosyasına bakın.

## Not

Örnekler öğretim amaçlıdır ve sentetik veri kullanır. Gerçek karar sistemlerinde posterior kalibrasyonu, OOD davranışı, senaryo sayısı duyarlılığı, baseline modeller, hesaplama bütçesi ve **out-of-sample downstream karar kalitesi** ayrıca doğrulanmalıdır.
