# Uygulamalı Notebooklar ve Rehberler

Bu klasör, Bayesçi sinir ağlarının ve yakın Bayesçi belirsizlik modellerinin endüstri mühendisliği / yöneylem araştırması kararlarına nasıl bağlanabileceğini **sekiz farklı mimari** üzerinden gösterir.

## 1. BNN ile Talep Belirsizliği ve Stok/Üretim Optimizasyonu

[`bnn_talep_stok_optimizasyonu.ipynb`](./bnn_talep_stok_optimizasyonu.ipynb)

```text
Talep verisi → Pyro BNN → posterior predictive senaryolar → SAA → Pyomo + HiGHS → üretim kararı
```

BNN'nin senaryo üreticisi, matematiksel programlamanın ise karar çözücüsü olduğu temel mimaridir.

## 2. BNN + CVaR + Chance Constraint

[`bnn_cvar_chance_constraint_uretim.ipynb`](./bnn_cvar_chance_constraint_uretim.ipynb)

Aynı posterior predictive talep dağılımı altında beklenen maliyet / SAA, CVaR95 ve %95 hizmet seviyesi / chance constraint yaklaşımlarını karşılaştırır. `uncertainty model` ile `risk preference` kavramlarının ayrı olduğunu gösterir.

## 3. Pyro BNN Surrogate + BoTorch ile Bayesçi Optimizasyon

[`bnn_botorch_bayesian_optimizasyon.ipynb`](./bnn_botorch_bayesian_optimizasyon.ipynb)

```text
Başlangıç deneyleri → Pyro BNN surrogate → posterior samples → BoTorch → acquisition → yeni deney
```

Pahalı fiziksel deneyler, ayrık olay simülasyonları, dijital ikiz, FEA/CFD ve proses optimizasyonu için BNN'nin surrogate rolünü gösterir.

## 4. Heteroskedastik BNN + Aleatorik/Epistemik Ayrıştırma

[`heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb`](./heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb)

Talep varyansının girdiye göre değiştiği durumda BNN koşullu ortalama ve girdiye bağlı gözlem gürültüsünü birlikte öğrenir. Toplam varyans yasısıyla

\[
Var(Y\mid x,D)=E_w[\sigma_w^2(x)]+Var_w[\mu_w(x)]
\]

ayrıştırması yapılır ve posterior predictive senaryolar CVaR / chance-constraint kararlarına bağlanır.

## 5. Bayesian Last Layer / Neural Linear + BoTorch

[`bayesian_last_layer_botorch.ipynb`](./bayesian_last_layer_botorch.ipynb)

Bu notebook **tam BNN değildir**. Gizli katmanlar deterministik olarak özellik öğrenir; yalnız son lineer katman Bayesçi modellenir.

```text
Deney verisi → deterministik backbone → φ(x) → Bayesçi lineer son katman → posterior → BoTorch
```

Gaussian prior + Gaussian likelihood altında son katman posterioru kapalı formda hesaplanabildiğinden full BNN'ye göre daha hafiftir. Sınırlaması: backbone belirsizliği posteriora taşınmaz.

## 6. NumPyro NUTS vs Pyro Variational Inference

[`numpyro_nuts_vs_pyro_vi_bnn.ipynb`](./numpyro_nuts_vs_pyro_vi_bnn.ipynb)

Aynı küçük BNN mimarisini ve aynı priorları iki çıkarım yöntemiyle karşılaştırır:

```text
aynı veri + aynı BNN
       ↓
Pyro VI       NumPyro NUTS
       ↓
posterior predictive
       ↓
RMSE / coverage / interval width
       ↓
beklenen maliyet / CVaR kararı
```

Karşılaştırma predictive accuracy ile sınırlı değildir; posterior yaklaşımının downstream operasyonel kararı değiştirip değiştirmediği ölçülür. NUTS otomatik olarak “ground truth posterior” ilan edilmez; divergence, ESS ve ciddi analizde çoklu-chain / R-hat kontrolleri gerekir.

## 7. Multi-output BNN + Çok Amaçlı Bayesçi Optimizasyon

[`multi_output_bnn_multi_objective_botorch.ipynb`](./multi_output_bnn_multi_objective_botorch.ipynb)

Aynı proses ayarının üç çıktısını ortak Bayesçi ağ ile modeller:

- kalite ↑,
- enerji ↓,
- çevrim süresi ↓.

```text
proses deneyleri
      ↓
shared multi-output BNN
      ↓
3 çıktının posterioru
      ↓
Pareto kümesi + hypervolume
      ↓
BoTorch qLogEHVI
      ↓
yeni deney noktası
```

Önemli ayrım: **multi-output** modelleme birden fazla rassal çıktının ortak modellenmesidir; **multi-objective optimization** ise bu çıktılar arasındaki tercih ve trade-off yapısını tanımlar. Notebook ortak Bayesçi gizli katman kullanır; residual likelihood diagonal olduğu için tam multivariate residual covariance modeli değildir.

## 8. Hierarchical BNN + Çok Fabrikalı Partial Pooling + Kapasite Tahsisi

[`hierarchical_bnn_partial_pooling_kapasite_tahsisi.md`](./hierarchical_bnn_partial_pooling_kapasite_tahsisi.md)

Dört fabrikanın aynı proses fiziğini paylaştığı fakat veri miktarlarının ve lokal performanslarının farklı olduğu bir örnek kurar.

```text
çok fabrikalı veri
      ↓
shared Bayesian NN
      +
fabrika random intercept / slope
      ↓
hierarchical prior + partial pooling
      ↓
fabrika bazlı posterior kapasite senaryoları
      ↓
Pyomo stokastik kapasite tahsisi
```

Modelde fabrika etkileri bağımsız sabit parametreler değildir:

\[
a_j\sim N(\mu_a,\tau_a^2),
\qquad
b_j\sim N(\mu_b,\tau_b^2).
\]

Bu nedenle az verili fabrikanın tahmini diğer fabrikalardan öğrenilen ortak yapıyla düzenlenebilir. Rehber ayrıca posterior-mean-only baseline ile stochastic allocation kararını karşılaştırır.

Kritik uyarı: **her hiyerarşik problem BNN gerektirmez**. Mixed-effects / hierarchical regression, GP ve complete/no-pooling baseline'ları mutlaka değerlendirilmelidir.

## Kurulum

Repo kök dizininde:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Kullanılan ana kütüphaneler

- `torch`: sinir ağı altyapısı
- `pyro-ppl`: BNN, hierarchical modeling ve variational inference
- `jax`, `numpyro`: NUTS / HMC tabanlı posterior örnekleme
- `botorch`: tek ve çok amaçlı Bayesçi optimizasyon
- `pyomo`: matematiksel programlama
- `highspy`: HiGHS LP/MIP çözücüsü
- `numpy`, `pandas`, `matplotlib`: veri işleme ve görselleştirme

## Endüstri mühendisliği açısından genel roller

Bu repoda BNN/Bayesian surrogate çoğunlukla optimizasyon çözücüsünün kendisi değildir. Şu rolleri üstlenir:

1. **Belirsiz parametre modeli:** talep, işlem süresi, lead time, arıza riski vb.
2. **Senaryo üreticisi:** stochastic programming, CVaR ve chance constraints.
3. **Surrogate model:** pahalı simülasyon veya fiziksel deneylerin Bayesçi optimizasyonu.
4. **Belirsizlik ayrıştırıcı:** aleatorik ve epistemik bileşenleri karar modeline taşıma.
5. **Hafif online surrogate:** Bayesian Last Layer ile düşük maliyetli posterior güncellemesi.
6. **Inference hassasiyet analizi:** VI / MCMC seçiminin downstream kararı değiştirip değiştirmediğini ölçme.
7. **Çok çıktılı karar modeli:** kalite–enerji–süre gibi trade-off'larda Pareto ve hypervolume tabanlı deney seçimi.
8. **Partial-pooling grup modeli:** fabrika/makine/tedarikçi gibi gruplar arasında bilgi paylaşarak grup bazlı karar belirsizliği üretme.

## BNN aileleri ve teorik dayanak

Yöntem taksonomisi için [`BNN_AILELERI.md`](../BNN_AILELERI.md), genel matematiksel dayanak için [`TEORIK_DAYANAK.md`](../TEORIK_DAYANAK.md), multi-output/MOBO notları için [`MULTI_OUTPUT_MOBO_DAYANAK.md`](../MULTI_OUTPUT_MOBO_DAYANAK.md) dosyasına bakın.

## Not

Örnekler öğretim amaçlıdır ve sentetik veri kullanır. Gerçek karar sistemlerinde posterior kalibrasyonu, OOD davranışı, grup yapısının geçerliliği, pooling seviyesi, baseline modeller, MCMC diagnostics, hesaplama bütçesi ve **out-of-sample downstream karar kalitesi** ayrıca doğrulanmalıdır.
