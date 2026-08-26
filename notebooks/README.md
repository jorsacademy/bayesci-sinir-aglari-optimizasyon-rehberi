# Uygulamalı Notebooklar ve Rehberler

Bu klasör, Bayesçi sinir ağları ve yakın belirsizlik modellerinin endüstri mühendisliği / yöneylem araştırması kararlarına nasıl bağlanabileceğini **dokuz farklı mimari** üzerinden gösterir.

## 1. BNN ile Talep Belirsizliği ve Stok/Üretim Optimizasyonu

[`bnn_talep_stok_optimizasyonu.ipynb`](./bnn_talep_stok_optimizasyonu.ipynb)

```text
Talep verisi → Pyro BNN → posterior predictive → SAA → Pyomo + HiGHS → üretim kararı
```

## 2. BNN + CVaR + Chance Constraint

[`bnn_cvar_chance_constraint_uretim.ipynb`](./bnn_cvar_chance_constraint_uretim.ipynb)

Aynı BNN posterioru altında beklenen maliyet, CVaR95 ve chance-constraint kararlarını karşılaştırır. Belirsizlik modeli ile risk tercihini ayrı katmanlar olarak ele alır.

## 3. Pyro BNN Surrogate + BoTorch

[`bnn_botorch_bayesian_optimizasyon.ipynb`](./bnn_botorch_bayesian_optimizasyon.ipynb)

```text
Başlangıç deneyleri → BNN surrogate → posterior samples → BoTorch acquisition → yeni deney
```

Pahalı simülasyon, dijital ikiz ve fiziksel deney optimizasyonu için surrogate kullanımını gösterir.

## 4. Heteroskedastik BNN

[`heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb`](./heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb)

BNN hem koşullu ortalamayı hem girdiye bağlı observation noise'u öğrenir:

\[
Var(Y\mid x,D)=E_w[\sigma_w^2(x)]+Var_w[\mu_w(x)].
\]

Posterior predictive senaryolar CVaR ve chance constraints'e bağlanır.

## 5. Bayesian Last Layer / Neural Linear + BoTorch

[`bayesian_last_layer_botorch.ipynb`](./bayesian_last_layer_botorch.ipynb)

Tam BNN değildir. Deterministik backbone sabit features üretir; yalnız son lineer katman Bayesçi modellenir. Full BNN'ye göre daha hafif bir online/sequential surrogate alternatifidir.

## 6. NumPyro NUTS vs Pyro VI

[`numpyro_nuts_vs_pyro_vi_bnn.ipynb`](./numpyro_nuts_vs_pyro_vi_bnn.ipynb)

Aynı küçük BNN'yi Pyro mean-field VI ve NumPyro NUTS ile karşılaştırır. RMSE yanında predictive coverage, interval width, posterior yayılımı, beklenen maliyet ve CVaR kararları değerlendirilir.

## 7. Multi-output BNN + Çok Amaçlı Bayesçi Optimizasyon

[`multi_output_bnn_multi_objective_botorch.ipynb`](./multi_output_bnn_multi_objective_botorch.ipynb)

Kalite ↑, enerji ↓ ve çevrim süresi ↓ amaçlarını ortak multi-output BNN ile modeller; Pareto, hypervolume ve BoTorch qLogEHVI ile yeni deney seçer.

Kritik ayrım: **multi-output modelleme ≠ multi-objective optimization**.

## 8. Hierarchical BNN + Partial Pooling + Kapasite Tahsisi

[`hierarchical_bnn_partial_pooling_kapasite_tahsisi.md`](./hierarchical_bnn_partial_pooling_kapasite_tahsisi.md)

Ortak nonlinear BNN response surface ile fabrika düzeyi random intercept / slope'ları birleştirir:

\[
a_j\sim N(\mu_a,\tau_a^2),
\qquad
b_j\sim N(\mu_b,\tau_b^2).
\]

Posterior fabrika kapasite senaryoları Pyomo stokastik kapasite tahsisine aktarılır. Her hiyerarşik problemin BNN gerektirmediği özellikle vurgulanır.

## 9. Laplace Approximation + Mevcut PyTorch Modeli

[`laplace_approximation_pytorch_stokastik_kapasite.ipynb`](./laplace_approximation_pytorch_stokastik_kapasite.ipynb)

Mevcut deterministik PyTorch modelini yeniden full BNN olarak eğitmeden post-hoc uncertainty-aware hale getirir:

```text
trained PyTorch NN
      ↓
MAP weights
      ↓
last-layer Laplace approximation
      ↓
approximate posterior predictive
      ↓
capacity scenarios
      ↓
Pyomo stochastic capacity planning
```

Notebook `laplace-torch` ile last-layer Laplace kullanır. Ayrıca full-network ve subnetwork Laplace seçeneklerini açıklar.

Laplace ile Bayesian Last Layer aynı yöntem değildir: BLL conjugate Bayesian linear posterior kullanırken Laplace, MAP çevresindeki curvature/Hessian ile yerel Gaussian posterior yaklaşımı kurar.

Teorik sınırlar için [`../LAPLACE_DAYANAK.md`](../LAPLACE_DAYANAK.md) dosyasına bakın.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Kullanılan ana kütüphaneler

- `torch`: sinir ağı altyapısı
- `pyro-ppl`: BNN, hierarchical modeling, VI
- `jax`, `numpyro`: HMC / NUTS
- `laplace-torch`: post-hoc Laplace approximation
- `botorch`: tek ve çok amaçlı Bayesçi optimizasyon
- `pyomo`, `highspy`: matematiksel programlama
- `numpy`, `pandas`, `matplotlib`: veri ve görselleştirme

## Endüstri mühendisliği açısından genel roller

Bu repoda BNN / Bayesian surrogate çoğunlukla çözücünün kendisi değildir. Başlıca roller:

1. belirsiz parametre modeli,
2. posterior / senaryo üreticisi,
3. pahalı sistemler için surrogate,
4. aleatorik–epistemik ayrıştırıcı,
5. hafif online uncertainty modeli,
6. inference hassasiyet analizi,
7. çok çıktılı / çok amaçlı deney tasarımı,
8. hierarchical partial pooling,
9. mevcut deterministic NN'ye post-hoc uncertainty ekleme.

## İlgili rehberler

- [`../BNN_AILELERI.md`](../BNN_AILELERI.md)
- [`../TEORIK_DAYANAK.md`](../TEORIK_DAYANAK.md)
- [`../MULTI_OUTPUT_MOBO_DAYANAK.md`](../MULTI_OUTPUT_MOBO_DAYANAK.md)
- [`../LAPLACE_DAYANAK.md`](../LAPLACE_DAYANAK.md)

## Not

Örnekler öğretim amaçlıdır ve çoğunlukla sentetik veri kullanır. Gerçek uygulamalarda calibration, OOD davranışı, posterior approximation hassasiyeti, baseline modeller, senaryo sayısı, computation budget ve **out-of-sample downstream karar kalitesi** ayrıca doğrulanmalıdır.
