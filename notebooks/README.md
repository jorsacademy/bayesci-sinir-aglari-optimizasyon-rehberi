# Uygulamalı Notebooklar

Bu klasör, Bayesçi sinir ağlarının ve yakın Bayesçi belirsizlik modellerinin endüstri mühendisliği / yöneylem araştırması kararlarına nasıl bağlanabileceğini altı farklı mimari üzerinden gösterir.

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

Talep varyansının girdiye göre değiştiği durumda BNN aynı anda koşullu ortalama ve girdiye bağlı gözlem gürültüsünü öğrenir. Toplam varyans yasısıyla:

\[
Var(Y\mid x,D)=E_w[\sigma_w^2(x)]+Var_w[\mu_w(x)]
\]

ayrıştırması yapılır ve posterior predictive senaryolar CVaR / chance-constraint kararlarına bağlanır.

## 5. Bayesian Last Layer / Neural Linear + BoTorch

[`bayesian_last_layer_botorch.ipynb`](./bayesian_last_layer_botorch.ipynb)

Bu notebook **tam BNN değildir**. Gizli katmanlar deterministik olarak özellik öğrenir; yalnız son lineer katmanın ağırlıkları Bayesçi modellenir:

```text
Deney verisi → deterministik backbone → φ(x) → Bayesçi lineer son katman → posterior örnekleri → BoTorch → yeni deney
```

Gaussian prior + Gaussian likelihood altında son katman posterioru kapalı formda hesaplanabildiğinden full BNN'ye göre çok daha hafiftir. Sınırlaması: backbone belirsizliği posteriora taşınmaz.

## 6. NumPyro NUTS vs Pyro Variational Inference

[`numpyro_nuts_vs_pyro_vi_bnn.ipynb`](./numpyro_nuts_vs_pyro_vi_bnn.ipynb)

Aynı küçük BNN mimarisini ve aynı priorları iki farklı posterior çıkarım yöntemiyle karşılaştırır:

```text
aynı veri + aynı BNN
       ↓
 ┌───────────────┬───────────────┐
 │ Pyro SVI / VI │ NumPyro NUTS │
 └───────────────┴───────────────┘
       ↓
posterior predictive
       ↓
RMSE + %90 coverage + interval width
       ↓
beklenen maliyet / CVaR üretim kararı
       ↓
sentetik gerçek dağılımda out-of-sample test
```

Notebookun temel sorusu şudur:

> Daha yaklaşık ama hızlı bir posterior ile daha pahalı MCMC posterioru aynı operasyonel kararı mı veriyor?

Özellikle şu çıktılar karşılaştırılır:

- predictive RMSE,
- %90 predictive interval coverage,
- interval genişliği,
- posterior predictive standart sapma,
- epistemik ortalama fonksiyonunun yayılımı,
- beklenen maliyet için optimum üretim miktarı,
- CVaR95 için optimum üretim miktarı,
- gerçek/sentetik out-of-sample maliyet ve stockout riski,
- hesaplama süresi.

NUTS otomatik olarak "doğru posterior" ilan edilmez. Notebook divergence sayısını ve `print_summary()` çıktılarını kontrol eder. Öğretim süresini sınırlamak için tek chain kullanılır; ciddi bir analizde birden fazla chain ve \(\hat R\) kontrolü gerekir.

## Kurulum

Repo kök dizininde:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

NumPyro notebooku için `requirements.txt` içinde `jax` ve `numpyro` da bulunur.

## Kullanılan kütüphaneler

- `torch`: sinir ağı altyapısı
- `pyro-ppl`: BNN ve variational inference
- `jax`, `numpyro`: NUTS / HMC tabanlı posterior örnekleme
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
6. **Inference hassasiyet analizi:** VI / MCMC seçiminin downstream kararı değiştirip değiştirmediğini ölçme.

## BNN aileleri ve teorik dayanak

Henüz notebooka dönüşmemiş yöntemler için [`BNN_AILELERI.md`](../BNN_AILELERI.md), matematiksel dayanak ve sınırlar için [`TEORIK_DAYANAK.md`](../TEORIK_DAYANAK.md) dosyasına bakın.

## Not

Örnekler öğretim amaçlıdır ve sentetik veri kullanır. Gerçek karar sistemlerinde posterior kalibrasyonu, OOD davranışı, senaryo sayısı duyarlılığı, baseline modeller, MCMC diagnostics, hesaplama bütçesi ve **out-of-sample downstream karar kalitesi** ayrıca doğrulanmalıdır.
