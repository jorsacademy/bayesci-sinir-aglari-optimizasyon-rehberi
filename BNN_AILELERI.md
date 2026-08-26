# Repo Kapsamında BNN Aileleri: Ne Var, Ne Eksik, Ne Zaman Gerekli?

Bu dosya, optimizasyon, endüstri mühendisliği ve yöneylem araştırması açısından BNN ve yakın Bayesian uncertainty yöntemlerini ayırır. Amaç her yöntemi kullanmak değil; hangi yöntemin hangi karar problemi için anlamlı olduğunu göstermektir.

## 1. Tam / katman bazlı BNN + Variational Inference

**Durum:** Repoda var.

Pyro ile ağırlıklara prior konur ve variational posterior öğrenilir. Talep, lead time, proses çıktısı ve posterior scenario generation için temel yöntemdir.

## 2. HMC / NUTS ile BNN

**Durum:** Repoda uygulamalı karşılaştırma var.

Notebook: [`notebooks/numpyro_nuts_vs_pyro_vi_bnn.ipynb`](./notebooks/numpyro_nuts_vs_pyro_vi_bnn.ipynb)

Aynı küçük BNN, Pyro VI ve NumPyro NUTS ile karşılaştırılır. Amaç NUTS'ı otomatik olarak doğru ilan etmek değil; inference seçiminin coverage ve downstream kararı değiştirip değiştirmediğini ölçmektir.

## 3. SGMCMC: SGLD / SGHMC

**Durum:** Henüz uygulamalı örnek yok.

Büyük veri / büyük ağlarda mini-batch MCMC-benzeri posterior sampling için araştırma ağırlıklı seçenektir.

## 4. Bayesian Last Layer / Neural Linear

**Durum:** Repoda uygulamalı notebook var.

Notebook: [`notebooks/bayesian_last_layer_botorch.ipynb`](./notebooks/bayesian_last_layer_botorch.ipynb)

Deterministik backbone özellik çıkarır; yalnız son lineer katman Bayesçidir. Gaussian prior + Gaussian likelihood altında son katman posterioru conjugate biçimde hesaplanabilir.

Avantaj: full BNN'den daha ucuz. Sınırlama: backbone epistemik belirsizliği posteriora girmez.

## 5. Laplace Approximation

**Durum:** Repoda uygulamalı notebook var.

Notebook: [`notebooks/laplace_approximation_pytorch_stokastik_kapasite.ipynb`](./notebooks/laplace_approximation_pytorch_stokastik_kapasite.ipynb)

Teorik not: [`LAPLACE_DAYANAK.md`](./LAPLACE_DAYANAK.md)

Önce deterministik / MAP ağ eğitilir; sonra posterior MAP çevresinde yaklaşık Gaussian alınır:

\[
p(\theta\mid D)
\approx
\mathcal N(\theta_{MAP},P^{-1}).
\]

Repodaki örnek `laplace-torch` ile **last-layer Laplace** uygular ve approximate posterior predictive kapasite senaryolarını Pyomo stokastik kapasite planlamasına bağlar.

Önemli ayrım:

- Bayesian Last Layer: sabit neural features üzerinde Bayesian linear regression.
- Laplace: MAP çevresindeki curvature/Hessian üzerinden yerel Gaussian posterior approximation.

`laplace-torch` ayrıca full-network ve subnetwork Laplace yapılarını da destekler.

Laplace'ın güçlü kullanım senaryosu:

```text
mevcut eğitilmiş PyTorch modeli var
+
full BNN yeniden eğitmek pahalı
+
uncertainty downstream kararda gerekli
```

Sınırlama: posterior multimodal veya MAP çevresi Gaussian'dan uzaksa approximation kalitesi düşebilir.

## 6. Heteroskedastik BNN

**Durum:** Repoda uygulamalı notebook var.

Notebook: [`notebooks/heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb`](./notebooks/heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb)

Model aynı anda \(\mu_w(x)\) ve girdiye bağlı \(\sigma_w(x)\) öğrenir. Aleatorik / epistemik ayrım CVaR ve chance constraints'e bağlanır.

## 7. Multi-output BNN + Multi-objective Optimization

**Durum:** Repoda uygulamalı notebook var.

Notebook: [`notebooks/multi_output_bnn_multi_objective_botorch.ipynb`](./notebooks/multi_output_bnn_multi_objective_botorch.ipynb)

Kalite, enerji ve çevrim süresi ortak Bayesian representation üzerinden modellenir; Pareto, hypervolume ve qLogEHVI ile yeni deney seçilir.

**Multi-output ≠ multi-objective.** İlki probabilistic modelleme, ikincisi karar tercih yapısıdır.

## 8. Hierarchical BNN / Partial Pooling

**Durum:** Repoda uygulamalı rehber var.

Rehber: [`notebooks/hierarchical_bnn_partial_pooling_kapasite_tahsisi.md`](./notebooks/hierarchical_bnn_partial_pooling_kapasite_tahsisi.md)

Ortak nonlinear BNN ile fabrika düzeyi random intercept / slope'lar hierarchical prior altında öğrenilir. Posterior fabrika kapasite senaryoları stochastic allocation modeline aktarılır.

Her hierarchical problem BNN gerektirmez; mixed-effects / hierarchical regression önemli baseline'dır.

## 9. Bayesian RNN / LSTM / Temporal BNN

**Durum:** Henüz yok.

Talep, RUL, enerji yükü ve dinamik sistem durumları için düşünülebilir. State-space ve diğer probabilistic sequence modeller baseline olmalıdır.

## 10. Bayesian GNN

**Durum:** İleri seviye genişletme.

Ulaşım, tedarik zinciri, elektrik şebekesi, routing ve network design problemleri için potansiyel kullanım alanıdır.

## 11. Physics-Informed Bayesian Neural Networks

**Durum:** İleri seviye genişletme.

Fizik denklemleri / mühendislik kısıtları ile Bayesian uncertainty birleştirilir. FEA/CFD, enerji ve proses optimizasyonu için değerlidir.

## 12. Structured / Low-rank / Flow Variational Posterior

**Durum:** İleri araştırma seviyesi.

Mean-field posterior ağırlık korelasyonlarını göz ardı eder. Tail-risk veya chance constraint posterior geometrisine hassassa daha zengin posterior aileleri gerekebilir.

---

# BNN olmayan ama mutlaka karşılaştırılması gereken yöntemler

- Deep Ensemble,
- MC Dropout,
- SWAG,
- Gaussian Process,
- conformal prediction,
- quantile regression,
- mixed-effects / hierarchical regression,
- klasik response-surface ve Bayesian regression.

Bir endüstriyel projede BNN'nin gerçekten değer katıp katmadığı yalnız RMSE ile değil **downstream karar kalitesi** üzerinden ölçülmelidir.

# Matematiksel gerçeklik sınırı

Standart teori olan kısımlar:

- Bayesian posterior / posterior predictive,
- HMC/NUTS,
- variational inference,
- Bayesian linear regression,
- Laplace approximation,
- hierarchical priors / partial pooling,
- heteroskedastik likelihood,
- toplam varyans yasası,
- Pareto dominance / hypervolume,
- CVaR,
- chance constraints,
- SAA,
- Bayesian optimization / multi-objective BO.

Modelleme / approximation tercihi olan kısımlar:

- Normal likelihood,
- `AutoDiagonalNormal`,
- Laplace curvature yapısı,
- prior / hyperprior ölçekleri,
- ağ mimarisi,
- random effect yapısı,
- senaryo / posterior sample sayısı,
- residual covariance varsayımı,
- hypervolume reference point,
- sentetik veri fonksiyonları.

`Matematiksel olarak geçerli` ile `belirli bir fabrikada ampirik olarak doğru` aynı iddia değildir. İkincisi calibration, holdout ve out-of-sample karar testleri gerektirir.

# Repo için sonraki öncelikler

1. **Correlated multi-output likelihood / multi-task BNN**
2. **Constrained multi-objective Bayesian optimization**
3. **Hierarchical BNN için NUTS / richer-VI hassasiyet analizi**
4. **Temporal / Bayesian sequence models**
5. Bayesian GNN
6. Physics-Informed BNN
7. SGMCMC / SGLD / SGHMC

Bu sıra akademik çeşitlilikten çok OR ve endüstri mühendisliği açısından pratik karar değerine göre önerilmiştir.
