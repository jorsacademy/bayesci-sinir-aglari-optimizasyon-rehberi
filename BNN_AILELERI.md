# Repo Kapsamında BNN Aileleri: Ne Var, Ne Eksik, Ne Zaman Gerekli?

Bu dosya, optimizasyon, endüstri mühendisliği ve yöneylem araştırması açısından Bayesçi sinir ağı (BNN) ailesini yöntem bazında ayırır. Amaç her yöntemi mutlaka uygulamak değil; hangi belirsizlik yaklaşımının hangi karar problemine uygun olduğunu görmek.

## 1. Tam / katman bazlı BNN + Variational Inference

**Durum:** Repoda var.

Pyro ile ağırlıklara prior konur ve `AutoDiagonalNormal` ile yaklaşık posterior öğrenilir.

Uygun kullanım:

- talep / lead-time belirsizliği,
- posterior senaryo üretimi,
- stochastic programming,
- CVaR / chance constraints,
- BNN surrogate ile Bayesçi optimizasyon.

## 2. HMC / NUTS ile BNN

**Durum:** Repoda uygulamalı karşılaştırma var.

Notebook: [`notebooks/numpyro_nuts_vs_pyro_vi_bnn.ipynb`](./notebooks/numpyro_nuts_vs_pyro_vi_bnn.ipynb)

Aynı küçük BNN, Pyro VI ve NumPyro NUTS ile karşılaştırılır. Amaç NUTS'ı otomatik olarak “doğru” ilan etmek değil; posterior yaklaşım yönteminin predictive coverage ve downstream üretim kararını değiştirip değiştirmediğini ölçmektir.

Küçük/orta ağlarda güçlü benchmark; büyük ağlarda maliyetlidir.

## 3. SGMCMC: SGLD / SGHMC

**Durum:** Henüz uygulamalı notebook yok.

Mini-batch MCMC yaklaşımıyla büyük veri ve daha büyük ağlara ölçeklenmeye çalışır.

OR açısından potansiyel kullanım:

- büyük üretim/sensör veri setleri,
- yüksek boyutlu surrogate modeller,
- posterior sample tabanlı scenario generation.

## 4. Bayesian Last Layer / Neural Linear

**Durum:** Repoda uygulamalı notebook var.

Notebook: [`notebooks/bayesian_last_layer_botorch.ipynb`](./notebooks/bayesian_last_layer_botorch.ipynb)

Deterministik backbone özellik çıkarır; yalnız son katman Bayesçidir:

\[
f(x)=\tilde\phi(x)^\top\beta,
\qquad
\beta\mid D\sim\mathcal N(m,\Sigma).
\]

Avantajları:

- full BNN'den daha ucuz,
- posterior güncellemesi hızlı,
- BoTorch entegrasyonu pratik,
- online/sequential optimization için uygun.

Sınırlama: backbone epistemik belirsizliği posteriora girmez.

## 5. Laplace Approximation BNN

**Durum:** Henüz uygulamalı notebook yok.

Önce MAP/deterministik ağ eğitilir; optimum çevresinde ağırlık posterioru yaklaşık Gaussian alınır.

Mevcut endüstriyel PyTorch modelini sonradan uncertainty-aware hale getirmek için pratik bir yöntem olabilir.

## 6. Heteroskedastik BNN

**Durum:** Repoda uygulamalı notebook var.

Notebook: [`notebooks/heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb`](./notebooks/heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb)

Model aynı anda

\[
\mu_w(x),\qquad \sigma_w(x)
\]

öğrenir ve toplam varyans yasasıyla

\[
Var(Y\mid x,D)=E_w[\sigma_w^2(x)]+Var_w[\mu_w(x)]
\]

ayrıştırmasını kullanır.

Özellikle girdiye bağlı talep, işlem süresi, taşıma süresi ve kalite oynaklığı için değerlidir.

## 7. Multi-output BNN + Multi-objective Optimization

**Durum:** Repoda uygulamalı notebook var.

Notebook: [`notebooks/multi_output_bnn_multi_objective_botorch.ipynb`](./notebooks/multi_output_bnn_multi_objective_botorch.ipynb)

Örnek aynı proses kararının üç çıktısını ortak Bayesçi hidden representation ile modeller:

- kalite,
- enerji,
- çevrim süresi.

Ardından objective yönleri

\[
(\text{quality},-\text{energy},-\text{cycle})
\]

şeklinde maksimize edilecek uzaya çevrilir ve BoTorch qLogEHVI ile Pareto/hypervolume iyileştiren yeni deney seçilir.

### Kritik ayrım

**Multi-output ≠ multi-objective.**

- Multi-output BNN: birden fazla rassal çıktıyı modeller.
- Multi-objective optimization: karar vericinin bu çıktılar arasındaki trade-off'u nasıl ele aldığını belirler.

Notebook ortak Bayesçi backbone kullanır, fakat residual likelihood diagonal covariance varsayar. Dolayısıyla tam multivariate residual covariance modeli değildir.

## 8. Hierarchical BNN / Partial Pooling

**Durum:** Repoda uygulamalı rehber var.

Rehber: [`notebooks/hierarchical_bnn_partial_pooling_kapasite_tahsisi.md`](./notebooks/hierarchical_bnn_partial_pooling_kapasite_tahsisi.md)

Dört fabrikanın aynı temel nonlinear proses ilişkisini paylaştığı fakat lokal sapmalarının ve veri miktarlarının farklı olduğu bir örnek kurulur.

Ortak BNN:

\[
g_w(x)=W_2\tanh(W_1x+b_1)+b_2.
\]

Fabrika düzeyi random effect'ler:

\[
a_j\sim\mathcal N(\mu_a,\tau_a^2),
\]

\[
b_j\sim\mathcal N(\mu_b,\tau_b^2).
\]

Tahmin:

\[
\mu_{ij}=g_w(x_{ij})+a_j+b_j\widetilde{load}_{ij}.
\]

Bu yapı **partial pooling** üretir: fabrikalar tamamen ayrı değildir fakat aynı davranışa da zorlanmaz. Az verili fabrikanın posterioru ortak grup bilgisiyle düzenlenebilir ve genellikle daha geniş belirsizlik taşır.

Posterior predictive fabrika kapasite senaryoları daha sonra Pyomo ile stokastik kapasite tahsisi problemine aktarılır.

IE açısından doğal kullanım alanları:

- çok fabrika kapasite / verim modeli,
- makine grupları için işlem süresi,
- tedarikçi grupları için lead time / kalite,
- ürün aileleri için talep,
- bölgesel talep ve network allocation.

### Kritik sınır

**Her hiyerarşik problem BNN gerektirmez.**

Eğer ilişki basitse hierarchical linear model / GLMM / mixed-effects model daha yorumlanabilir ve daha ucuz olabilir. BNN, nonlinear ortak response surface gerçekten gerekliyse anlamlıdır.

Ayrıca mean-field VI hiyerarşik posterior korelasyonlarını zayıf temsil edebilir; HMC/NUTS veya daha zengin variational posteriorlarla hassasiyet analizi yapılabilir.

## 9. Bayesian RNN / LSTM / temporal BNN

**Durum:** Henüz yok.

Zaman bağımlı süreçlerde:

- talep,
- remaining useful life,
- enerji yükü,
- dinamik üretim durumu

için düşünülebilir.

Pratikte state-space modeller ve diğer probabilistic sequence modeller de baseline olmalıdır.

## 10. Bayesian GNN

**Durum:** İleri seviye genişletme.

OR bağlantıları:

- ulaşım ağları,
- tedarik zinciri ağları,
- elektrik şebekeleri,
- routing / network design surrogate'ları.

## 11. Physics-Informed Bayesian Neural Networks

**Durum:** İleri seviye genişletme.

Fiziksel denklemler veya mühendislik kısıtları likelihood/loss yapısına dahil edilir; BNN belirsizliği taşır.

Uygulamalar:

- enerji sistemleri,
- akış / ısı transferi,
- yapısal tasarım,
- proses mühendisliği,
- FEA / CFD surrogate optimizasyonu.

## 12. Structured / low-rank / flow variational posterior

**Durum:** İleri araştırma seviyesi.

Mean-field posterior ağırlık korelasyonlarını göz ardı eder. Daha zengin posterior aileleri:

- low-rank Gaussian,
- full-rank Gaussian,
- normalizing flows,
- structured variational inference.

Tail-risk, CVaR veya chance constraints posterior geometrisine hassassa önemli olabilir.

---

# BNN olmayan ama mutlaka karşılaştırılması gereken yöntemler

- Deep Ensemble,
- MC Dropout,
- SWAG,
- Gaussian Process,
- conformal prediction,
- quantile regression,
- hierarchical linear / mixed-effects modeller,
- klasik response-surface / Bayesian regression modelleri.

Bir endüstriyel projede BNN'nin gerçekten değer katıp katmadığı yalnız RMSE ile değil **downstream karar kalitesi** üzerinden ölçülmelidir.

# Matematiksel gerçeklik sınırı

Standart teori olan kısımlar:

- Bayesçi posterior ve posterior predictive,
- HMC/NUTS,
- Bayesçi lineer regresyon posterioru,
- hierarchical priors / partial pooling,
- random effects,
- heteroskedastik likelihood,
- toplam varyans yasası,
- Pareto dominance,
- hypervolume,
- CVaR,
- chance constraints,
- SAA,
- Bayesçi optimizasyon / multi-objective BO.

Modelleme veya yaklaşık çıkarım tercihi olan kısımlar:

- Normal likelihood,
- `AutoDiagonalNormal`,
- ağ mimarisi,
- prior/hyperprior ölçekleri,
- random intercept/slope yapısı,
- senaryo/posterior sample sayısı,
- residual covariance'in diagonal seçilmesi,
- hypervolume reference point,
- sentetik veri fonksiyonları.

`Matematiksel olarak geçerli` ile `belirli bir fabrikada ampirik olarak doğru` aynı iddia değildir. İkincisi calibration, holdout ve out-of-sample karar testleri gerektirir.

# Repo için sonraki öncelikler

1. **Laplace Approximation + mevcut PyTorch modelini uncertainty-aware hale getirme**
2. **Correlated multi-output likelihood / multi-task BNN**
3. **Constrained multi-objective BO**
4. **Hierarchical BNN için NUTS / richer-VI hassasiyet analizi**
5. Bayesian RNN / temporal BNN
6. Bayesian GNN
7. Physics-Informed BNN
8. SGMCMC / SGLD / SGHMC

Bu sıra akademik çeşitlilikten çok OR ve endüstri mühendisliği açısından pratik karar değerine göre önerilmiştir.
