# Repo Kapsamında BNN Aileleri: Ne Eksik, Ne Zaman Gerekli?

Bu dosya, optimizasyon, endüstri mühendisliği ve yöneylem araştırması açısından Bayesçi sinir ağı (BNN) ailesini yöntem bazında ayırır. Amaç her yöntemi mutlaka uygulamak değil; hangi belirsizlik yaklaşımının hangi karar problemine uygun olduğunu görmek.

## 1. Tam / katman bazlı BNN + Variational Inference

**Durum:** Repoda uygulamalı olarak var.

Ağırlıklara önsel dağılım konur ve yaklaşık posterior çoğunlukla mean-field variational inference ile öğrenilir.

Uygun kullanım:

- talep ve lead-time belirsizliği,
- proses response surface,
- kestirimci bakım,
- posterior senaryo üretimi,
- BNN surrogate ile Bayesçi optimizasyon.

Araçlar: Pyro, NumPyro, PyMC.

## 2. HMC / NUTS ile BNN

**Durum:** Repoda uygulamalı notebook var.

Notebook: [`notebooks/numpyro_nuts_vs_pyro_vi_bnn.ipynb`](./notebooks/numpyro_nuts_vs_pyro_vi_bnn.ipynb)

Variational inference yerine Hamiltonian Monte Carlo ailesinden NUTS ile ağırlık posterioru örneklenir.

Repo örneğinde aynı küçük BNN:

- aynı veri,
- aynı prior,
- aynı likelihood,
- aynı gizli katman boyutu

altında Pyro mean-field VI ve NumPyro NUTS ile karşılaştırılır.

OR açısından asıl ölçüm yalnız posterior benzerliği değildir. Notebook şu zinciri karşılaştırır:

\[
\text{inference}
\rightarrow
\text{predictive coverage}
\rightarrow
\text{senaryolar}
\rightarrow
\text{üretim kararı}
\rightarrow
\text{out-of-sample maliyet / CVaR}.
\]

### NUTS neden faydalı?

Küçük ve orta ölçekli modellerde VI için güçlü bir posterior benchmark'ı olabilir. Mean-field VI'ın kaçırabileceği posterior korelasyonlarını ve karmaşık geometrileri daha iyi temsil etme potansiyeli vardır.

### NUTS neden her problem için çözüm değildir?

- ağ büyüdükçe maliyet hızla artar,
- MCMC diagnostics gerekir,
- divergence gözlenebilir,
- birden fazla chain ve \(\hat R\) kontrolü gerekir,
- sonlu örneklem nedeniyle NUTS da "tam posteriorun kendisi" değildir.

NumPyro özellikle JAX tabanlı NUTS/HMC için güçlü bir seçenektir.

## 3. SGMCMC: SGLD / SGHMC tabanlı BNN

**Durum:** Henüz uygulamalı örnek yok.

Büyük veri ve daha büyük ağlarda mini-batch MCMC yaklaşımı sunar. Tam HMC/NUTS'tan daha ölçeklenebilir olabilir.

OR açısından potansiyel kullanım:

- büyük üretim/sensör veri setleri,
- yüksek boyutlu surrogate modeller,
- posterior örneklerinin senaryo üretiminde kullanılması.

## 4. Bayesian Last Layer / Neural Linear Model

**Durum:** Repoda uygulamalı notebook var.

Notebook: [`notebooks/bayesian_last_layer_botorch.ipynb`](./notebooks/bayesian_last_layer_botorch.ipynb)

Sinir ağının özellik çıkarıcı kısmı deterministik tutulur; yalnız son katman Bayesçi modellenir:

\[
\phi(x)=\text{deterministik backbone çıktısı},
\]

\[
f(x)=\tilde\phi(x)^\top\beta,
\qquad
\beta\sim\mathcal N(0,\alpha^{-1}I).
\]

Gaussian likelihood altında son katman posterioru kapalı formda hesaplanabilir:

\[
\Sigma=(\alpha I+\tau\Phi^\top\Phi)^{-1},
\]

\[
m=\tau\Sigma\Phi^\top y.
\]

Avantajları:

- tam BNN'den daha ucuz,
- posterior güncellemesi hızlı,
- BoTorch entegrasyonu pratik,
- online/sequential optimization için uygun.

Sınırlaması: backbone ağırlıkları nokta tahminidir. BLL full BNN değildir ve OOD bölgelerinde fazla güvenli olabilir.

## 5. Laplace Approximation BNN

**Durum:** Henüz uygulamalı örnek yok.

Önce MAP / deterministik ağ eğitilir; sonra optimum çevresinde ağırlık posterioru yaklaşık Gaussian olarak modellenir.

Özellikle mevcut PyTorch modelini sonradan uncertainty-aware hale getirmek için değerlidir.

## 6. Heteroskedastik BNN

**Durum:** Repoda uygulamalı notebook var.

Notebook: [`notebooks/heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb`](./notebooks/heteroskedastik_bnn_belirsizlik_cvar_chance.ipynb)

Yalnız ortalama değil, girdiye bağlı gözlem varyansı da modellenir:

\[
y\mid x,w \sim \mathcal N(\mu_w(x),\sigma_w^2(x)).
\]

Toplam varyans yasasıyla:

\[
Var(Y\mid x,\mathcal D)
=
E_w[\sigma_w^2(x)]
+
Var_w[\mu_w(x)]
\]

ayrıştırması yapılır ve posterior predictive senaryolar CVaR/chance-constrained üretim modeline aktarılır.

## 7. Multi-output / Multi-task BNN

**Durum:** Henüz uygulamalı örnek yok.

Aynı karar noktasında birden fazla ilişkili çıktı birlikte modellenir.

Örnek:

- kalite + enerji + çevrim süresi,
- talep + fiyat + iade oranı,
- arıza riski + bakım süresi + üretim kaybı.

Multi-objective optimization ve joint chance constraints için değerlidir.

## 8. Hierarchical BNN

**Durum:** Henüz uygulamalı örnek yok.

Fabrika, makine, ürün, tedarikçi veya bölge düzeyinde ortak ve yerel parametreleri hiyerarşik önsellerle birlikte öğrenir.

Örnek:

- çok fabrikalı üretim sistemi,
- tedarikçiler arası lead-time belirsizliği,
- ürün aileleri arasında partial pooling.

Endüstri mühendisliği açısından oldukça doğal bir BNN türüdür.

## 9. Bayesian RNN / LSTM / zaman serisi BNN

**Durum:** Henüz yok.

Uygun kullanım:

- talep tahmini,
- remaining useful life,
- enerji yükü,
- dinamik üretim sistemi durumları.

Ancak state-space modelleri ve diğer probabilistic temporal modeller mutlaka baseline olmalıdır.

## 10. Bayesian GNN

**Durum:** İleri seviye genişletme.

OR bağlantıları:

- ulaşım ağları,
- tedarik zinciri ağları,
- elektrik şebekeleri,
- facility/network design,
- routing surrogate modelleri.

## 11. Physics-Informed Bayesian Neural Networks

**Durum:** İleri seviye genişletme.

Fiziksel denklemler veya mühendislik kısıtları model yapısına dahil edilir; BNN belirsizliği de taşır.

Uygulamalar:

- enerji sistemleri,
- akış/ısı transferi,
- yapısal tasarım,
- proses mühendisliği,
- pahalı FEA/CFD surrogate modelleri.

## 12. Structured / low-rank / flow variational posterior

**Durum:** İleri araştırma seviyesi.

Mean-field `AutoDiagonalNormal`, ağırlık korelasyonlarını göz ardı eder. Alternatifler:

- low-rank Gaussian,
- full-rank Gaussian,
- normalizing flows,
- structured variational inference.

Tail-risk'e hassas karar problemlerinde posterior ailesinin kalitesi CVaR/chance-constraint sonuçlarını değiştirebilir.

---

# BNN olmayan ama mutlaka karşılaştırılması gereken yöntemler

Strict anlamda tam BNN olmayan fakat uncertainty benchmark'ı olarak önemli yöntemler:

- Deep Ensemble,
- MC Dropout,
- SWAG,
- Gaussian Process,
- conformal prediction,
- quantile regression.

Özellikle endüstriyel projede BNN'nin gerçekten değer katıp katmadığı bu baseline'lara karşı **downstream karar kalitesi** üzerinden ölçülmelidir.

# Matematiksel gerçeklik sınırı

Standart teori olan kısımlar:

- Bayesçi posterior / posterior predictive,
- variational inference,
- HMC / NUTS,
- Bayesçi lineer regresyonun Gaussian kapalı-form posterioru,
- heteroskedastik likelihood,
- toplam varyans yasası,
- CVaR,
- chance constraints,
- stochastic programming / SAA,
- Bayesçi optimizasyon.

Modelleme / yaklaşık çıkarım tercihi olan kısımlar:

- Normal likelihood seçimi,
- `AutoDiagonalNormal` mean-field posterioru,
- NUTS warmup/sample/chain sayısı,
- prior ölçekleri,
- ağ mimarisi,
- sentetik veri fonksiyonları,
- senaryo sayısı.

Bu nedenle `matematiksel olarak geçerli` ile `belirli bir fabrikada ampirik olarak doğru` aynı iddia değildir. İkincisi kalibrasyon, MCMC diagnostics ve out-of-sample karar testi gerektirir.

# Repo için önerilen sonraki öncelikler

Heteroskedastik BNN, Bayesian Last Layer ve NUTS-vs-VI karşılaştırması artık eklendi. Bundan sonraki notebook sırası:

1. **Multi-output BNN + multi-objective optimization**
2. **Hierarchical BNN + çok fabrika / çok tedarikçi problemi**
3. **Laplace Approximation + mevcut PyTorch modelini uncertainty-aware hale getirme**
4. **Bayesian RNN / temporal BNN**
5. **Bayesian GNN**
6. **Physics-Informed BNN**
7. **SGMCMC / SGLD / SGHMC**
8. **Structured / flow variational posterior karşılaştırması**

Bu sıra akademik çeşitlilikten çok OR ve endüstri mühendisliği açısından pratik karar değerine göre önerilmiştir.
