# Laplace Approximation: Teorik Dayanak ve Endüstriyel Kullanım

Bu dosya, repodaki `notebooks/laplace_approximation_pytorch_stokastik_kapasite.ipynb` örneğinin teorik sınırlarını ve pratik kullanım amacını açıklar.

## 1. Temel fikir

Deterministik bir sinir ağı eğitildikten sonra MAP noktası çevresinde posterior yaklaşık Gaussian alınır:

\[
p(\theta\mid D)
\approx
\mathcal N
\left(
\theta_{MAP},
P^{-1}
\right).
\]

Posterior precision yaklaşık olarak likelihood curvature ile prior precision toplamından oluşur:

\[
P \approx H_{likelihood}+P_0.
\]

Bu yaklaşım tam posterior değildir. MAP çevresinde ikinci dereceden / yerel Gaussian approximation'dır.

## 2. `laplace-torch`

Ağustos 2026 itibarıyla `laplace-torch` mevcut PyTorch modellerine Laplace approximation uygulamak için aktif bir kütüphanedir.

Desteklenen başlıca seçenekler:

- full-network Laplace,
- subnetwork Laplace,
- last-layer Laplace,
- farklı Hessian / curvature yapıları,
- regression ve classification,
- posterior predictive hesapları,
- marginal-likelihood ile prior precision tuning,
- input'a backpropagation; bu özellik Bayesian optimization gibi uygulamalar için önemlidir.

Kaynaklar:

- Proje: https://github.com/aleximmer/Laplace
- Dokümantasyon: https://aleximmer.github.io/Laplace
- Daxberger et al. (2021), *Laplace Redux — Effortless Bayesian Deep Learning*.

## 3. Last-layer Laplace neden kullanılıyor?

Repodaki örnek, mevcut bir endüstriyel PyTorch modeline düşük maliyetli belirsizlik ekleme senaryosunu hedefler.

Bu nedenle yalnız son katman uncertainty-aware yapılır:

\[
\theta_{last}\mid D
\approx
\mathcal N
\left(
\theta_{last,MAP},
P_{last}^{-1}
\right).
\]

Avantajları:

- posterior boyutu küçüktür,
- curvature hesaplama daha ucuzdur,
- mevcut model mimarisini değiştirmeden uygulanabilir.

Sınırlaması:

- backbone / feature extractor ağırlıklarının epistemik belirsizliği posteriora girmez.

Bu nedenle last-layer Laplace, full BNN ile eşdeğer değildir.

## 4. Bayesian Last Layer ile farkı

Repoda iki ayrı yöntem vardır.

### Bayesian Last Layer / Neural Linear

Sabitlenmiş neural features üzerinde Gaussian prior + Gaussian likelihood kullanıldığında son lineer katmanın posterioru conjugate biçimde kapalı formda hesaplanabilir.

### Laplace Last Layer

Önce deterministik/MAP ağ eğitilir. Daha sonra MAP çevresindeki curvature kullanılarak posterior Gaussian olarak yaklaşık hesaplanır.

Dolayısıyla ikisi benzer bir kullanım profiline sahip olabilir fakat matematiksel olarak aynı inference yöntemi değildir.

## 5. Regression predictive variance

Laplace GLM predictive ile yaklaşık function posterior elde edilir:

\[
f(x)\mid D
\approx
\mathcal N(\mu_f(x),V_f(x)).
\]

Observation-level predictive dağılım için homoskedastik Gaussian noise varsayımında:

\[
Y\mid x,D
\approx
\mathcal N
\left(
\mu_f(x),
V_f(x)+\sigma_\epsilon^2
\right).
\]

Repodaki kapasite senaryoları bu ayrımı açıkça uygular.

## 6. OR / IE bağlantısı

Laplace approximation'ın asıl değeri yalnız uncertainty plot üretmek değildir.

Posterior predictive senaryolar:

\[
C^{(s)}\sim p(C\mid x,D)
\]

stokastik programlamaya aktarılabilir:

\[
\min_x
c(x)+
\frac{1}{S}
\sum_{s=1}^{S}
Q(x,C^{(s)}).
\]

Benzer mimari:

- kapasite planlama,
- bakım planlama,
- stok / üretim,
- kalite riski,
- dijital ikiz,
- surrogate optimization,
- enerji planlama

problemlerine uygulanabilir.

## 7. Hangi durumda tercih edilmemeli?

Laplace approximation yerel Gaussian yaklaşımıdır. Şu durumlarda dikkat gerekir:

- güçlü multimodal posterior,
- çok zayıf tanımlanmış parametreler,
- MAP çevresinin Gaussian geometriye kötü uyması,
- ağır distribution shift / OOD,
- tail-risk kararlarının küçük posterior hatalarına çok hassas olması.

Bu durumlarda HMC/NUTS, VI, Deep Ensemble, GP veya başka uncertainty yöntemleriyle karşılaştırma gerekir.

## 8. Endüstriyel doğrulama

Laplace eklemek otomatik olarak iyi karar üretmez. En az şu ölçümler yapılmalıdır:

- RMSE / MAE,
- predictive interval coverage,
- NLL veya uygun proper scoring rule,
- calibration,
- out-of-sample constraint violation,
- expected operational cost,
- CVaR,
- OOD davranışı,
- deterministic NN / BLL / ensemble / BNN / GP baseline karşılaştırması.

Özetle:

\[
\boxed{
\text{trained PyTorch model}
\rightarrow
\text{Laplace posterior approximation}
\rightarrow
\text{predictive scenarios}
\rightarrow
\text{risk-aware optimization}
}
\]

Bu zincirin ilk iki adımı yaklaşık Bayesçi inference, son iki adımı ise karar bilimi / yöneylem araştırması katmanıdır.
