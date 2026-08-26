# Teorik Dayanak: Bu Repodaki Yöntemler Nereden Geliyor?

Bu dosyanın amacı, repodaki yöntemlerin hangi kısımlarının standart olasılık / yöneylem araştırması teorisine dayandığını ve hangi kısımların pratik modelleme tercihi olduğunu açıkça ayırmaktır.

## 1. Bayesçi sinir ağı ve posterior predictive

Bayesçi sinir ağında ağırlıklar tek bir nokta tahmini yerine rassal değişkenler olarak modellenir:

\[
w\sim p(w), \qquad
p(w\mid\mathcal D) \propto p(\mathcal D\mid w)p(w).
\]

Yeni bir girdi için posterior predictive:

\[
p(y^*\mid x^*,\mathcal D)
=
\int
p(y^*\mid x^*,w)p(w\mid\mathcal D)\,dw.
\]

Pyro resmi dokümantasyonu:

- https://docs.pyro.ai/en/stable/contrib.bnn.html
- https://docs.pyro.ai/en/stable/nn.html

## 2. Variational Inference: ne yapıyoruz?

Variational inference, gerçek posterioru doğrudan örneklemek yerine daha basit bir dağılım ailesinden bir yaklaşım seçer:

\[
q_\phi(w) \approx p(w\mid D).
\]

Tipik olarak \(q_\phi\), KL divergence veya eşdeğer ELBO hedefi üzerinden optimize edilir:

\[
\phi^*=
\arg\min_\phi
KL\left(q_\phi(w)\,\|\,p(w\mid D)\right).
\]

Repoda `AutoDiagonalNormal`, latent uzayda diagonal Gaussian yaklaşımı kullanır. Bu hızlıdır; fakat ağırlıklar arasındaki posterior korelasyonlarını doğrudan temsil etmez.

Kaynaklar:

- Blei, Kucukelbir & McAuliffe (2017), *Variational Inference: A Review for Statisticians*:
  https://arxiv.org/abs/1601.00670
- Pyro AutoGuide / AutoDiagonalNormal:
  https://docs.pyro.ai/en/stable/infer.autoguide.html

Önemli nokta: "VI posterioru dar olur" evrensel bir teorem değildir. Mean-field ve reverse-KL tabanlı VI bazı modellerde varyansı düşük tahmin edebilir; bunun belirli bir BNN'de olup olmadığı predictive coverage ve downstream karar testiyle ölçülmelidir.

## 3. HMC / NUTS: ne farkı var?

Hamiltonian Monte Carlo, posterior uzayında gradient bilgisini kullanarak MCMC örnekleri üretir. No-U-Turn Sampler (NUTS), HMC trajectory length seçimini otomatikleştiren adaptif bir yöntemdir.

NUTS posterioru bir variational aileyle sınırlamaz. Ancak bu, NUTS çıktısının otomatik olarak "gerçek posterior" olduğu anlamına gelmez. Sonlu MCMC koşusu için:

- warmup,
- divergence,
- effective sample size,
- birden fazla chain,
- \(\hat R\)

gibi diagnostics kontrol edilmelidir.

Kaynaklar:

- Hoffman & Gelman (2014), *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*:
  https://jmlr.org/papers/v15/hoffman14a.html
- NumPyro MCMC / NUTS resmi dokümantasyonu:
  https://num.pyro.ai/en/stable/mcmc.html
- NumPyro güncel örneklerinde `NUTS`, `MCMC`, `get_samples()` ve posterior predictive kullanımı:
  https://num.pyro.ai/en/latest/tutorials/effect_handlers.html

## 4. Neden VI vs NUTS'u karar kalitesiyle karşılaştırıyoruz?

OR açısından çıkarım yönteminin değeri yalnız posterior parametrelerinde değil, aşağı akış kararında ortaya çıkar.

Repo notebooku şu zinciri kullanır:

\[
\text{inference yöntemi}
\rightarrow
p(y\mid x,D)
\rightarrow
\text{senaryolar}
\rightarrow
\text{risk ölçüsü}
\rightarrow
x^*.
\]

Aynı tahmin ortalamasına sahip iki posterior farklı tail davranışı veya interval genişliği üretebilir. Bu durumda:

- CVaR,
- chance constraint,
- safety stock,
- kapasite,
- exploration/exploitation

kararları farklılaşabilir.

Bu nedenle `numpyro_nuts_vs_pyro_vi_bnn.ipynb` şu metrikleri birlikte inceler:

- RMSE,
- predictive interval coverage,
- interval width,
- posterior predictive yayılım,
- beklenen maliyet optimumu,
- CVaR optimumu,
- out-of-sample gerçek maliyet / stockout riski,
- hesaplama süresi.

## 5. Heteroskedastik aleatorik belirsizlik

Heteroskedastik regresyonda gözlem varyansı girdiye bağlıdır:

\[
Y\mid x,w \sim \mathcal N(\mu_w(x),\sigma_w^2(x)).
\]

Bu yaklaşım yeni veya spekülatif değildir.

Kaynaklar:

- Kendall & Gal (2017), *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*
  https://arxiv.org/abs/1703.04977
- Depeweg et al., aleatorik / epistemik ayrıştırma ve risk-duyarlı karar verme:
  https://arxiv.org/abs/1710.07283

## 6. Aleatorik / epistemik varyans ayrıştırması

Toplam varyans yasası:

\[
Var(Y\mid x,\mathcal D)
=
E_{w\mid\mathcal D}[Var(Y\mid x,w)]
+
Var_{w\mid\mathcal D}(E[Y\mid x,w]).
\]

Gaussian heteroskedastik likelihood altında:

\[
Var(Y\mid x,\mathcal D)
=
E_w[\sigma_w^2(x)]
+
Var_w[\mu_w(x)].
\]

Bu ayrıştırma Monte Carlo posterior örnekleriyle yaklaşık hesaplanır.

## 7. Chance constraints

Chance constraint klasik OR problemidir:

\[
P(g(x,\xi)\le 0)\ge 1-\alpha.
\]

Kaynaklar:

- Stanford EE364A:
  https://stanford.edu/class/ee364a/lectures/chance_constr.pdf
- Campi & Garatti (2008):
  https://epubs.siam.org/doi/10.1137/07069821X
- Campi & Garatti (2011):
  https://doi.org/10.1007/s10957-010-9754-6

Sonlu posterior senaryoda %95 ampirik feasibility görmek, tek başına gerçek dağılım altında %95 garanti değildir. Scenario-approach varsayımları veya bağımsız out-of-sample doğrulama gerekir.

## 8. CVaR ve Rockafellar–Uryasev reformülasyonu

\[
CVaR_\alpha(L)
=
\min_\eta
\left[
\eta+
\frac{1}{1-\alpha}
E[(L-\eta)^+]
\right].
\]

Kaynak:

- Rockafellar & Uryasev (2000), *Optimization of Conditional Value-at-Risk*:
  https://doi.org/10.21314/JOR.2000.038

## 9. SAA: Sample Average Approximation

\[
\min_x E[Q(x,\xi)]
\]

problemi örneklenen senaryolarla:

\[
\min_x \frac{1}{S}\sum_{s=1}^S Q(x,\xi_s)
\]

şeklinde yaklaşık çözülebilir. Repoda BNN posterior predictive dağılımı bu senaryoların kaynağıdır.

## 10. Bayesian Last Layer / Neural Linear Model

Deterministik backbone:

\[
\phi(x)=h_\theta(x),
\]

Bayesçi son katman:

\[
f(x)=\tilde\phi(x)^\top\beta,
\qquad
\beta\sim\mathcal N(0,\alpha^{-1}I).
\]

Gaussian likelihood altında:

\[
\Sigma=(\alpha I+\tau\Phi^\top\Phi)^{-1},
\]

\[
m=\tau\Sigma\Phi^\top y,
\]

\[
\beta\mid D\sim\mathcal N(m,\Sigma).
\]

Bu klasik Bayesçi lineer regresyon sonucudur. Fakat posterior yalnız sabitlenmiş özellikler üzerindeki son katman içindir; backbone belirsizliğini içermez.

Örnek literatür:

- Fiedler & Lucia (2023): https://arxiv.org/abs/2302.10975
- Moberg et al. (2019): https://arxiv.org/abs/1912.06760
- Zahavy & Mannor (2019): https://arxiv.org/abs/1901.08612

## 11. BNN / BLL surrogate'ın BoTorch ile kullanılması

BoTorch Monte Carlo acquisition fonksiyonları surrogate modelin GP olmasını zorunlu kılmaz. Örneklenebilir bir posterior ve gradient akışı sağlayan custom modeller kullanılabilir.

Kaynak:

- https://botorch.org/docs/models

## 12. Ne standart teori değildir?

Aşağıdakiler teorem değil, modelleme / mühendislik tercihidir:

- Normal likelihood seçimi,
- gizli katman boyutu,
- prior ölçekleri,
- `AutoDiagonalNormal` seçimi,
- NUTS için 700 warmup / 1000 posterior sample kullanılması,
- tek chain ile öğretim koşusu,
- 500 senaryo veya 256 posterior örneği kullanılması,
- sentetik black-box / talep fonksiyonları,
- acquisition restart sayıları.

Bunlar problemden probleme doğrulanmalıdır.

## 13. Endüstriyel doğrulama için minimum kontrol listesi

Bir BNN/BLL + optimization çalışmasının gerçek uygulama iddiası taşıması için en az:

1. train/validation/test veya temporal holdout,
2. predictive interval coverage,
3. NLL veya proper scoring rule,
4. calibration kontrolü,
5. deterministic NN / Deep Ensemble / GP / klasik istatistiksel baseline,
6. senaryo ve posterior sample duyarlılığı,
7. out-of-sample constraint violation rate,
8. beklenen maliyet ve CVaR,
9. distribution shift / OOD testi,
10. inference yöntemi hassasiyeti,
11. MCMC divergence / ESS / \(\hat R\) diagnostics,
12. optimizasyon regret / best-so-far / sample efficiency,
13. hesaplama süresi ve operasyonel güncelleme maliyeti

raporlanmalıdır.

Kısacası: repodaki temel matematiksel yapılar literatürde yerleşik yapılardır; fakat belirli bir endüstriyel sistemde işe yaradıkları **veri ve downstream karar performansıyla ayrıca kanıtlanmalıdır**.
