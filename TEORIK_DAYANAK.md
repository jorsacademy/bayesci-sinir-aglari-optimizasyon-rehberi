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

## 2. Heteroskedastik aleatorik belirsizlik

Heteroskedastik regresyonda gözlem varyansı girdiye bağlıdır:

\[
Y\mid x,w \sim \mathcal N(\mu_w(x),\sigma_w^2(x)).
\]

Bu yaklaşım yeni veya spekülatif değildir. Kendall ve Gal (2017), girdiye bağlı aleatorik belirsizliği ve epistemik belirsizliği birlikte ele alan Bayesian deep learning çerçevesini açıkça formüle eder.

Kaynaklar:

- Kendall, A. & Gal, Y. (2017), *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*
  https://arxiv.org/abs/1703.04977
- Depeweg et al., aleatorik / epistemik ayrıştırma ve risk-duyarlı karar verme:
  https://arxiv.org/abs/1710.07283

## 3. Aleatorik / epistemik varyans ayrıştırması

Notebookta kullanılan temel eşitlik toplam varyans yasasıdır:

\[
Var(Y\mid x,\mathcal D)
=
E_{w\mid\mathcal D}[Var(Y\mid x,w)]
+
Var_{w\mid\mathcal D}(E[Y\mid x,w]).
\]

Gaussian heteroskedastik likelihood altında:

\[
E[Y\mid x,w]=\mu_w(x),
\qquad
Var(Y\mid x,w)=\sigma_w^2(x),
\]

ve dolayısıyla:

\[
Var(Y\mid x,\mathcal D)
=
E_w[\sigma_w^2(x)]
+
Var_w[\mu_w(x)].
\]

Notebook bu iki terimi posterior Monte Carlo örnekleriyle yaklaşıklar.

Önemli nüans: `sigma(x)` fonksiyonunun parametreleri de Bayesçiyse, `E_w[\sigma_w^2(x)]` posterior boyunca beklenen koşullu veri varyansıdır. `sigma(x)` fonksiyonunun kendisi üzerindeki parametre belirsizliği ayrıca üçüncü bir terim halinde raporlanmıyor.

## 4. Chance constraints

Chance constraint klasik OR problemidir:

\[
P(g(x,\xi)\le 0)\ge 1-\alpha.
\]

Modern scenario approach literatürü, örneklenmiş kısıtlarla çözülen problemler için hangi koşullarda olasılıksal feasibility garantileri elde edilebileceğini inceler.

Kaynaklar:

- Stanford EE364A chance-constrained optimization:
  https://stanford.edu/class/ee364a/lectures/chance_constr.pdf
- Campi & Garatti (2008), *The Exact Feasibility of Randomized Solutions of Uncertain Convex Programs*:
  https://epubs.siam.org/doi/10.1137/07069821X
- Campi & Garatti (2011), sampling-and-discarding:
  https://doi.org/10.1007/s10957-010-9754-6

Bu nedenle repodaki notebook bilinçli olarak şu uyarıyı yapar:

> Sonlu posterior senaryoda %95 ampirik feasibility görmek, tek başına gerçek dağılım altında %95 güvence kanıtı değildir.

Gerçek garanti için scenario-approach varsayımları, örneklem büyüklüğü teorisi veya bağımsız out-of-sample doğrulama gerekir.

## 5. CVaR ve Rockafellar–Uryasev reformülasyonu

Kayıp rassal değişkeni \(L\) için CVaR'ın klasik optimizasyon gösterimi:

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

Bu, repoda Pyomo modeline doğrudan çevrilen formüldür.

Kaynak:

- Rockafellar, R. T. & Uryasev, S. (2000), *Optimization of Conditional Value-at-Risk*, Journal of Risk, 2(3), 21–41.
  https://doi.org/10.21314/JOR.2000.038

## 6. SAA: Sample Average Approximation

Stokastik programlamada:

\[
\min_x E[Q(x,\xi)]
\]

tipi bir problem, örneklenen \(S\) senaryoyla:

\[
\min_x \frac{1}{S}\sum_{s=1}^S Q(x,\xi_s)
\]

şeklinde yaklaşık çözülebilir.

Repodaki BNN'nin rolü, belirsiz girdiler için posterior predictive senaryolar üretmektir.

## 7. Bayesian Last Layer / Neural Linear Model

Bayesian Last Layer (BLL) notebookunda tüm ağ Bayesçi değildir. Deterministik bir backbone önce özellik çıkarır:

\[
\phi(x)=h_\theta(x),
\]

sonra yalnız son katman Bayesçi lineer modeldir:

\[
f(x)=\tilde\phi(x)^\top\beta,
\qquad
\beta\sim\mathcal N(0,\alpha^{-1}I).
\]

Gaussian likelihood:

\[
y\mid\beta\sim\mathcal N(\Phi\beta,\sigma_\epsilon^2 I)
\]

altında posterior yine Gaussian'dır. \(\tau=1/\sigma_\epsilon^2\) için:

\[
\Sigma=(\alpha I+\tau\Phi^\top\Phi)^{-1},
\]

\[
m=\tau\Sigma\Phi^\top y,
\]

\[
\beta\mid D\sim\mathcal N(m,\Sigma).
\]

Bu formüller klasik Bayesçi lineer regresyon sonucudur; spekülatif bir neural-network formülü değildir. Neural-linear/BLL yaklaşımı, sinir ağının temsil öğrenme kapasitesini lineer Bayesçi son katmanın ucuz posterior güncellemesiyle birleştirir.

Örnek literatür:

- Fiedler & Lucia (2023), *Improved uncertainty quantification for neural networks with Bayesian last layer*:
  https://arxiv.org/abs/2302.10975
- Moberg et al. (2019), *Bayesian Linear Regression on Deep Representations*:
  https://arxiv.org/abs/1912.06760
- Zahavy & Mannor (2019), *Deep Neural Linear Bandits*:
  https://arxiv.org/abs/1901.08612

### BLL'nin teorik sınırı

Kapalı-form posterior **yalnız sabitlenmiş özellikler üzerindeki lineer son katman** için geçerlidir. Backbone parametreleri deterministik eğitildiyse onların belirsizliği bu posteriora girmez.

Dolayısıyla:

> `Bayesian Last Layer posterioru tam olarak hesaplandı` ifadesi, `tüm sinir ağının Bayesçi posterioru tam olarak hesaplandı` anlamına gelmez.

Bu ayrım özellikle OOD / extrapolation problemlerinde önemlidir.

## 8. BLL / BNN surrogate'ın BoTorch ile kullanılması

BoTorch v0.18.1 dokümantasyonuna göre Monte Carlo acquisition fonksiyonları surrogate modelin GP olmasını zorunlu kılmaz. Custom modelin temel gereksinimi `posterior()` döndürmesi ve posteriorun `rsample()` ile örneklenebilir olmasıdır. Gradient-based acquisition optimization kullanılacaksa bu örneklerden model girdisine backpropagation yapılabilmelidir.

Kaynak:

- BoTorch Models, v0.18.1:
  https://botorch.org/docs/models

Repodaki BLL notebooku, son katman posteriorundan örneklenen ağırlıkları `EnsembleModel` biçiminde BoTorch'a aktarır. Aynı posterior ağırlık örneği tüm aday noktalarda kullanıldığı için her örnek bir fonksiyon realization'ı gibi davranır.

Bu BoTorch API uyumluluğu standarttır; ancak hangi surrogate'ın iyi optimize edeceği ampirik bir sorudur.

## 9. Ne standart teori değildir?

Aşağıdakiler “teorem” değil, modelleme / mühendislik tercihidir:

- talebin Normal likelihood ile modellenmesi,
- ağda 16, 20 veya 32 gizli nöron kullanılması,
- prior precision/standart sapma seçimi,
- 500 senaryo veya 256 posterior ağırlık örneği kullanılması,
- `AutoDiagonalNormal` seçilmesi,
- BLL backbone'un MSE ile aynı veri üzerinde eğitilmesi,
- sentetik black-box fonksiyonları,
- acquisition optimizer restart sayıları.

Bunlar problemden probleme doğrulanmalıdır.

## 10. Endüstriyel doğrulama için minimum kontrol listesi

Bir BNN/BLL + optimization çalışmasının gerçek uygulama iddiası taşıması için en az:

1. train/validation/test veya temporal holdout,
2. predictive interval coverage,
3. NLL veya proper scoring rule,
4. calibration kontrolü,
5. baseline: deterministic NN / Deep Ensemble / GP / klasik istatistiksel model,
6. senaryo sayısı ve posterior sample duyarlılığı,
7. out-of-sample constraint violation rate,
8. beklenen maliyet ve CVaR,
9. distribution shift / OOD testi,
10. inference yöntemi hassasiyeti,
11. optimizasyon açısından regret / best-so-far / sample efficiency,
12. hesaplama süresi ve operasyonel güncelleme maliyeti

raporlanmalıdır.

Kısacası: repodaki temel matematiksel yapılar literatürde yerleşik yapılardır; fakat belirli bir endüstriyel sistemde işe yaradıkları **veri ve downstream karar performansıyla ayrıca kanıtlanmalıdır**.
