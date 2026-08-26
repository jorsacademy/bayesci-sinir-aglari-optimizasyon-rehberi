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

Pyro'nun resmi dokümantasyonunda BNN katmanları ve ağırlık belirsizliğinin variational dağılımlarla temsil edilmesi doğrudan yer alır:

- Pyro BNN docs: https://docs.pyro.ai/en/stable/contrib.bnn.html
- Pyro `PyroModule` / `PyroSample`: https://docs.pyro.ai/en/stable/nn.html

## 2. Heteroskedastik aleatorik belirsizlik

Heteroskedastik regresyonda gözlem varyansı girdiye bağlıdır:

\[
Y\mid x,w \sim \mathcal N(\mu_w(x),\sigma_w^2(x)).
\]

Bu yaklaşım yeni veya spekülatif değildir. Kendall ve Gal (2017), girdiye bağlı aleatorik belirsizliği ve epistemik belirsizliği birlikte ele alan Bayesian deep learning çerçevesini açıkça formüle eder.

Birincil kaynak:

- Kendall, A. & Gal, Y. (2017), *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*
  https://arxiv.org/abs/1703.04977

Depeweg ve arkadaşları da Bayesian deep learning içinde aleatorik ve epistemik belirsizliğin ayrıştırılmasını risk-duyarlı karar verme bağlamında inceler:

- https://arxiv.org/abs/1710.07283

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

dolayısıyla:

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

Bu sınıf 1950'lerden beri yöneylem araştırmasında kullanılmaktadır. Modern scenario approach literatürü, örneklenmiş kısıtlarla çözülen problemler için hangi koşullarda olasılıksal feasibility garantileri elde edilebileceğini inceler.

Kaynaklar:

- Stanford EE364A chance-constrained optimization notları:
  https://stanford.edu/class/ee364a/lectures/chance_constr.pdf
- Campi & Garatti (2008), *The Exact Feasibility of Randomized Solutions of Uncertain Convex Programs*, SIAM Journal on Optimization:
  https://epubs.siam.org/doi/10.1137/07069821X
- Campi & Garatti (2011), sampling-and-discarding:
  https://doi.org/10.1007/s10957-010-9754-6

Bu nedenle repodaki notebook bilinçli olarak şu uyarıyı yapar:

> 500 posterior senaryoda %95 ampirik feasibility görmek, tek başına gerçek dağılım altında %95 güvence kanıtı değildir.

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

Birincil kaynak:

- Rockafellar, R. T. & Uryasev, S. (2000), *Optimization of Conditional Value-at-Risk*, Journal of Risk, 2(3), 21–41.
  DOI: https://doi.org/10.21314/JOR.2000.038
- Yazarın yayın listesi:
  https://uryasev.github.io/publications/

Senaryo yaklaşımında beklenti sonlu örneklem ortalamasıyla değiştirilerek lineer / konveks programlama formülasyonları elde edilebilir.

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

Repodaki BNN'nin rolü burada dağılımın kendisini analitik olarak varsaymak yerine posterior predictive senaryolar üretmektir.

## 7. Ne standart teori değildir?

Aşağıdakiler “teorem” değil, modelleme tercihidir:

- talebin Normal likelihood ile modellenmesi,
- ağda 16 veya 20 gizli nöron kullanılması,
- prior standart sapmasının 0.5 veya 0.8 seçilmesi,
- 500 senaryo kullanılması,
- `AutoDiagonalNormal` seçilmesi,
- sentetik veri fonksiyonları.

Bunlar problemden probleme doğrulanmalıdır.

## 8. Endüstriyel doğrulama için minimum kontrol listesi

Bir BNN + optimization çalışmasının gerçek uygulama iddiası taşıması için en az:

1. train/validation/test veya temporal holdout,
2. predictive interval coverage,
3. NLL veya proper scoring rule,
4. calibration kontrolü,
5. baseline: deterministic NN / ensemble / GP / klasik istatistiksel model,
6. senaryo sayısı duyarlılığı,
7. out-of-sample constraint violation rate,
8. beklenen maliyet ve CVaR,
9. distribution shift / OOD testi,
10. inference yöntemi hassasiyeti

raporlanmalıdır.

Kısacası: repodaki matematiksel yapılar literatürde yerleşik yapılardır; fakat belirli bir endüstriyel sistemde işe yaradıkları **veri ve karar performansıyla ayrıca kanıtlanmalıdır**.
