# Multi-output BNN ve Çok Amaçlı Bayesçi Optimizasyon: Teorik Dayanak

Bu dosya, `notebooks/multi_output_bnn_multi_objective_botorch.ipynb` örneğinin matematiksel ve algoritmik dayanağını özetler.

## 1. Multi-output model ile multi-objective optimization aynı şey değildir

Bir multi-output probabilistic model aynı girdiden birden fazla rassal çıktı üretir:

\[
Y(x)=\begin{bmatrix}Y_1(x) & Y_2(x) & \cdots & Y_m(x)\end{bmatrix}^\top.
\]

Multi-objective optimization ise karar vericinin birden fazla objective arasındaki trade-off'u ele aldığı optimizasyon problemidir:

\[
\max_x \; (f_1(x),\ldots,f_m(x)).
\]

Dolayısıyla bir BNN'nin üç çıktı üretmesi tek başına problemi çok amaçlı yapmaz. Bu çıktıların objective olarak tanımlanması, yönlerinin belirlenmesi ve Pareto karşılaştırmasına sokulması karar modelinin parçasıdır.

## 2. Repodaki multi-output BNN

Notebook ortak Bayesçi hidden representation kullanır:

\[
h_w(x)=\tanh(W_1x+b_1),
\]

\[
\mu_w(x)=W_2h_w(x)+b_2\in\mathbb R^3.
\]

Likelihood:

\[
Y\mid x,w\sim\mathcal N\left(\mu_w(x),\operatorname{diag}(\sigma_1^2,\sigma_2^2,\sigma_3^2)\right).
\]

Bu yapı geçerli bir multi-output BNN'dir; fakat residual covariance diagonal seçildiği için en zengin joint-output model değildir.

Çıktılar yine de tamamen bağımsız değildir. Aynı Bayesçi hidden ağırlıkları ve aynı posterior realization'ı paylaştıkları için posterior function samples objective'ler arasında ortak latent belirsizlik taşır.

Daha ileri modelde residual covariance de tam matris olarak modellenebilir veya multi-task yapı kurulabilir.

## 3. Minimizasyon objective'lerini maksimize edilen uzaya çevirmek

BoTorch multi-objective acquisition fonksiyonlarında standart kullanım bütün objective'lerin maksimize edilmesidir.

Bu nedenle fiziksel problem:

\[
\text{quality}\uparrow,\qquad
\text{energy}\downarrow,\qquad
\text{cycle time}\downarrow
\]

şu objective vektörüne çevrilebilir:

\[
f(x)=\left(\text{quality}(x),-\text{energy}(x),-\text{cycle}(x)\right).
\]

Bu yalnızca işaret dönüşümüdür; Pareto sıralamasını doğru yönde ifade eder.

## 4. Pareto dominance

Maksimizasyon probleminde objective vektörü \(a\), \(b\)'yi domine eder eğer:

\[
a_i\ge b_i \quad \forall i
\]

ve en az bir objective için

\[
a_j>b_j.
\]

Hiçbir başka feasible nokta tarafından domine edilmeyen çözümler Pareto kümesini oluşturur.

BoTorch `is_non_dominated` yardımcı fonksiyonu bu hesabı destekler.

Resmi kaynak:

- BoTorch Multi-Objective Bayesian Optimization, v0.18.1: https://botorch.org/docs/multi_objective

## 5. Hypervolume

Pareto cephesini tek bir sayıyla değerlendirmek için hypervolume sık kullanılan bir metriktir.

Maksimizasyon probleminde reference point \(r\), ilgilenilen objective bölgesinin kötü tarafında seçilir. Pareto kümesinin \(r\)'ye göre domine ettiği hacim hypervolume'dur.

Hypervolume değeri reference point'e bağlıdır. Dolayısıyla reference point seçimi matematiksel zorunluluk değil, domain bilgisine dayalı önemli bir modelleme tercihidir.

BoTorch hem Pareto utility'leri hem hypervolume / box decomposition araçları sağlar.

Kaynak:

- https://botorch.org/docs/multi_objective

## 6. qLogEHVI

BoTorch 0.18.1 çok amaçlı Bayesian optimization için şu acquisition fonksiyonlarını doğrudan destekler:

- `qLogExpectedHypervolumeImprovement` (qLogEHVI),
- `qLogNoisyExpectedHypervolumeImprovement` (qLogNEHVI),
- `qLogNParEGO`.

Resmi dokümantasyon:

- https://botorch.org/docs/multi_objective
- API: https://botorch.readthedocs.io/en/latest/acquisition.html

qLogEHVI'nin temel fikri yeni adayın mevcut Pareto cephesine göre beklenen hypervolume improvement'ını posterior altında değerlendirmektir.

Notebookta kullanılan sınıfın güncel API imzası BoTorch 0.18.1'de:

```python
qLogExpectedHypervolumeImprovement(
    model,
    ref_point,
    partitioning,
    sampler=None,
    ...
)
```

şeklindedir.

BoTorch dokümantasyonu log-EHVI ailesini eski EHVI/NEHVI hesaplarına göre daha iyi numerik davranış için öne çıkarmaktadır.

Temel çalışmalar:

- Daulton, Balandat & Bakshy (2020), *Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization*.
- Daulton, Balandat & Bakshy (2021), *Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement*.
- Ament et al. (2023), *Unexpected Improvements to Expected Improvement for Bayesian Optimization*.

Bu çalışmaların bağlantıları BoTorch'un resmi multi-objective dokümantasyonunda verilmiştir.

## 7. Neden custom BNN BoTorch'ta kullanılabilir?

BoTorch model arayüzü GP ile sınırlı değildir. Monte Carlo acquisition fonksiyonları için temel koşul modelin bir `posterior()` döndürmesi ve posteriorun örneklenebilir olmasıdır.

Resmi kaynak:

- BoTorch Models v0.18.1: https://botorch.org/docs/models

Notebook Pyro posterior örneklerini `EnsembleModel` arayüzü üzerinden BoTorch'a taşır. Acquisition optimizasyonu gradient tabanlı olduğundan örneklerden input'a backpropagation yolu korunmalıdır.

## 8. Notebookun bilinçli sınırlamaları

Aşağıdakiler teorem değil, öğretim/modelleme tercihidir:

- 24 başlangıç deneyi,
- iki karar değişkeni,
- üç objective,
- Gaussian likelihood,
- diagonal residual covariance,
- `AutoDiagonalNormal`,
- 128 posterior sample,
- reference point'in başlangıç örneklerinden türetilmesi,
- qLogEHVI kullanılması,
- objective standardizasyonu.

Gerçek uygulamada şu kontroller yapılmalıdır:

1. her output için predictive calibration,
2. output korelasyonlarının doğrulanması,
3. Pareto front'un out-of-sample testi,
4. hypervolume reference-point duyarlılığı,
5. GP / Deep Ensemble / deterministic response-surface baseline'ları,
6. qLogEHVI vs qLogNEHVI seçimi,
7. constraint'lerin ayrıca modellenmesi,
8. deney maliyeti ve sample efficiency,
9. distribution shift / OOD testi,
10. karar vericinin Pareto kümesinden son seçim mekanizması.

## 9. Endüstri mühendisliği kullanım alanları

- kalite – enerji – çevrim süresi proses optimizasyonu,
- servis seviyesi – maliyet – emisyon tedarik zinciri tasarımı,
- throughput – tardiness – enerji çizelgeleme,
- availability – bakım maliyeti – arıza riski bakım politikası,
- performans – ağırlık – üretim maliyeti mühendislik tasarımı.

Özetle: multi-output BNN ile MOBO bağlantısı standart Bayesian surrogate + Pareto/hypervolume optimizasyon mantığına dayanır. Bunun belirli bir endüstriyel sistemde iyi çalışıp çalışmadığı ise ayrıca veri ve karar performansıyla doğrulanmalıdır.
