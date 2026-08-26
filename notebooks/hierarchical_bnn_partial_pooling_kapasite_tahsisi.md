# Hierarchical BNN: Çok Fabrikalı Partial Pooling + Stokastik Kapasite Tahsisi

Bu rehber, aynı üretim teknolojisinin birden fazla fabrikada kullanıldığı fakat fabrikaların veri miktarı ve performansının farklı olduğu bir endüstri mühendisliği problemini ele alır.

Amaç iki katmanlıdır:

1. **Hierarchical BNN** ile ortak proses ilişkisini ve fabrika düzeyi sapmaları birlikte öğrenmek.
2. Posterior predictive kapasite senaryolarını **stokastik kapasite tahsisi** problemine aktarmak.

```text
çok fabrikalı proses verisi
        ↓
ortak Bayesçi sinir ağı
        +
fabrika random effect'leri
        ↓
partial pooling posterioru
        ↓
fabrika bazlı kapasite senaryoları
        ↓
Pyomo + HiGHS
        ↓
maliyet / kapasite riski dengeli tahsis
```

Temel fikir: her fabrikaya tamamen ayrı model kurmak **no pooling**, fabrika farklarını tamamen yok saymak **complete pooling**, hiyerarşik model ise **partial pooling** yaklaşımıdır.

---

## Matematiksel model

Ortak nonlinear proses ilişkisini Bayesçi sinir ağı öğrenir:

\[
h_w(x)=\tanh(W_1x+b_1),
\]

\[
g_w(x)=W_2h_w(x)+b_2.
\]

Fabrika \(j\) için random intercept ve random load slope tanımlıyoruz:

\[
a_j\sim\mathcal N(\mu_a,\tau_a^2),
\]

\[
b_j\sim\mathcal N(\mu_b,\tau_b^2).
\]

Gözlem ortalaması:

\[
\mu_{ij}=g_w(x_{ij})+a_j+b_j\,\widetilde{load}_{ij}.
\]

ve

\[
y_{ij}\mid x_{ij},j,w,a_j,b_j
\sim
\mathcal N(\mu_{ij},\sigma_j^2).
\]

Burada \(\mu_a,\tau_a,\mu_b,\tau_b\) de hiyerarşik parametrelerdir. Fabrika etkileri bağımsız sabit katsayılar değildir; ortak grup dağılımlarından gelir. Bu yapı partial pooling üretir.

Az verili fabrikanın tahmini diğer fabrikalardan öğrenilen ortak yapıya doğru daha fazla düzenlenebilir. Bu **shrinkage** veri miktarı, gürültü ve öğrenilen grup varyansına bağlıdır.

---

## Çalıştırılabilir Python örneği

Aşağıdaki kod doğrudan bir Jupyter hücresine veya `.py` dosyasına yapıştırılabilir.

```python
# pip install torch pyro-ppl numpy pandas matplotlib pyomo highspy

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample

import pyomo.environ as pyo

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pyro.set_rng_seed(SEED)
torch.set_default_dtype(torch.float32)

# ------------------------------------------------------------
# 1. Sentetik çok fabrikalı veri
# ------------------------------------------------------------

factory_names = ["Fabrika A", "Fabrika B", "Fabrika C", "Fabrika D"]
n_by_factory = [90, 60, 35, 12]

# Sentetik gerçek fabrika farkları; model bunları bilmiyor.
factory_intercept = np.array([-7.0, -1.5, 4.0, 9.0])
factory_load_effect = np.array([-5.0, -1.0, 3.0, 6.0])
factory_noise = np.array([4.2, 4.8, 5.2, 6.2])

rows = []

for j, n_j in enumerate(n_by_factory):
    temperature = np.random.normal(25.0 + 0.8*j, 4.5, n_j)
    load = np.random.uniform(0.40, 0.98, n_j)
    maintenance = np.random.uniform(0.25, 1.00, n_j)

    shared_mean = (
        94.0
        + 22.0 * np.tanh(2.2 * (load - 0.62))
        - 0.45 * np.abs(temperature - 23.0)
        + 9.0 * maintenance
        - 11.0 * (load - 0.82)**2
    )

    local_mean = (
        shared_mean
        + factory_intercept[j]
        + factory_load_effect[j] * (load - 0.65)
    )

    capacity = local_mean + np.random.normal(
        0.0, factory_noise[j], n_j
    )

    for i in range(n_j):
        rows.append(
            {
                "factory_id": j,
                "factory": factory_names[j],
                "temperature": temperature[i],
                "load": load[i],
                "maintenance": maintenance[i],
                "capacity": capacity[i],
            }
        )

df = pd.DataFrame(rows)
print(df.groupby("factory").size())

# ------------------------------------------------------------
# 2. Standardizasyon
# ------------------------------------------------------------

features = ["temperature", "load", "maintenance"]

X_raw = df[features].to_numpy(np.float32)
y_raw = df["capacity"].to_numpy(np.float32)
factory_id_np = df["factory_id"].to_numpy(np.int64)

X_mean = X_raw.mean(axis=0, keepdims=True)
X_std = X_raw.std(axis=0, keepdims=True) + 1e-6
y_mean = float(y_raw.mean())
y_std = float(y_raw.std() + 1e-6)

X_np = ((X_raw - X_mean) / X_std).astype(np.float32)
y_np = ((y_raw - y_mean) / y_std).astype(np.float32)

X = torch.tensor(X_np)
y = torch.tensor(y_np)
factory_id = torch.tensor(factory_id_np, dtype=torch.long)

# ------------------------------------------------------------
# 3. Hierarchical BNN
# ------------------------------------------------------------

class HierarchicalCapacityBNN(PyroModule):
    def __init__(self, in_features, n_factories, hidden=18):
        super().__init__()
        self.n_factories = n_factories

        self.hidden = PyroModule[nn.Linear](in_features, hidden)
        self.hidden.weight = PyroSample(
            dist.Normal(0.0, 0.8)
            .expand([hidden, in_features])
            .to_event(2)
        )
        self.hidden.bias = PyroSample(
            dist.Normal(0.0, 0.8)
            .expand([hidden])
            .to_event(1)
        )

        self.out = PyroModule[nn.Linear](hidden, 1)
        self.out.weight = PyroSample(
            dist.Normal(0.0, 0.8)
            .expand([1, hidden])
            .to_event(2)
        )
        self.out.bias = PyroSample(
            dist.Normal(0.0, 0.8)
            .expand([1])
            .to_event(1)
        )

    def forward(self, X, factory_id, y=None):
        base = self.out(torch.tanh(self.hidden(X))).squeeze(-1)

        mu_a = pyro.sample("mu_a", dist.Normal(0.0, 0.30))
        tau_a = pyro.sample("tau_a", dist.HalfNormal(0.40))

        mu_b = pyro.sample("mu_b", dist.Normal(0.0, 0.25))
        tau_b = pyro.sample("tau_b", dist.HalfNormal(0.30))

        with pyro.plate("factories", self.n_factories):
            a_factory = pyro.sample(
                "a_factory", dist.Normal(mu_a, tau_a)
            )
            b_factory = pyro.sample(
                "b_factory", dist.Normal(mu_b, tau_b)
            )
            sigma_factory = pyro.sample(
                "sigma_factory",
                dist.LogNormal(-1.15, 0.30),
            )

        # Standardize X içinde load ikinci kolondur.
        load_z = X[:, 1]

        mean = (
            base
            + a_factory[factory_id]
            + b_factory[factory_id] * load_z
        )

        pyro.deterministic("mu", mean)

        with pyro.plate("data", X.shape[0]):
            pyro.sample(
                "obs",
                dist.Normal(mean, sigma_factory[factory_id]),
                obs=y,
            )

        return mean

# ------------------------------------------------------------
# 4. Variational inference
# ------------------------------------------------------------

pyro.clear_param_store()

model = HierarchicalCapacityBNN(
    in_features=X.shape[1],
    n_factories=len(factory_names),
    hidden=18,
)

guide = AutoDiagonalNormal(model)

svi = SVI(
    model,
    guide,
    pyro.optim.Adam({"lr": 0.012}),
    loss=Trace_ELBO(),
)

losses = []
for step in range(3200):
    loss = svi.step(X, factory_id, y) / len(y)
    losses.append(loss)
    if (step + 1) % 640 == 0:
        print(f"Adım {step+1}: ELBO/gözlem={loss:.4f}")

# ------------------------------------------------------------
# 5. Fabrika random effect posteriorları
# ------------------------------------------------------------

latent_predictive = Predictive(
    model,
    guide=guide,
    num_samples=2500,
    return_sites=(
        "a_factory",
        "b_factory",
        "mu_a",
        "tau_a",
        "mu_b",
        "tau_b",
    ),
)

latent = latent_predictive(X, factory_id)

a_samples = (
    latent["a_factory"].detach().cpu().numpy() * y_std
)
b_samples = (
    latent["b_factory"].detach().cpu().numpy() * y_std
)

summary_rows = []
for j, name in enumerate(factory_names):
    summary_rows.append(
        {
            "factory": name,
            "n": n_by_factory[j],
            "a_mean": a_samples[:, j].mean(),
            "a_q05": np.quantile(a_samples[:, j], 0.05),
            "a_q95": np.quantile(a_samples[:, j], 0.95),
            "b_mean": b_samples[:, j].mean(),
            "b_q05": np.quantile(b_samples[:, j], 0.05),
            "b_q95": np.quantile(b_samples[:, j], 0.95),
        }
    )

random_effect_summary = pd.DataFrame(summary_rows)
print(random_effect_summary)

# ------------------------------------------------------------
# 6. Gelecek koşulunda fabrika kapasite dağılımları
# ------------------------------------------------------------

future_raw = np.array(
    [[30.0, 0.86, 0.70]] * len(factory_names),
    dtype=np.float32,
)
future_np = ((future_raw - X_mean) / X_std).astype(np.float32)

future_X = torch.tensor(future_np)
future_factory_id = torch.arange(
    len(factory_names), dtype=torch.long
)

future_predictive = Predictive(
    model,
    guide=guide,
    num_samples=3000,
    return_sites=("obs", "mu"),
)

future = future_predictive(
    future_X,
    future_factory_id,
)

capacity_samples = (
    future["obs"].detach().cpu().numpy() * y_std + y_mean
)
mean_function_samples = (
    future["mu"].detach().cpu().numpy() * y_std + y_mean
)

capacity_samples = np.clip(capacity_samples, 0.0, None)

capacity_summary = pd.DataFrame(
    {
        "factory": factory_names,
        "posterior_mean_capacity": capacity_samples.mean(axis=0),
        "q05_capacity": np.quantile(capacity_samples, 0.05, axis=0),
        "q50_capacity": np.quantile(capacity_samples, 0.50, axis=0),
        "q95_capacity": np.quantile(capacity_samples, 0.95, axis=0),
        "epistemic_sd_mean_function": mean_function_samples.std(axis=0),
    }
)

print(capacity_summary)

# ------------------------------------------------------------
# 7. Stokastik kapasite tahsisi
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)

S = 350
idx = rng.choice(
    capacity_samples.shape[0],
    size=S,
    replace=False,
)
scenario_capacity = capacity_samples[idx]

unit_reservation_cost = np.array([1.75, 1.95, 2.20, 1.85])
max_plan = np.array([135.0, 135.0, 135.0, 135.0])

demand = 365.0
shortage_penalty = 12.0

m = pyo.ConcreteModel()
m.J = pyo.RangeSet(0, len(factory_names) - 1)
m.S = pyo.RangeSet(0, S - 1)

m.x = pyo.Var(m.J, domain=pyo.NonNegativeReals)
m.delivered = pyo.Var(m.J, m.S, domain=pyo.NonNegativeReals)
m.shortage = pyo.Var(m.S, domain=pyo.NonNegativeReals)

for j in m.J:
    m.x[j].setub(float(max_plan[j]))

m.plan_link = pyo.Constraint(
    m.J,
    m.S,
    rule=lambda M, j, s: M.delivered[j, s] <= M.x[j],
)

m.capacity_link = pyo.Constraint(
    m.J,
    m.S,
    rule=lambda M, j, s: (
        M.delivered[j, s] <= float(scenario_capacity[s, j])
    ),
)

m.shortage_def = pyo.Constraint(
    m.S,
    rule=lambda M, s: (
        M.shortage[s]
        >= demand - sum(M.delivered[j, s] for j in M.J)
    ),
)

m.obj = pyo.Objective(
    expr=(
        sum(unit_reservation_cost[j] * m.x[j] for j in m.J)
        + shortage_penalty
        * (1.0 / S)
        * sum(m.shortage[s] for s in m.S)
    ),
    sense=pyo.minimize,
)

result = pyo.SolverFactory("appsi_highs").solve(m)

hierarchical_plan = np.array(
    [pyo.value(m.x[j]) for j in m.J]
)

print("Çözüm:", result.solver.termination_condition)
print(
    pd.DataFrame(
        {
            "factory": factory_names,
            "planned_capacity": hierarchical_plan,
            "unit_cost": unit_reservation_cost,
        }
    )
)

# ------------------------------------------------------------
# 8. Posterior mean-only baseline
# ------------------------------------------------------------

mean_capacity = capacity_samples.mean(axis=0)

d = pyo.ConcreteModel()
d.J = pyo.RangeSet(0, len(factory_names) - 1)

d.x = pyo.Var(d.J, domain=pyo.NonNegativeReals)
d.delivered = pyo.Var(d.J, domain=pyo.NonNegativeReals)
d.shortage = pyo.Var(domain=pyo.NonNegativeReals)

for j in d.J:
    d.x[j].setub(float(max_plan[j]))

d.plan_link = pyo.Constraint(
    d.J,
    rule=lambda M, j: M.delivered[j] <= M.x[j],
)

d.capacity_link = pyo.Constraint(
    d.J,
    rule=lambda M, j: (
        M.delivered[j] <= float(mean_capacity[j])
    ),
)

d.shortage_def = pyo.Constraint(
    expr=d.shortage >= demand - sum(d.delivered[j] for j in d.J)
)

d.obj = pyo.Objective(
    expr=(
        sum(unit_reservation_cost[j] * d.x[j] for j in d.J)
        + shortage_penalty * d.shortage
    ),
    sense=pyo.minimize,
)

_ = pyo.SolverFactory("appsi_highs").solve(d)

mean_only_plan = np.array(
    [pyo.value(d.x[j]) for j in d.J]
)

# ------------------------------------------------------------
# 9. Out-of-sample posterior senaryolarla karar karşılaştırması
# ------------------------------------------------------------

eval_idx = rng.choice(
    capacity_samples.shape[0],
    size=1200,
    replace=True,
)
eval_capacity = capacity_samples[eval_idx]


def evaluate_plan(plan, cap_scenarios):
    delivered = np.minimum(
        plan.reshape(1, -1),
        cap_scenarios,
    )
    total_delivered = delivered.sum(axis=1)
    shortage = np.maximum(demand - total_delivered, 0.0)

    reservation = float(
        np.dot(unit_reservation_cost, plan)
    )
    total_cost = reservation + shortage_penalty * shortage

    q95 = np.quantile(total_cost, 0.95)
    cvar95 = total_cost[total_cost >= q95].mean()

    return {
        "reservation_cost": reservation,
        "mean_shortage": shortage.mean(),
        "q95_shortage": np.quantile(shortage, 0.95),
        "P(shortage>0)": np.mean(shortage > 1e-8),
        "mean_total_cost": total_cost.mean(),
        "cvar95_total_cost": cvar95,
    }

comparison = pd.DataFrame(
    [
        {
            "method": "Hierarchical BNN + stochastic allocation",
            **evaluate_plan(hierarchical_plan, eval_capacity),
        },
        {
            "method": "Posterior mean only",
            **evaluate_plan(mean_only_plan, eval_capacity),
        },
    ]
)

print(comparison)
```

---

## OR/IE açısından neden anlamlı?

Bu mimari aşağıdaki yapılara doğrudan uyarlanabilir:

| Grup yapısı | Tahmin edilen belirsizlik | Karar problemi |
|---|---|---|
| Fabrikalar | kapasite / verim | üretim tahsisi |
| Makineler | işlem süresi | çizelgeleme |
| Tedarikçiler | lead time / kalite | sourcing / stok |
| Depolar | talep / hizmet süresi | replenishment |
| Bölgeler | talep | network allocation |
| Hatlar | arıza oranı | bakım ve yedek kapasite |

### Ne zaman hierarchical BNN mantıklıdır?

- Gruplar ortak bir mekanizma paylaşıyorsa,
- her grubun veri miktarı farklıysa,
- tamamen ayrı modeller veri açısından pahalıysa,
- grup bazlı belirsizlik karar modeline taşınacaksa,
- ortak nonlinear response surface gerekli ise.

### Ne zaman gereksiz olabilir?

- her grubun çok fazla verisi varsa,
- gruplar gerçekten ortak bir mekanizma paylaşmıyorsa,
- basit hiyerarşik regresyon zaten yeterliyse,
- nonlinear BNN kapasitesine ihtiyaç yoksa.

Bu son nokta kritik: **her hierarchical problem BNN gerektirmez**. Hiyerarşik lineer/GLM, Gaussian Process veya klasik mixed-effects model baseline olarak denenmelidir.

---

## Metodolojik sınırlar

Bu örnekte:

- shared neural-network ağırlıkları Bayesçidir,
- fabrika random intercept ve slope'ları hiyerarşiktir,
- posterior `AutoDiagonalNormal` ile yaklaşık hesaplanır,
- kapasite senaryoları posterior predictive'den gelir,
- stochastic allocation Pyomo ile çözülür.

Fakat aşağıdakiler otomatik garanti değildir:

1. Mean-field VI posterior korelasyonlarını doğru temsil etmeyebilir.
2. Sentetik veri gerçek fabrika heterojenliğinin tamamını temsil etmez.
3. Posterior predictive model yanlışsa optimizasyon da yanlış karar verebilir.
4. Partial pooling faydalı olabilir ama model yanlış kurulduysa aşırı pooling mümkündür.
5. Yeni fabrika / yeni proses koşulu OOD olabilir.

Gerçek çalışmada en az şu baseline ve kontroller gerekir:

- complete pooling,
- no pooling,
- hierarchical linear / mixed-effects model,
- GP veya Deep Ensemble,
- temporal holdout,
- fabrika bazlı predictive coverage,
- out-of-sample karar maliyeti,
- posterior calibration,
- inference yöntemi hassasiyeti.

---

## Teorik dayanak

Pyro'nun resmi dokümantasyonu `PyroModule` / `PyroSample` ile Bayesçi neural-network parametrelerini ve `plate` yapısıyla genel hiyerarşik modellemeyi destekler. `SVI`, `AutoDiagonalNormal`, HMC ve NUTS gibi çıkarım yöntemleri de resmi inference API'sinin parçasıdır.

- Pyro neural networks: https://docs.pyro.ai/en/stable/nn.html
- Pyro inference: https://docs.pyro.ai/en/stable/inference.html
- Pyro hierarchical forecasting / `plate`: https://docs.pyro.ai/en/stable/contrib.forecast.html

Bu nedenle burada kullanılan yapı “fabrika ID'sini feature olarak eklemek” değildir; Bayesçi neural network ile hiyerarşik random effects'in aynı probabilistic model içinde birlikte tanımlanmasıdır.
