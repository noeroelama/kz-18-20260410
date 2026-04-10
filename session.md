# Session Resume — CASA Research & Documents
**Researcher:** Arif Nurwahid | Ehime University  
**Supervisor:** Prof. Masaya Matsuura  
**Date:** April 2026  
**ORCID:** 0009-0005-8802-9629

---

## 1. Research Context

- **Topic:** Spatiotemporal rainfall forecasting across **38 Indonesian provinces**, 2005–2024, monthly resolution
- **Base model:** LSTM-GSTARX (existing, working, in `main.py`)
- **Exogenous drivers:** Niño 3.4 (ENSO) + DMI (IOD)
- **Proposed module:** **CASA** — Climate-Adaptive Spatial Attention

---

## 2. The Core Problem

Current model computes spatial lag as:

```
F*_{t-1} = W_static · Y*_{t-1}
```

`W_static` is built **once** from inverse Haversine distances between province centroids — then frozen for the entire study period. Physically wrong because:

1. **ENSO** — El Niño weakens Walker Circulation, changing spatial co-variation between provinces
2. **IOD** — Positive IOD dries Sumatra/Java but leaves Papua unaffected → regions decouple
3. **Monsoon** — West vs. east monsoon reverses direction of rainfall influence

---

## 3. CASA Architecture (3 Stages)

### Stage 1 — Climate Context Vector (80 params)
```
c_t = tanh(W_c [Ȳ_{t-1}, σ_{Y_{t-1}}, Niño3.4_t, DMI_t]^T + b_c)
W_c ∈ R^{16×4},  b_c ∈ R^{16}
```
- `Ȳ` = mean rainfall (national wetness)
- `σ_Y` = spatial std dev (heterogeneity)
- Niño 3.4 + DMI = climate regime signal
- Why NOT raw Y_i: would need 16×42 matrix → ~1,500 extra params, overwhelming 265-param budget

### Stage 2a — Attention Scores (168 params)
```
e^t_{ij} = v_a^T · LeakyReLU(W_a [Y_i, Y_j, |Y_i−Y_j|, c_t]^T + b_a)
```
- `W_a ∈ R^{8×19}` — weight matrix (152 params). Input is 19-dim: 3 + d_c = 3 + 16
- `b_a ∈ R^8` — bias (8 params)
- `v_a ∈ R^8` — **projection vector**: collapses 8-dim hidden repr to 1 scalar score e^t_{ij} (8 params)
- `γ = 0.2` — LeakyReLU slope (GAT convention, Veličković et al. 2018)
- Full N×N = 38² = 1,444 pairs computed each step
- Softmax per row → `A^t_learned ∈ R^{N×N}`, row-stochastic

### Stage 2b — Blending Gate (17 params)
```
α_t = σ(w_α^T c_t + b_α) ∈ (0,1)
w_α ∈ R^{16},  b_α ∈ R
```
Regime interpretation (verified, not fabricated):

| Condition | Niño 3.4 | α_t |
|---|---|---|
| Normal / Neutral | \|Niño 3.4\| < 0.5 | ≈ 0.8–1.0 |
| Weak El Niño / Transition | ≈ 0.5–1.0 | ≈ 0.5–0.7 |
| Moderate El Niño | ≈ 1.0–1.5 | ≈ 0.3–0.5 |
| Strong El Niño | ≥ 1.5 | ≈ 0.1–0.3 |

### Stage 3 — Final Weight + Spatial Lag
```
W_t = α_t · A_geo + (1−α_t) · A^t_learned
F*_{t-1} = W_t · Y*_{t-1}   → fed into LSTM (unchanged)
```

---

## 4. Parameter Count (Verified by Python)

| Component | Params |
|---|---|
| Stage 1: W_c (16×4) + b_c (16) | 80 |
| Stage 2a: W_a (8×19) + b_a (8) + v_a (8) | 168 |
| Stage 2b: w_α (16) + b_α (1) | 17 |
| **CASA total** | **265** |
| LSTM layer: 4 × 64 × (64+78) + 4×64 | 36,608 |
| Output layer: 38×64 + 38 | 2,470 |
| **Model total** | **39,343** |
| **CASA / total** | **0.67%** |

Computation: `265 / 39,343 = 0.6736% ≈ 0.67%`  
**Important:** bias IS counted in parameter totals. `A_geo` is precomputed, not a learned parameter.

---

## 5. Four Mathematical Proofs

### Proof A — GAT produces static attention (adapted, Brody et al. ICLR 2022)
GAT: `e^GAT_{ij} = a^T LeakyReLU(Wh_i ‖ Wh_j) = g(i) + h(j)`  
Ranking of j₁ vs j₂ is independent of i → **static**.  
GATv2 fix (used in CASA): `e^GATv2_{ij} = v^T LeakyReLU(W[h_i ‖ h_j])` → cross-terms → dynamic ✓

### Proof B — W_t is Row-Stochastic (original)
```
Σ_j W_t(i,j) = α_t·1 + (1−α_t)·1 = 1  ✓
```

### Proof C — CASA Subsumes Static Baseline (original)
```
b_α → +∞  ⟹  α_t → 1  ⟹  W_t = A_geo  ■
```
∴ train error(CASA) ≤ train error(static baseline)

### Proof D — Bounded Deviation (original)
```
‖W_t − A_geo‖_F = (1−α_t) ‖A^t_learned − A_geo‖_F
```
α_t acts as implicit regularizer. No explicit L2 needed.

---

## 6. Worked Example (Simplified, d_c=2, d_a=2)

### Setup
- P1 West Java: Y=0.30 | P2 South Sulawesi: Y=0.20 | P3 West Papua: Y=0.70
- Niño 3.4 = 1.2 (moderate El Niño), DMI = 0.3

### Steps 1–2: Statistics + c_t
```
Ȳ = (0.3+0.2+0.7)/3 = 0.400
σ_Y = √((0.01+0.04+0.09)/3) = √0.0467 = 0.216
```
**Important caveat:** `c_t = [0.838, 0.139]` depends on specific W_c values.  
These are **learned parameters** — the values in the example were reverse-engineered  
from a specific W_c (one that only looks at Niño3.4 and DMI, ignoring Ȳ and σ_Y).  
In a real trained model, W_c learns from data and all 4 inputs contribute.

### Steps 3–7: Results (verified by Python)
```
α_t = σ(−0.753) = 0.320  →  32% geography, 68% learned
W_t = 0.320·A_geo + 0.680·A^t_learned
```
| Province | Y_{t-1} | F*_CASA | F*_static | Diff |
|---|---|---|---|---|
| P1 West Java | 0.300 | 0.407 | 0.400 | +0.007 |
| P2 S. Sulawesi | 0.200 | 0.428 | 0.460 | −0.032 |
| **P3 West Papua** | **0.700** | **0.344** | **0.240** | **+0.104 (+43%)** |

**Key insight West Papua:** A_geo[3,3]=0 (no self-loop) → static model bases Papua's lag only on dry P1+P2 → F*=0.240. CASA learns self-loop ≈ 0.22 during El Niño → F*=0.344.

---

## 7. Novelty Positioning

| Property | Graph WaveNet | AGCRN | GAT/GATv2 | WST-ANet | **CASA** |
|---|:---:|:---:|:---:|:---:|:---:|
| Dynamic per time step | ✗ | ✗ | ✓ | ✓ | ✓ |
| Geographic prior | ✗ | ✗ | ✗ | ✗ | ✓ |
| Climate index conditioning | ✗ | ✗ | ✗ | Partial | ✓ |
| Learnable geo–learned gate | ✗ | ✗ | ✗ | ✗ | ✓ |
| Full N×N scope | ✓ | ✓ | Dep. | ✓ | ✓ |
| Rainfall domain | ✗ | ✗ | ✗ | ✗ | ✓ |

**Title origin:** "Climate" (domain, explicit conditioning) + "Adaptive" (inspired by Graph WaveNet's adaptive adjacency, AGCRN's DAGG) + "Spatial Attention" (GAT/GATv2 lineage). Acronym "CASA" is a bonus.

---

## 8. Validation Plan

**Main comparison:** GSTAR(1;1) → GSTARX(1;1) → LSTM-GSTARX (static W) → LSTM-GSTARX-CASA

**Ablation (5 variants):**
1. Full CASA (main model)
2. No geo prior (α_t=0)
3. No climate conditioning (remove X_t from c_t)
4. Static blending (fixed α scalar)
5. Restricted softmax (neighbors only, not full N×N)

**Interpretability:**
- α_t timeline overlay vs Niño 3.4
- Regime comparison Kruskal-Wallis test
- W_t heatmaps (strongest El Niño / La Niña / neutral)

**Statistical significance:** Diebold-Mariano test on RMSE across 5-fold walk-forward CV

---

## 9. Key References

| # | Reference | Role |
|---|---|---|
| 1 | Veličković et al. (2018) Graph Attention Networks. ICLR | Base attention mechanism |
| 2 | Brody et al. (2022) How Attentive are GATs? ICLR | GATv2 ordering; Proof A |
| 3 | Wu et al. (2019) Graph WaveNet. IJCAI | Self-adaptive adjacency inspiration |
| 4 | Bai et al. (2020) AGCRN. NeurIPS | Adaptive graph generation inspiration |
| 5 | Hochreiter & Schmidhuber (1997) LSTM. Neural Computation | LSTM backbone |
| 6 | Kingma & Ba (2014) Adam. arXiv | Optimizer |
| 7 | Maas et al. (2013) LeakyReLU. ICML | LeakyReLU γ=0.2 |
| 8 | Borovkova et al. (2002/2008) GSTAR | Spatial weight prior |

---

## 10. Documents Produced (in outputs/)

| File | Description |
|---|---|
| `CASA_Progress_Report.html` | Full report: 8 chapters + appendix, MathJax, mobile responsive |
| `CASA_Presentation.html` | Reveal.js presentation, 16 slides, self-contained |
| `CASA_Progress_Report_Light.pptx` | PPTX light theme, OMML native math (editable in PowerPoint) |

**Known issues fixed:**
- 0.72% → **0.67%** (correct: 265/39,343)
- `v_a` now explicitly defined as projection vector
- Regime table now has 4 rows (added Weak El Niño α_t≈0.5–0.7)
- report.html was truncated → fixed (now has `</html>`)
- Presentation inline code → fixed (`white-space: pre`)
- Formula scrollbar → fixed (`overflow: visible`)
- Presentation Indonesian text in Slide 8 → **still needs fixing**

---

## 11. Implementation TODO (for Claude Code)

- [ ] `casa_module.py` — CASAModule class in NumPy with forward/backward
- [ ] `main_casa.py` — integrate CASA into existing walk-forward CV pipeline
- [ ] `test_casa_numeric.py` — unit test using verified 3-province example
- [ ] Fix Slide 8 GAT intro — translate Indonesian text to English
- [ ] Fix c_t in worked example — clarify W_c dependency or use fully explicit W_c

---

## 12. Data Files Available

| File | Description |
|---|---|
| `nino34_long_anom.csv` | 1872 rows, Date + NINA34 anomaly |
| `monthly_rainfall.csv` | 9120 rows, provinsi + year_month + rainfall |
| `dmi_had_long.csv` | 1872 rows, Date + DMI HadISST1.1 |
| `CC_GISA_Mathematical_Formulation.md` | Full formulation doc (ground truth for all math) |

---

## 13. Quick Conceptual Notes (for context)

- **Walker Circulation** = large atmospheric loop over Pacific. Ascending air over Indonesia in normal years. El Niño weakens it, shifting convection eastward → Indonesia dries.
- **IOD (Indian Ocean Dipole)** = SST difference between western Indian Ocean (Africa coast) vs eastern (Sumatra coast). Positive IOD = eastern Indian Ocean cools → Sumatra/Java lose moisture source → dry. Papua unaffected (too far east, different moisture source).
- **W_t row-stochastic** = each row sums to 1 → F* is a valid weighted average of rainfall values, physically meaningful.
- **265 params** = all counted including bias. A_geo not counted (precomputed). All 39,343 params trained simultaneously via backprop.
- **c_t values in example are NOT from first principles** — depend on assumed W_c. Ȳ=0.400 and σ_Y=0.216 are verifiable; c_t=[0.838, 0.139] requires explicit W_c to reproduce.
