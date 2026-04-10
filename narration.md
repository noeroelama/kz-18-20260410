# Presentation Narration — CASA (Climate-Adaptive Spatial Attention)
**Doctoral Seminar · Ehime University · April 2026**
**Presenter: Arif Nurwahid**

---

## Slide 1: Title

> Good afternoon, everyone. Thank you for joining today.
>
> My name is Arif Nurwahid, working under the supervision of Professor Matsuura. Today I will present my progress on the CASA module — Climate-Adaptive Spatial Attention — a new part for spatiotemporal rainfall forecasting across Indonesian provinces.
>
> The main idea: instead of using the same fixed spatial weights for 20 years, CASA makes those weights change every month, based on real climate conditions like ENSO and IOD.

---

## Slide 2: Section — Why Static Spatial Weights Fail

*(Section divider — move on naturally.)*

---

## Slide 3: One Assumption That Does Not Hold Physically

> Let me start with the problem. In our current LSTM-GSTARX model, the spatial lag is computed using a static weight matrix — W-static — that is made once from Haversine distances between province centers, then kept the same for the entire period from 2005 to 2024.
>
> This assumes that the spatial relationships between provinces never change. But we know they do, for three reasons.
>
> First, ENSO. During El Niño, the Walker Circulation becomes weaker. Provinces that normally have similar rainfall can suddenly behave in opposite ways.
>
> Second, the Indian Ocean Dipole. A positive IOD dries out Sumatra and Java, but Papua on the eastern side is not affected. So provinces that normally move together suddenly become disconnected.
>
> Third, monsoon changes. The main moisture flow reverses direction between seasons, which changes which provinces affect which.
>
> What I propose is to replace the static W with a dynamic W-sub-t — a mix of geographic weights and learned weights that changes each month.

---

## Slide 4: Section — Background

*(Section divider.)*

---

## Slide 5: Base Model — From GSTAR to LSTM-GSTARX

> Before I explain CASA, let me first show the model where CASA lives.
>
> We started with GSTAR — a linear model that uses geographic weights to capture spatial relationships. Then we added climate indices as extra inputs to get GSTARX. Then we replaced the linear part with an LSTM to handle nonlinear patterns — that gives us LSTM-GSTARX.
>
> All of these models use the same static weight matrix. CASA is the fourth step: it replaces that static matrix with a dynamic one. Everything else — the LSTM, the training — stays exactly the same. CASA only changes how the spatial weights are calculated.

---

## Slide 6: What is a Graph Neural Network?

> Now, some background on graph neural networks, since this is where the attention idea comes from.
>
> A graph is just nodes connected by edges. In our case, the 38 provinces are nodes, and edges show which provinces are close to each other.
>
> A GNN updates each node by collecting information from its neighbors. The idea is simple: the new value of node i comes from combining information from the nodes around it.
>
> The Graph Convolutional Network — GCN — by Kipf and Welling uses fixed weights based on how many connections each node has. But that means all neighbors are treated the same way. For rainfall, this is a problem: during El Niño, a faraway province connected by atmospheric patterns might matter more than a nearby one. We need weights that can learn which neighbors actually matter.

---

## Slide 7: GAT and GATv2

> GAT — Graph Attention Network — solves this by adding learnable attention weights. The model learns how much each neighbor matters, using a score that looks at the features of both nodes.
>
> The score is computed by joining the features of nodes i and j — the vertical bar symbol here means concatenation, which just means putting two vectors together end to end — then passing through a small neural network with LeakyReLU.
>
> But there is a hidden problem that Brody and colleagues found in 2022. The importance ranking of neighbors in GAT does not actually change depending on who is asking. I will show the proof of this later.
>
> Their fix, called GATv2, applies the nonlinearity after joining the features instead of before. This way the model can see real connections between the two nodes.
>
> CASA is built on top of GATv2 with three changes: we look at all province pairs, not just neighbors. We add climate information. And we mix the result with geographic weights.

---

## Slide 8: Related Work

> Before showing CASA, let me put it next to other methods that also try to learn spatial structure.
>
> Graph WaveNet by Wu and colleagues learns an adaptive adjacency matrix through node embeddings. It is a good idea, but the matrix is made once and then stays the same — it does not change every month. And there is no geographic information in it.
>
> AGCRN by Bai and colleagues is more flexible — it builds graph structure from data. But again, the graph is fixed after training. It does not adjust for different climate conditions.
>
> WST-ANet uses wavelet decomposition with spatial attention. It has some climate conditioning, but there is no geographic base and no way to mix between geography and learned weights.
>
> The comparison table later will show that CASA is the first to combine all of these: weights that change over time, a geographic starting point, climate conditioning, and a learnable mixing gate.

---

## Slide 9: Section — CASA Architecture

*(Section divider.)*

---

## Slide 10: Three Stages Overview

> Here is the CASA architecture. Three stages, all running at every time step.
>
> We take the previous month's rainfall plus climate indices. Stage 1 compresses the climate state into a 16-number context vector. This goes into Stage 2a, which calculates attention scores for all province pairs, and Stage 2b, which calculates a mixing gate — one number that decides how much to trust geography versus learned patterns.
>
> Stage 3 mixes the geographic weights and the learned weights, then multiplies with the rainfall vector to get the spatial lag. This goes into the LSTM, which stays the same as before.

---

## Slide 11: Stage 1 — Climate Context Vector (Formula)

> Stage 1 calculates the climate context vector c-sub-t. We take four inputs: mean rainfall, spatial standard deviation, Niño 3.4, and DMI. We multiply by a learnable matrix, add bias, and apply tanh.
>
> Mean rainfall tells us: is Indonesia as a whole wet or dry? Standard deviation tells us: is the rainfall spread out evenly or concentrated in a few areas? And the two climate indices directly show the Pacific and Indian Ocean conditions.

---

## Slide 12: What Does c_t Represent?

> What does this vector actually mean? Think of it as a climate fingerprint for the current month. It has 16 numbers, and each one learns to respond to some mix of the four inputs.
>
> Why tanh? It keeps everything between minus one and plus one, so no single extreme climate event can take over. And why only 4 inputs? Using all 38 provinces would need too many parameters. The summary numbers already capture the overall pattern. Province-level detail comes later, in Stage 2.
>
> One important point: this context vector is used two times — in the attention scores and in the mixing gate. So it shapes both which provinces affect each other and how much we trust the result.

---

## Slide 13: Activation Functions

> Before Stage 2, let me quickly go over three functions we use.
>
> Softmax takes raw scores and turns them into shares that add up to 1. Higher score means bigger share. We use this to turn attention scores into proper weights.
>
> Sigmoid squeezes any number into the range between 0 and 1. We use this for the mixing gate — alpha-t.
>
> LeakyReLU passes positive values through, and lets negative values through with a small slope of 0.2. This prevents the "dead neuron" problem where a neuron stops learning completely. We use this in the attention calculation, following the standard from the GAT papers.

---

## Slide 14: Stage 2a — Attention Scores

> Stage 2a calculates how much each province should affect every other province this month.
>
> For each pair of provinces i and j, we make an input vector: the rainfall of both provinces, the difference between them, and the 16-number climate context. That is 19 inputs in total. This goes through a weight matrix, LeakyReLU, and a projection to get a single score.
>
> We do this for all 38 times 38 — that is 1,444 — pairs. Then row-by-row softmax makes sure each row adds up to 1. The result is our learned attention matrix.

---

## Slide 15: Stages 2b & 3 — Gate and Final Weights

> Stage 2b is simple but important. It calculates alpha-t — a single number between 0 and 1 — using sigmoid on a linear combination of the context vector.
>
> When alpha-t is close to 1, geography dominates. When it is close to 0, the learned weights take over. The table shows what we expect for different climate conditions. During normal months, we expect alpha-t near 1. During strong El Niño, below 0.3.
>
> I want to be clear: these are expected values based on our design, not measured results. We will check this in the experiments.
>
> Stage 3 mixes: W-sub-t is alpha-t times the geographic matrix plus one-minus-alpha-t times the learned matrix. Then we multiply by the rainfall vector.

---

## Slide 16: Forward Pass Algorithm

> This slide shows the complete forward pass as pseudocode. It brings together everything we discussed into one algorithm. The key thing to notice: this entire calculation replaces just one line in the original model — the static spatial lag.

---

## Slide 17: Section — Mathematical Proofs

*(Section divider.)*

---

## Slide 18: Proof A — GAT Makes Static Attention

> This proof explains a basic limitation of GAT that led us to choose the GATv2 approach.
>
> Think about it this way. Province i is asking: "which other province matters most to me right now?" The attention score e-ij measures how important province j is to province i.
>
> The problem with GAT is: the answer does not depend on who is asking. If Java thinks Sulawesi is more important than Papua, then every single province thinks the same — even Papua itself. During El Niño, that makes no sense. Papua should care about different provinces than Java does.
>
> The proof is short. Because LeakyReLU works on each number separately, the GAT score splits into two parts: g-of-i, which only depends on the province doing the asking, and h-of-j, which only depends on the province being looked at. When we compare two neighbors, the g-of-i parts are the same and cancel each other. So the ranking depends only on h-of-j — it does not matter who is asking.
>
> That is why CASA uses the GATv2 approach, where the weight matrix sees both provinces together before the nonlinearity is applied.

---

## Slide 19: Proofs B-D — Mathematical Guarantees

> CASA comes with three mathematical guarantees.
>
> Proof B: the weight matrix W-sub-t is row-stochastic. Every row adds up to 1, so the spatial lag is a proper weighted average. The output stays in the same scale as the rainfall data.
>
> Proof C: CASA can only match or beat the static model. If there is nothing useful to learn, the optimizer pushes the bias to a very large number, alpha goes to 1, and we get back the original geographic weights. In theory, the training error is guaranteed to be no worse. For test error, the extra parameters from CASA are very small compared to the total model, so the risk of overfitting is low — but we will still check this in our experiments.
>
> Proof D: the weights cannot move too far from geography. The distance between W-sub-t and the geographic matrix is directly controlled by alpha-t. Higher alpha-t means closer to geography. What is interesting here is that alpha-t does two jobs at once: it mixes the weights, and it also automatically limits how far the result can move from geography — like a built-in safety limit. We do not need to add a separate L2 penalty to the loss function. And this limit adjusts itself: tight during normal months when geography is enough, loose during El Niño when the model needs more freedom.

---

## Slide 20: Section — Positioning

*(Section divider.)*

---

## Slide 21: No Prior Method Combines All of These

> This table compares CASA with the methods I introduced earlier. You can see that Graph WaveNet and AGCRN learn adaptive weights but not every time step, and they have no geographic base. GAT and GATv2 are dynamic but do not use climate information and have no geographic starting point. WST-ANet has partial climate conditioning but no mixing gate.
>
> CASA is the first to put all of these together in one module, made for rainfall forecasting.

---

## Slide 22: Section — Validation Plan

*(Section divider.)*

---

## Slide 23: Why Validation? And What Will We Test?

> CASA is still a proposal at this point. The proofs show the math is correct, but we have not yet tested it on real data. We need to check four things.
>
> First: does CASA beat the baseline models? We will compare the full chain — from GSTAR to GSTARX to LSTM-GSTARX to LSTM-GSTARX-CASA.
>
> Second: which parts actually help? We will run ablation experiments — taking out the geographic base, taking out climate input, freezing the gate, limiting to neighbors only. If each removal makes the results worse, it shows that every design choice matters.
>
> Third: does alpha-t behave the way we expect based on climate science? We will plot its value over time next to Niño 3.4, and compare alpha-t values across El Niño, La Niña, and Neutral periods using the Kruskal-Wallis test — a standard statistical test that checks whether three or more groups come from the same distribution. We will also look at heat maps of the weight matrix W-sub-t under different climate conditions.
>
> Fourth: is the improvement real, or just random chance? We will use the Diebold-Mariano test — a standard test that compares the prediction errors of two forecasting models and tells us whether one is truly better than the other. We apply this to RMSE values from five-fold walk-forward cross-validation.

---

## Slide 24: Thank You

> That is my progress report. In short: CASA is a small module that makes spatial weights respond to climate conditions. It has three stages — context, attention, and mixing — with mathematical guarantees that the weights are valid, performance cannot get worse, and the weights stay close to geography when the data says they should.
>
> Next steps: write the CASA code, connect it to the cross-validation pipeline, and run the validation experiments.
>
> Thank you. I am happy to take questions.

---

## Slide 25: References

*(Show if needed. The full list is on screen.)*
