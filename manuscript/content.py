# -*- coding: utf-8 -*-
"""Manuscript text for the IEEE Access version. Pure data, no logic.

Inline markup understood by build_ieee.py:
    *italic*   **bold**   ~subscript~   ^superscript^   [[yellow highlight: author must complete]]
"""

TITLE = ("An Empirical Benchmark of Parameter-Efficient Federated Learning Under "
         "Extreme Cross-Specialty Heterogeneity in Medical Imaging")

AUTHORS = "Mohan Babu A^1^[[, Second B. Author^2^, Member, IEEE (add all co-authors and IEEE membership grades, or delete)]]"
AFFILIATION = "^1^[[Department, Institution, City, Postal Code, Country (add one line per affiliation)]]"
CORRESPONDING = "Corresponding author: Mohan Babu A (e-mail: [[author@institution.edu]])."
SUPPORT = "[[State funding and grant numbers here, for example “This work was supported by …”. If there was no funding, delete this paragraph.]]"

ABSTRACT = (
    "Federated learning promises collaborative model training across hospitals without sharing images, but its "
    "benefit is uncertain when participants hold data from different medical specialties. We benchmarked five "
    "training strategies for a frozen ImageNet-pretrained ResNet-18 with a shared low-rank adapter and "
    "client-specific classifiers: centralized training, local-only training, FedAvg, FedProx, and FedSAA, a "
    "similarity-weighted personalized aggregation rule proposed here. Two simulated federations were built from "
    "MedMNIST v2, each with 15 clients (five specialties, three clients each), and every strategy was run with "
    "three seeds for 40 communication rounds. In the first federation, FedAvg, FedProx, and FedSAA reached 67.5%, "
    "67.7%, and 67.4% mean client accuracy, below local-only (70.1%) and centralized training (70.9%); in the "
    "second, all five strategies fell within 1.8 percentage points. FedSAA did not improve on FedAvg in either "
    "federation and had lower macro-F1. The effect of federation was specialty-dependent: relative to local-only "
    "training, FedAvg gained 1.0–4.8 points in five specialties and lost 1.4–8.1 points in the other five. "
    "Federated runs were still improving at round 40, so these are fixed-budget results. With three seeds, no "
    "accuracy difference survived correction for multiple comparisons. Code, raw results, and analysis scripts "
    "are released."
)

INDEX_TERMS = ("Benchmark testing, data heterogeneity, federated learning, low-rank adaptation, medical image "
               "classification, negative transfer, parameter-efficient fine-tuning, personalized federated learning.")

# Each item: (kind, payload...)
#   ("h1", text)            numbered section heading
#   ("h1u", text)           unnumbered heading
#   ("h2", text)            lettered subsection heading (restarts in every section)
#   ("p", text)             paragraph (first after a heading is not indented)
#   ("eq", index)           numbered display equation, index into EQUATIONS
#   ("fig", file, caption)  single-column figure
#   ("table", key)          table defined in TABLES
BODY = [
    ("h1", "INTRODUCTION"),
    ("p", "Medical images are difficult to pool across institutions because of privacy regulation, governance, and "
          "data ownership. Federated learning (FL) addresses this by training a shared model while the images stay "
          "where they were acquired [1]–[4]. In its canonical form, federated averaging (FedAvg), each client trains "
          "locally for a few epochs and a server averages the resulting parameters [5]."),
    ("p", "The main obstacle to FL in practice is statistical heterogeneity. When client distributions differ, local "
          "updates drift apart and the averaged model can converge slowly or to a worse solution [6]–[10]. Most "
          "medical FL studies consider one clinical task observed at several sites, where heterogeneity comes from "
          "scanners, protocols, and populations [11]. A harder case arises when the participants are specialty "
          "departments or single-specialty centers. A pathology laboratory, a dermatology clinic, and a radiology "
          "department do not share an imaging modality, a label space, or a task. The only thing they could usefully "
          "share is a representation."),
    ("p", "Parameter-efficient fine-tuning makes that kind of sharing cheap. A pretrained backbone is frozen and only "
          "a small adapter, such as a low-rank adaptation (LoRA) module [12], is trained and exchanged, while each "
          "client keeps its own classifier. Whether sharing an adapter across specialties helps is an empirical "
          "question. Adapters trained on different specialties might capture common structure in medical images, or "
          "they might pull the shared parameters in incompatible directions, the phenomenon known as negative "
          "transfer [13], [14]."),
    ("p", "Here we test this directly. We build two simulated federations from MedMNIST v2 [15], each with five "
          "specialties and three clients per specialty, and train a shared low-rank adapter on a frozen ResNet-18 "
          "[16] under five strategies: centralized training, local-only training, FedAvg, FedProx, and FedSAA "
          "(Specialty-Aware Adapter Aggregation). FedSAA is a personalized aggregation rule that we designed for "
          "this setting. It weights other clients’ adapter updates by their similarity to a client’s own update, so "
          "that clients of the same specialty can collaborate more closely than clients of different specialties."),
    ("p", "The outcome is negative for the proposed method and mixed for federation as a whole, and we report it as "
          "such. The study makes four contributions. First, it defines a reproducible cross-specialty benchmark "
          "built entirely from public data. Second, it compares five strategies with per-specialty, per-client, and "
          "convergence analyses, and with statistics that are explicit about what three seeds can support. Third, "
          "it specifies FedSAA fully and identifies, in an exploratory sensitivity analysis, the component most "
          "likely responsible for its underperformance. Fourth, it releases the code, the raw per-run results, and "
          "the scripts that regenerate every table and figure. The clients are simulated from public datasets; "
          "nothing here is a clinical validation or a real multi-institution deployment."),
    ("p", "The rest of this article is organized as follows. Section II reviews related work. Section III describes "
          "the data, model, training strategies, and statistical analysis. Section IV reports the results. Section V "
          "discusses them and their limitations, and Section VI concludes."),

    ("h1", "RELATED WORK"),
    ("h2", "FEDERATED BENCHMARKS IN MEDICAL IMAGING"),
    ("p", "FLamby provides cross-silo healthcare datasets with natural site splits and baseline results for standard "
          "FL algorithms [11]. Benchmarks built from MedMNIST usually partition a single dataset among clients with "
          "label skew. Two recent preprints instead assemble federations across imaging modalities from MedMNIST "
          "subsets: OmniFM evaluates a modality-robust framework on a cross-modality classification task [17], and "
          "FM^2^ introduces a benchmark that includes a configuration in which each client holds a single, "
          "non-overlapping modality [18]. Both propose new methods. Our study keeps the model deliberately simple, "
          "includes local-only and centralized reference points, and reports where federation does not help."),
    ("h2", "HETEROGENEITY AND PERSONALIZATION"),
    ("p", "FedProx adds a proximal term that keeps local models near the global model [8], and SCAFFOLD corrects "
          "client drift with control variates [7]. Personalized FL keeps part of the model local, for example the "
          "classifier [19], [20] or the normalization layers [21], or trains a regularized personal model beside the "
          "global one [22]. A related family lets similar clients collaborate more: FedAMP uses attention-based "
          "message passing between clients [23], clustered FL groups clients by the similarity of their updates or "
          "losses [24], [25], and FedFomo weights other clients’ models by their usefulness on local data [26]; a "
          "survey is given in [27]. Benchmarks of personalized FL find that no method wins consistently across "
          "datasets [28], [29], and local adaptation can be needed before a federated model is useful to an "
          "individual participant [30]."),
    ("h2", "FEDERATED LOW-RANK ADAPTATION"),
    ("p", "FedIT averages the LoRA factors *A* and *B* separately across clients [31]. Averaging factors is not "
          "equivalent to averaging their product, a discrepancy that FLoRA removes by stacking client modules [32]. "
          "FFA-LoRA freezes *A* and trains only *B* [33]; FedSA-LoRA shares *A* and keeps *B* local [34]; HetLoRA "
          "supports different ranks across clients [35]; and FlexLoRA averages full-size updates and redistributes "
          "them by singular value decomposition (SVD) [36]. In medical imaging, FedOPAL learns orthogonal global and "
          "personal adapters for vision foundation models [37]. FedSAA combines two of these ideas, product-space "
          "aggregation with SVD redistribution [36] and similarity-based attention between clients [23], and adds a "
          "step that retains part of each client’s own adapter."),
    ("h2", "REPORTING COMPARISONS"),
    ("p", "Differences between learning algorithms can be smaller than the variance from random seeds and data "
          "sampling, and conclusions drawn from a few runs are fragile [38], [39]. We therefore report every seed, "
          "give confidence intervals, and correct for multiple comparisons."),

    ("h1", "METHODS"),
    ("h2", "DATA AND CLIENT CONSTRUCTION"),
    ("p", "All images come from MedMNIST v2 [15]. Each federation, called Preset A and Preset B, combines five "
          "two-dimensional subsets that stand for five specialties (Table I). The official training and test splits "
          "of each subset were divided among three simulated hospitals, giving 15 clients per federation, and each "
          "client was capped at 2,000 training and 1,000 test images. [[AUTHORS: state how each subset is "
          "partitioned among its three clients, whether the partition depends on the seed, and the training-set "
          "size of every client.]] Images were resized to 64 × 64 pixels, converted to three channels, and "
          "normalized. Label spaces differ between specialties; the five ordinal grades of RetinaMNIST were treated "
          "as five classes. MedMNIST v2 is released under CC BY 4.0, except DermaMNIST (CC BY-NC 4.0). The study "
          "used only public, de-identified data and involved no new data from human participants."),
    ("table", "composition"),
    ("h2", "MODEL"),
    ("p", "The backbone is a ResNet-18 [16] pretrained on ImageNet [40] and truncated after its third residual "
          "stage. Global average pooling gives a feature vector *x* of dimension *d* = 256. All backbone parameters "
          "are frozen. A low-rank residual adapter transforms the feature as"),
    ("eq", 0),
    ("pc", "where *A* is an *r* × *d* matrix, *B* is a *d* × *r* matrix, the rank is *r* = 16, and the scaling is "
           "*α* = 32. The adapter has 2*rd* = 8,192 parameters, which is all that is communicated. Each client has "
           "its own linear classifier on *h*(*x*); classifiers are never aggregated. Because the adapter and the "
           "classifier are both linear, their composition is a linear function of *x*. [[AUTHORS: state how *A* "
           "and *B* are initialized, and confirm that no activation, dropout, or normalization follows the "
           "adapter.]]"),
    ("h2", "TRAINING STRATEGIES"),
    ("p", "All strategies used the Adam optimizer [41] with a learning rate of 0.001, a batch size of 64, and "
          "cross-entropy loss. One round is three local epochs, and each run lasted 40 rounds. The five strategies "
          "are as follows."),
    ("p", "*Local-only:* Each client trains its own adapter and classifier with no communication."),
    ("p", "*FedAvg* [5]: After each round the server replaces every client’s *A* and *B* with the "
          "sample-size-weighted average of all clients’ factors."),
    ("p", "*FedProx* [8]: As FedAvg, with a proximal penalty of weight *μ* = 0.01 on the distance between the local "
          "and global adapter parameters."),
    ("p", "*FedSAA:* The server computes a personalized adapter for every client, as described in Section III-D."),
    ("p", "*Centralized:* One adapter and one classifier per specialty are trained on the pooled data of all "
          "clients, with batches shuffled across specialties. [[AUTHORS: confirm the centralized schedule, that all "
          "15 clients participate in every round, and whether the optimizer state is reset each round.]]"),
    ("h2", "FEDSAA AGGREGATION"),
    ("p", "After local training in each round, the server holds the factors of all *N* = 15 clients. For client *i* "
          "it forms the update"),
    ("eq", 1),
    ("pc", "and its cosine similarity to the update of every client *j*, including *j* = *i*,"),
    ("eq", 2),
    ("pc", "where the inner product and norms are Frobenius. The similarities are converted to attention weights "
           "with temperature *τ*,"),
    ("eq", 3),
    ("pc", "which give a personalized aggregate and, for comparison, an unweighted global mean:"),
    ("eq", 4),
    ("pc", "The two are blended with weight *λ* on the global mean,"),
    ("eq", 5),
    ("pc", "and the blend is reduced to rank *r* by a truncated SVD that keeps the *r* largest singular values,"),
    ("eq", 6),
    ("pc", "from which new factors *A*′~*i*~ and *B*′~*i*~ are formed whose product equals the truncation. Finally, "
           "a retention step interpolates the new factors with the client’s own locally trained factors:"),
    ("eq", 7),
    ("eq", 8),
    ("pc", "Here *ρ* is the retention coefficient; it is distinct from the adapter scaling *α*. The defaults were "
           "*τ* = 0.5, *λ* = 0.5, and *ρ* = 0.3."),
    ("p", "FedSAA does not reduce to FedAvg. With *λ* = 1 and *ρ* = 0, every client receives the rank-*r* truncation "
          "of the unweighted mean of the products *BA*, as in FlexLoRA [36]. FedAvg averages *A* and *B* separately "
          "with sample-size weights, and the product of averaged factors differs from the average of products [32]."),
    ("h2", "EVALUATION"),
    ("p", "After every round each client’s model was evaluated on that client’s test set. We computed accuracy, "
          "macro-F1, and one-versus-rest area under the receiver operating characteristic curve (AUC) per client, "
          "and then averaged over the 15 clients with equal weight. The worst-client accuracy is the minimum over "
          "clients. Per-specialty accuracy is the mean over the three clients of a specialty. All reported results "
          "are from round 40. Per-round curves are descriptive only and were not used to select models or "
          "hyperparameters."),
    ("h2", "STATISTICAL ANALYSIS"),
    ("p", "The unit of analysis is the seed, with three runs per strategy per federation. Results are summarized as "
          "the mean ± sample standard deviation (SD), and every per-seed value is reported. Strategies were compared "
          "on the per-seed differences with two-sided paired *t*-tests (two degrees of freedom) at a significance "
          "level of 0.05, with 95% confidence intervals (CIs) from the *t* distribution. Within each federation and "
          "metric, the *p* values of the ten pairwise comparisons were adjusted by Holm’s method [42]."),
    ("p", "Normality of the differences cannot be assessed with three observations, and the exact sign-flip test "
          "has a minimum two-sided *p* value of 0.25 at this sample size. The *p* values are therefore reported as "
          "descriptive measures, and no claim rests on a significance threshold alone. Sampling error from finite "
          "test sets was approximated by the binomial standard error of the client-averaged accuracy. The "
          "sensitivity analysis used one seed and was not tested. No runs were excluded."),
    ("h2", "IMPLEMENTATION"),
    ("p", "Experiments were implemented in PyTorch [43] and run on one graphics processing unit; the 30 runs took "
          "73 minutes in total (50–236 s per run). Tables, statistics, and figures were produced from the raw "
          "per-run records with NumPy and SciPy [44]. Table II lists the configuration. [[AUTHORS: add the versions "
          "of PyTorch, torchvision, and medmnist, and the GPU model.]]"),
    ("table", "config"),

    ("h1", "RESULTS"),
    ("h2", "FEDERATED TRAINING DID NOT OUTPERFORM LOCAL-ONLY TRAINING"),
    ("p", "In Preset A the three federated strategies reached 67.4–67.7% accuracy, below local-only training (70.1%) "
          "and centralized training (70.9%), as Table III shows. Local-only exceeded FedAvg by 2.58 points (95% CI "
          "0.03 to 5.13), FedProx by 2.38 points (−0.01 to 4.78), and FedSAA by 2.73 points (1.63 to 3.83), with the "
          "same direction in all three seeds (Table IV). The unadjusted *p* values were 0.049, 0.050, and 0.009. "
          "After Holm correction across the ten pairwise comparisons they were 0.29, 0.29, and 0.087."),
    ("p", "In Preset B the five strategies spanned 77.9–79.6% and could not be separated. FedAvg and FedProx were "
          "nominally 0.45 and 0.24 points above local-only, with confidence intervals of roughly ±3 points."),
    ("table_wide", "main"),
    ("p", "Macro-F1 was lower for every federated strategy than for local-only training in both presets (A: "
          "0.509–0.545 versus 0.584; B: 0.681–0.695 versus 0.729). In Preset B, local-only also had higher AUC than "
          "FedAvg (difference 0.017, 95% CI 0.014 to 0.021, adjusted *p* = 0.026) and FedProx (0.018, 0.013 to "
          "0.023, adjusted *p* = 0.035). The lowest-accuracy client fared worse under federation: 46.8–47.8% against "
          "49.8% for local-only in Preset A, and 40.6–42.4% against 47.4% in Preset B. That client was a RetinaMNIST "
          "client in all 15 Preset B runs and a TissueMNIST client in 14 of 15 Preset A runs."),
    ("p", "Centralized training was not a reliable upper bound. It exceeded local-only by 0.85 points in Preset A "
          "(95% CI −1.74 to 3.43) and by 1.15 points in Preset B (−4.43 to 6.73)."),
    ("table", "contrasts"),
    ("h2", "FEDSAA DID NOT IMPROVE ON FEDAVG"),
    ("p", "FedSAA was 0.15 points below FedAvg in Preset A (95% CI −1.82 to 1.51) and 1.05 points below in Preset B "
          "(−2.61 to 0.50). Its macro-F1 was lower than that of FedAvg by 0.034 in Preset A (0.018 to 0.050; "
          "*p* = 0.012, adjusted 0.069) and by 0.015 in Preset B (−0.012 to 0.042). Its AUC was lower than that of "
          "FedAvg in Preset A (0.908 versus 0.915) and higher in Preset B (0.900 versus 0.892). Against local-only "
          "training in Preset A, FedSAA had lower macro-F1 (difference 0.075, adjusted *p* = 0.020) and lower AUC "
          "(0.004, adjusted *p* = 0.024)."),
    ("p", "FedSAA varied least across seeds in Preset A (SD 0.17 against 0.84 for FedAvg), but the order reversed in "
          "Preset B (0.65 against 0.05). A standard deviation estimated from three runs is too imprecise to support "
          "a claim about stability."),
    ("h2", "THE EFFECT OF FEDERATION DEPENDS ON THE SPECIALTY"),
    ("p", "Averaging over clients hides opposite effects (Fig. 1). Relative to local-only training, FedAvg gained "
          "accuracy in five specialties: PathMNIST (+1.0), DermaMNIST (+1.0), OrganAMNIST (+1.4), OrganCMNIST "
          "(+1.8), and PneumoniaMNIST (+4.8). It lost accuracy in the other five: TissueMNIST (−1.4), BreastMNIST "
          "(−1.5), RetinaMNIST (−4.3), BloodMNIST (−5.5), and OCTMNIST (−8.1). FedProx followed FedAvg closely."),
    ("fig", "fig1.png",
     "Accuracy difference from local-only training at round 40 for each specialty. Diamonds are means over three "
     "seeds and dots are individual seeds. The accuracy of a specialty is the mean over its three clients. The "
     "letter in parentheses gives the federation."),
    ("p", "FedSAA stayed within one point of local-only in seven specialties (−0.6 to +0.9) and fell 3.0–9.7 points "
          "below it in RetinaMNIST, BloodMNIST, and OCTMNIST, the three specialties in which FedAvg lost most. It "
          "therefore gave up most of the gains of FedAvg without avoiding its largest losses."),
    ("p", "At the client level, FedAvg exceeded local-only for 6 of 15 clients in Preset A and 10 of 15 in Preset B; "
          "FedSAA did so for 4 and 7. The largest single-client deficit was about 10 points. Centralized training "
          "exceeded local-only in eight of ten specialties, the exceptions being OCTMNIST (−1.9) and RetinaMNIST "
          "(−1.3). BreastMNIST estimates rest on 52 test images per client and are the least precise."),
    ("h2", "FEDERATED RUNS HAD NOT CONVERGED AT ROUND 40"),
    ("p", "Local-only training came within one point of its final accuracy by round 7 in Preset A and round 8 in "
          "Preset B (Fig. 2). Its mean curve peaked at round 22 in both presets (71.4% and 79.8%) and then drifted "
          "down, by 0.8 and 0.5 points between rounds 30 and 40. The federated strategies were still rising when "
          "training stopped. Over the last ten rounds FedAvg gained 1.1 points in Preset A and 0.4 in Preset B; "
          "FedSAA gained 1.8 and 1.9."),
    ("fig", "fig2.png",
     "Client-averaged test accuracy by communication round. Lines are means over three seeds and bands are ±1 SD. "
     "One round is three local epochs."),
    ("p", "The gap between local-only and FedAvg in Preset A narrowed from 8.6 points at round 10 to 6.0 at round "
          "20, 4.5 at round 30, and 2.6 at round 40. FedSAA trailed FedAvg at every checkpoint, by 6.2 points at "
          "round 10 in Preset B and by 1.1 at round 40. The ranking in Table III is therefore specific to a budget "
          "of 40 rounds of three local epochs. These data do not show where the federated strategies would plateau."),
    ("h2", "SENSITIVITY ANALYSIS POINTS TO THE RETENTION STEP"),
    ("p", "The hyperparameters of FedSAA were varied one at a time around their defaults in Preset A with seed 0 "
          "(Table V). Only summary values of these runs were archived, so this analysis is exploratory. Adapter rank "
          "had no visible effect (67.3–67.9% for ranks 2 to 32), and the temperature changed accuracy by at most 1.3 "
          "points. Accuracy was lowest when the personalized term was removed (*λ* = 1; 65.4%)."),
    ("p", "The clearest signal came from the retention coefficient *ρ*. With no retention (*ρ* = 0) FedSAA reached "
          "69.8%, against 66.4–68.3% for *ρ* between 0.1 and 0.7. This was the only setting that exceeded FedAvg on "
          "the same seed (68.5%), and it approached local-only training (70.0%). The main experiments used "
          "*ρ* = 0.3, which was fixed before this analysis."),
    ("table", "sensitivity"),
    ("h2", "PRECISION OF THE ESTIMATES"),
    ("p", "Two sources of noise limit these comparisons. First, with three seeds the 95% confidence interval "
          "multiplies the standard error by 4.30, and an exact sign-flip test on three paired differences cannot "
          "return a two-sided *p* value below 0.25. The *t*-test *p* values therefore depend on a normality "
          "assumption that three observations cannot check. After Holm correction no accuracy contrast in either "
          "preset had an adjusted *p* value below 0.05; the smallest was 0.087 (local-only versus FedSAA, Preset A)."),
    ("p", "Second, test sets are finite. The binomial standard error of the client-averaged accuracy is about 0.45 "
          "points in Preset A and 0.85 points in Preset B, where six of the 15 clients have fewer than 140 test "
          "images."),

    ("h1", "DISCUSSION"),
    ("h2", "WHAT THE RESULTS SHOW"),
    ("p", "In two simulated cross-specialty federations, sharing a low-rank adapter on a frozen backbone gave no "
          "reliable accuracy benefit over training each client alone within 40 rounds. In one federation the "
          "federated strategies were 2.4–2.7 points below local-only training; in the other the strategies were "
          "indistinguishable. FedSAA, the personalized aggregation rule we proposed, did not improve on FedAvg in "
          "either federation and had lower macro-F1."),
    ("p", "The per-specialty results qualify the phrase negative transfer. Federation did not harm every "
          "participant. It raised accuracy in five specialties and lowered it in five, and the federation-level "
          "average was the net of those opposite effects. The average is therefore a property of the mix of "
          "specialties and not of federated learning as such. For a department deciding whether to join, the "
          "relevant quantity is the change for its own specialty, and here half the specialties would have done "
          "better alone at this training budget."),
    ("p", "We have no evidence for why particular specialties gained or lost. The pattern does not follow image "
          "color or dataset size in any simple way: optical coherence tomography and abdominal computed tomography "
          "are both grayscale, yet the first lost most and the second gained, and the small PneumoniaMNIST clients "
          "gained most while the small RetinaMNIST clients lost."),
    ("h2", "WHY FEDSAA DID NOT HELP"),
    ("p", "Three features of the design of FedSAA could explain its result. First, the softmax weighting is "
          "diffuse. The similarity of a client to its own update is 1, so with *τ* = 0.5 and 14 peers whose updates "
          "were orthogonal to its own, a client would still assign 65% of its personalized aggregate to those "
          "unrelated peers, and 79% after blending with the global mean. The weighting therefore cannot isolate a "
          "client from dissimilar peers."),
    ("p", "Second, the blended update is truncated to rank 16 every round, although a mixture of 15 rank-16 updates "
          "can have rank up to 240."),
    ("p", "Third, the retention step interpolates factors that are not expressed in the same basis. A low-rank "
          "factorization is unique only up to an invertible transformation, so the factors returned by the SVD and "
          "the locally trained factors of the client are not aligned, and interpolating them produces cross terms "
          "that pair mismatched bases. The sensitivity analysis is consistent with this explanation, because "
          "removing retention gave the highest accuracy, but it rests on a single seed. If it is confirmed, "
          "retention should be applied to the product *BA* before a single factorization. The lower macro-F1 of "
          "FedSAA at similar accuracy also suggests that it performed worse on minority classes."),
    ("h2", "THE MODEL CLASS LIMITS THE INTERPRETATION"),
    ("p", "The adapter is a linear map and the classifier is linear, so their composition is linear in the frozen "
          "feature. The set of decision functions that each client can reach is then the same as for a linear probe "
          "on the frozen features [45], with or without the adapter. Sharing the adapter cannot give a client "
          "representational capacity that its own classifier lacks. The differences we measured must come from "
          "optimization: aggregation changes the shared adapter every round, and the classifier of each client has "
          "to follow it. This reading is consistent with the slower convergence of every federated strategy and "
          "with the absence of any effect of adapter rank."),
    ("p", "It also means that the study says nothing direct about LoRA modules placed inside the layers of a "
          "backbone, where adapters alter nonlinear features. The negative transfer observed here is interference "
          "in the optimization of a shared linear reparameterization. It is not evidence that the specialties lack "
          "shared structure."),
    ("p", "The two reference strategies are not bounds. Centralized training used one classifier per specialty "
          "trained on the pooled data of three clients, so its small and uncertain advantage over local-only "
          "training may come from pooling classifier data and not from the shared adapter. The federated strategies "
          "never shared classifiers, even among clients of the same specialty. A design that shares classifiers "
          "within a specialty is an obvious comparator that we did not test."),
    ("h2", "LIMITATIONS"),
    ("p", "The federated runs had not converged, so the ranking holds for one training budget only. Three seeds give "
          "wide intervals, and several clients have small test sets. The clients are simulated by partitioning "
          "public datasets, so clients of the same specialty do not differ by scanner, protocol, or population as "
          "real hospitals would. Images were 64 × 64 pixels and training sets were capped at 2,000 images per "
          "client. We used one backbone, one adapter placement, and one learning rate for all strategies, without "
          "tuning each strategy separately. The defaults of FedSAA were fixed in advance, and the sensitivity "
          "analysis suggests they were not the best choice. We did not study privacy mechanisms or communication "
          "cost. Nothing here constitutes clinical validation."),
    ("h2", "PRACTICAL IMPLICATIONS"),
    ("p", "Three practical points follow. Evaluations of federated learning across specialties should report results "
          "for each participant against a local-only baseline, because the average can conceal which participants "
          "lose. Convergence curves should be reported alongside final accuracy, because a fixed budget favors "
          "strategies that converge quickly. And similarity-weighted aggregation of low-rank updates should not be "
          "assumed to help: in this benchmark it preserved the losses of plain averaging and removed most of its "
          "gains."),

    ("h1", "CONCLUSION"),
    ("p", "We benchmarked five strategies for training a shared low-rank adapter on a frozen backbone across two "
          "simulated federations of medical imaging specialties. Within 40 communication rounds, federated training "
          "did not reliably outperform local-only training, its effect differed in sign between specialties, and "
          "the proposed FedSAA rule did not improve on FedAvg. The federated strategies had not converged, and "
          "three seeds do not support strong statistical claims, so these are fixed-budget findings. Future work "
          "should train to convergence with more seeds, apply retention in product space, place adapters inside the "
          "backbone of a medical foundation model, and test classifier sharing within specialty groups."),

    ("h1u", "DATA AND CODE AVAILABILITY"),
    ("p", "MedMNIST v2 is publicly available at https://medmnist.com. The training script, the raw per-run results "
          "(30 records with per-client, per-specialty, and per-round metrics), and the analysis scripts that "
          "regenerate every table and figure are available at [[repository URL and DOI]]."),

    ("h1u", "ACKNOWLEDGMENT"),
    ("p", "[[AUTHORS: edit this statement so that it matches what was actually done; IEEE requires it.]] A generative "
          "artificial intelligence system (Claude, Anthropic) was used in preparing this article. It was used to "
          "audit the experimental outputs; to write the analysis scripts that compute the statistics, tables, and "
          "figures of Section IV from the raw results; and to draft and edit the text of all sections. The authors "
          "reviewed and verified all content and are responsible for it."),
]

EQUATIONS = [
    r"h(x) = x + \frac{\alpha}{r}\,BAx",
    r"U_i = B_i A_i",
    r"S_{ij} = \frac{\langle U_i, U_j\rangle}{\lVert U_i\rVert\,\lVert U_j\rVert}",
    r"w_{ij} = \frac{\exp(S_{ij}/\tau)}{\sum_{k=1}^{N}\exp(S_{ik}/\tau)}",
    r"\bar{U}_i = \sum_{j=1}^{N} w_{ij}U_j,\qquad G = \frac{1}{N}\sum_{j=1}^{N} U_j",
    r"\tilde{U}_i = (1-\lambda)\,\bar{U}_i + \lambda G",
    r"\tilde{U}_i \approx P_i \Sigma_i Q_i^{\top}",
    r"A_i \leftarrow (1-\rho)\,A_i' + \rho A_i",
    r"B_i \leftarrow (1-\rho)\,B_i' + \rho B_i",
]

# Tables. widths in twips (1 inch = 1440). One text column is about 4820 wide; the page is about 10040.
TABLES = {
    "composition": {
        "number": "I", "title": "Composition of the Two Federations",
        "widths": [1230, 1720, 520, 1230], "size": 15,
        "header": ["MedMNIST v2 subset", "Modality", "Classes", "Test images per client"],
        "align": ["left", "left", "center", "center"],
        "rows": [
            ("group", "Preset A"),
            ["PathMNIST", "Colon pathology", "9", "1,000 each"],
            ["DermaMNIST", "Dermatoscopy", "7", "669 / 668 / 668"],
            ["BloodMNIST", "Blood cell microscopy", "8", "1,000 each"],
            ["OCTMNIST", "Retinal OCT", "4", "334 / 333 / 333"],
            ["TissueMNIST", "Kidney cortex microscopy", "8", "1,000 each"],
            ("group", "Preset B"),
            ["OrganAMNIST", "Abdominal CT, axial", "11", "1,000 each"],
            ["PneumoniaMNIST", "Chest X-ray", "2", "208 each"],
            ["RetinaMNIST", "Fundus photography", "5", "134 / 133 / 133"],
            ["BreastMNIST", "Breast ultrasound", "2", "52 each"],
            ["OrganCMNIST", "Abdominal CT, coronal", "11", "1,000 each"],
        ],
        "note": "Each subset is divided among three clients. OCT = optical coherence tomography; CT = computed "
                "tomography. RetinaMNIST has five ordinal grades.",
    },
    "config": {
        "number": "II", "title": "Experimental Configuration",
        "widths": [2150, 2550], "size": 15,
        "header": ["Setting", "Value"],
        "align": ["left", "left"],
        "rows": [
            ["Backbone", "ResNet-18, ImageNet-pretrained, frozen"],
            ["Feature", "Third residual stage, 256-d"],
            ["Adapter rank *r* / scaling *α*", "16 / 32"],
            ["Clients per federation", "15 (5 specialties × 3 clients)"],
            ["Communication rounds", "40"],
            ["Local epochs per round", "3"],
            ["Optimizer / learning rate", "Adam / 0.001"],
            ["Batch size", "64"],
            ["Image size", "64 × 64"],
            ["Per-client cap, training / test", "2,000 / 1,000 images"],
            ["FedProx *μ*", "0.01"],
            ["FedSAA *λ* / *τ* / *ρ*", "0.5 / 0.5 / 0.3"],
            ["Seeds", "0, 1, 2"],
        ],
        "note": None,
    },
    "main": {
        "number": "III", "title": "Main Results at Round 40",
        "widths": [1300, 1500, 2300, 1500, 1700, 1500], "size": 16,
        "header": ["Strategy", "Accuracy (%)", "Accuracy by seed (0 / 1 / 2)", "Macro-F1", "AUC", "Worst client (%)"],
        "align": ["left", "center", "center", "center", "center", "center"],
        "rows": [
            ("group", "Preset A"),
            ["Centralized", "70.95 ± 0.67", "71.24 / 70.18 / 71.42", "0.578 ± 0.010", "0.9219 ± 0.0008", "51.23 ± 1.60"],
            ["Local-only", "70.10 ± 0.37", "69.96 / 70.52 / 69.82", "0.584 ± 0.004", "0.9126 ± 0.0013", "49.77 ± 1.68"],
            ["FedAvg", "67.52 ± 0.84", "68.49 / 67.02 / 67.06", "0.543 ± 0.007", "0.9148 ± 0.0007", "46.83 ± 2.48"],
            ["FedProx", "67.72 ± 0.71", "68.51 / 67.14 / 67.51", "0.545 ± 0.007", "0.9154 ± 0.0004", "46.90 ± 2.04"],
            ["FedSAA", "67.37 ± 0.17", "67.56 / 67.29 / 67.26", "0.509 ± 0.002", "0.9084 ± 0.0011", "47.77 ± 0.64"],
            ("group", "Preset B"),
            ["Centralized", "79.64 ± 1.37", "78.35 / 81.08 / 79.48", "0.726 ± 0.024", "0.9031 ± 0.0049", "44.61 ± 2.64"],
            ["Local-only", "78.48 ± 1.12", "78.54 / 77.34 / 79.58", "0.729 ± 0.014", "0.9090 ± 0.0045", "47.37 ± 2.71"],
            ["FedAvg", "78.94 ± 0.05", "78.99 / 78.93 / 78.89", "0.695 ± 0.001", "0.8917 ± 0.0053", "40.85 ± 4.83"],
            ["FedProx", "78.73 ± 0.06", "78.69 / 78.79 / 78.70", "0.692 ± 0.002", "0.8910 ± 0.0062", "40.60 ± 3.98"],
            ["FedSAA", "77.88 ± 0.65", "77.96 / 78.49 / 77.20", "0.681 ± 0.011", "0.8997 ± 0.0025", "42.36 ± 2.85"],
        ],
        "note": "Values are the mean ± sample SD over three seeds. Worst client is the lowest of the 15 client "
                "accuracies in a run.",
    },
    "contrasts": {
        "number": "IV", "title": "Paired Accuracy Contrasts Over Seeds",
        "widths": [1560, 620, 1120, 560, 760], "size": 15,
        "header": ["Contrast", "Diff.", "95% CI", "*p*", "Adj. *p*"],
        "align": ["left", "center", "center", "center", "center"],
        "rows": [
            ("group", "Preset A"),
            ["Local-only − FedAvg", "+2.58", "0.03 to 5.13", "0.049", "0.294"],
            ["Local-only − FedProx", "+2.38", "−0.01 to 4.78", "0.050", "0.294"],
            ["Local-only − FedSAA", "+2.73", "1.63 to 3.83", "0.009", "0.087"],
            ["Centralized − Local-only", "+0.85", "−1.74 to 3.43", "0.294", "1.000"],
            ["Centralized − FedAvg", "+3.42", "1.36 to 5.49", "0.019", "0.134"],
            ["FedAvg − FedProx", "−0.19", "−0.75 to 0.36", "0.269", "1.000"],
            ["FedAvg − FedSAA", "+0.15", "−1.51 to 1.82", "0.732", "1.000"],
            ("group", "Preset B"),
            ["Local-only − FedAvg", "−0.45", "−3.28 to 2.37", "0.562", "1.000"],
            ["Local-only − FedProx", "−0.24", "−3.16 to 2.67", "0.753", "1.000"],
            ["Local-only − FedSAA", "+0.60", "−3.78 to 4.98", "0.616", "1.000"],
            ["Centralized − Local-only", "+1.15", "−4.43 to 6.73", "0.467", "1.000"],
            ["Centralized − FedAvg", "+0.70", "−2.78 to 4.18", "0.478", "1.000"],
            ["FedAvg − FedProx", "+0.21", "0.00 to 0.42", "0.049", "0.494"],
            ["FedAvg − FedSAA", "+1.05", "−0.50 to 2.61", "0.101", "0.905"],
        ],
        "note": "Diff. is the first strategy minus the second, in percentage points. Two-sided paired *t*-test over "
                "three seeds. Adj. *p* is Holm-adjusted across all ten pairwise contrasts within a preset.",
    },
    "sensitivity": {
        "number": "V", "title": "One-at-a-Time Sensitivity of FedSAA (Preset A, Seed 0)",
        "widths": [1150, 1200, 1150, 1200], "size": 15,
        "header": ["*λ*", "Accuracy (%)", "*ρ*", "Accuracy (%)"],
        "align": ["center", "center", "center", "center"],
        "rows": [
            ["0", "66.92", "0", "69.78"],
            ["0.1", "67.52", "0.1", "68.34"],
            ["0.3", "66.47", "0.2", "66.37"],
            ["0.5", "67.56", "0.3", "67.56"],
            ["0.7", "67.51", "0.5", "67.12"],
            ["1.0", "65.44", "0.7", "67.91"],
            ("header", ["*r*", "Accuracy (%)", "*τ*", "Accuracy (%)"]),
            ["2", "67.42", "0.1", "68.31"],
            ["4", "67.26", "0.3", "67.04"],
            ["8", "67.87", "0.5", "67.56"],
            ["16", "67.56", "1.0", "67.50"],
            ["32", "67.89", "2.0", "67.50"],
        ],
        "note": "Defaults are *λ* = 0.5, *ρ* = 0.3, *r* = 16, and *τ* = 0.5, which give 67.56%. Same-seed "
                "references: FedAvg 68.49%, local-only 69.96%.",
    },
}

REFERENCES = [
    "N. Rieke et al., “The future of digital health with federated learning,” *npj Digit. Med.*, vol. 3, no. 1, Art. no. 119, 2020.",
    "G. A. Kaissis, M. R. Makowski, D. Rückert, and R. F. Braren, “Secure, privacy-preserving and federated machine learning in medical imaging,” *Nature Mach. Intell.*, vol. 2, no. 6, pp. 305–311, Jun. 2020.",
    "M. J. Sheller et al., “Federated learning in medicine: Facilitating multi-institutional collaborations without sharing patient data,” *Sci. Rep.*, vol. 10, no. 1, Art. no. 12598, 2020.",
    "I. Dayan et al., “Federated learning for predicting clinical outcomes in patients with COVID-19,” *Nature Med.*, vol. 27, no. 10, pp. 1735–1743, Oct. 2021.",
    "H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in *Proc. 20th Int. Conf. Artif. Intell. Statist. (AISTATS)*, 2017, pp. 1273–1282.",
    "Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, “Federated learning with non-IID data,” 2018, *arXiv:1806.00582*.",
    "S. P. Karimireddy, S. Kale, M. Mohri, S. J. Reddi, S. U. Stich, and A. T. Suresh, “SCAFFOLD: Stochastic controlled averaging for federated learning,” in *Proc. 37th Int. Conf. Mach. Learn. (ICML)*, 2020, pp. 5132–5143.",
    "T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in *Proc. Mach. Learn. Syst. (MLSys)*, vol. 2, 2020, pp. 429–450.",
    "Q. Li, Y. Diao, Q. Chen, and B. He, “Federated learning on non-IID data silos: An experimental study,” in *Proc. IEEE 38th Int. Conf. Data Eng. (ICDE)*, 2022, pp. 965–978.",
    "P. Kairouz et al., “Advances and open problems in federated learning,” *Found. Trends Mach. Learn.*, vol. 14, nos. 1–2, pp. 1–210, 2021.",
    "J. Ogier du Terrail et al., “FLamby: Datasets and benchmarks for cross-silo federated learning in realistic healthcare settings,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, 2022.",
    "E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.",
    "Z. Wang, Z. Dai, B. Póczos, and J. Carbonell, “Characterizing and avoiding negative transfer,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 11293–11302.",
    "T. Standley, A. R. Zamir, D. Chen, L. Guibas, J. Malik, and S. Savarese, “Which tasks should be learned together in multi-task learning?” in *Proc. 37th Int. Conf. Mach. Learn. (ICML)*, 2020, pp. 9120–9132.",
    "J. Yang et al., “MedMNIST v2—A large-scale lightweight benchmark for 2D and 3D biomedical image classification,” *Sci. Data*, vol. 10, no. 1, Art. no. 41, 2023.",
    "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.",
    "M. Liu, J. Wang, and J. Shan, “OmniFM: Toward modality-robust and task-agnostic federated learning for heterogeneous medical imaging,” 2026, *arXiv:2603.21660*.",
    "S. Chen and T. Shu, “FM^2^: Unified federated foundation models for heterogeneous multimodal medical imaging,” 2026, *arXiv:2607.13386*.",
    "M. G. Arivazhagan, V. Aggarwal, A. K. Singh, and S. Choudhary, “Federated learning with personalization layers,” 2019, *arXiv:1912.00818*.",
    "L. Collins, H. Hassani, A. Mokhtari, and S. Shakkottai, “Exploiting shared representations for personalized federated learning,” in *Proc. 38th Int. Conf. Mach. Learn. (ICML)*, 2021, pp. 2089–2099.",
    "X. Li, M. Jiang, X. Zhang, M. Kamp, and Q. Dou, “FedBN: Federated learning on non-IID features via local batch normalization,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.",
    "T. Li, S. Hu, A. Beirami, and V. Smith, “Ditto: Fair and robust federated learning through personalization,” in *Proc. 38th Int. Conf. Mach. Learn. (ICML)*, 2021, pp. 6357–6368.",
    "Y. Huang et al., “Personalized cross-silo federated learning on non-IID data,” in *Proc. AAAI Conf. Artif. Intell.*, vol. 35, no. 9, 2021, pp. 7865–7873.",
    "F. Sattler, K.-R. Müller, and W. Samek, “Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints,” *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 32, no. 8, pp. 3710–3722, Aug. 2021.",
    "A. Ghosh, J. Chung, D. Yin, and K. Ramchandran, “An efficient framework for clustered federated learning,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020, pp. 19586–19597.",
    "M. Zhang, K. Sapra, S. Fidler, S. Yeung, and J. M. Alvarez, “Personalized federated learning with first order model optimization,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.",
    "A. Z. Tan, H. Yu, L. Cui, and Q. Yang, “Towards personalized federated learning,” *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 34, no. 12, pp. 9587–9603, Dec. 2023.",
    "D. Chen, D. Gao, W. Kuang, Y. Li, and B. Ding, “pFL-Bench: A comprehensive benchmark for personalized federated learning,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, 2022.",
    "S. Wu et al., “Motley: Benchmarking heterogeneity and personalization in federated learning,” 2022, *arXiv:2206.09262*.",
    "T. Yu, E. Bagdasaryan, and V. Shmatikov, “Salvaging federated learning by local adaptation,” 2020, *arXiv:2002.04758*.",
    "J. Zhang et al., “Towards building the FederatedGPT: Federated instruction tuning,” in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, 2024, pp. 6915–6919.",
    "Z. Wang et al., “FLoRA: Federated fine-tuning large language models with heterogeneous low-rank adaptations,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 37, 2024.",
    "Y. Sun, Z. Li, Y. Li, and B. Ding, “Improving LoRA in privacy-preserving federated learning,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2024.",
    "P. Guo, S. Zeng, Y. Wang, H. Fan, F. Wang, and L. Qu, “Selective aggregation for low-rank adaptation in federated learning,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025.",
    "Y. J. Cho, L. Liu, Z. Xu, A. Fahrezi, and G. Joshi, “Heterogeneous LoRA for federated fine-tuning of on-device foundation models,” in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2024.",
    "J. Bai, D. Chen, B. Qian, L. Yao, and Y. Li, “Federated fine-tuning of large language models under heterogeneous tasks and client resources,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 37, 2024.",
    "A. Tupper and C. Gagné, “Personalized federated fine-tuning of vision foundation models for healthcare,” 2025, *arXiv:2510.12741*.",
    "X. Bouthillier et al., “Accounting for variance in machine learning benchmarks,” in *Proc. Mach. Learn. Syst. (MLSys)*, vol. 3, 2021, pp. 747–769.",
    "P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, “Deep reinforcement learning that matters,” in *Proc. AAAI Conf. Artif. Intell.*, vol. 32, no. 1, 2018, pp. 3207–3214.",
    "J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2009, pp. 248–255.",
    "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015.",
    "S. Holm, “A simple sequentially rejective multiple test procedure,” *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979.",
    "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 8024–8035.",
    "P. Virtanen et al., “SciPy 1.0: Fundamental algorithms for scientific computing in Python,” *Nature Methods*, vol. 17, no. 3, pp. 261–272, Mar. 2020.",
    "S. Kornblith, J. Shlens, and Q. V. Le, “Do better ImageNet models transfer better?” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 2661–2671.",
]

BIOS = [
    ("MOHAN BABU A",
     " [[received the … degree in … from …, City, Country, in 20XX. State the current position and institution, "
     "and research interests. IEEE Access requires a short biography for every author; add one paragraph per "
     "author and place each author’s photograph (1 in wide by 1.25 in high) beside it.]]"),
]
