# -*- coding: utf-8 -*-
"""Reference database for the IEEE Access manuscript: citation key -> IEEE-style entry.

The reference list in the document is ordered by first citation, automatically.
"""
import content as C

# keys of the 45 references of the short version, in their original numeric order
OLD_KEYS = [
    "rieke2020", "kaissis2020", "sheller2020", "dayan2021", "mcmahan2017", "zhao2018", "karimireddy2020",
    "li2020fedprox", "li2022silos", "kairouz2021", "flamby", "hu2022lora", "wang2019neg", "standley2020",
    "medmnist", "he2016", "omnifm", "fm2", "arivazhagan2019", "collins2021", "li2021fedbn", "li2021ditto",
    "huang2021fedamp", "sattler2021", "ghosh2020", "zhang2021fedfomo", "tan2023", "pflbench", "motley",
    "yu2020salvaging", "fedit", "flora", "ffalora", "fedsalora", "hetlora", "flexlora", "fedopal",
    "bouthillier2021", "henderson2018", "deng2009", "kingma2015", "holm1979", "paszke2019", "virtanen2020",
    "kornblith2019",
]
assert len(OLD_KEYS) == len(C.REFERENCES) == 45
REFS = dict(zip(OLD_KEYS, C.REFERENCES))

NIPS = "*Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*"
ICLR = "*Proc. Int. Conf. Learn. Represent. (ICLR)*"

REFS.update({
    "hsu2019": "T.-M. H. Hsu, H. Qi, and M. Brown, “Measuring the effects of non-identical data distribution for federated visual classification,” 2019, *arXiv:1909.06335*.",
    "li2020spm": "T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, “Federated learning: Challenges, methods, and future directions,” *IEEE Signal Process. Mag.*, vol. 37, no. 3, pp. 50–60, May 2020.",
    "wang2020fednova": f"J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, “Tackling the objective inconsistency problem in heterogeneous federated optimization,” in {NIPS}, vol. 33, 2020.",
    "smith2017": f"V. Smith, C.-K. Chiang, M. Sanjabi, and A. Talwalkar, “Federated multi-task learning,” in {NIPS}, vol. 30, 2017, pp. 4424–4434.",
    "fallah2020": f"A. Fallah, A. Mokhtari, and A. Ozdaglar, “Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach,” in {NIPS}, vol. 33, 2020.",
    "dinh2020": f"C. T. Dinh, N. H. Tran, and T. D. Nguyen, “Personalized federated learning with Moreau envelopes,” in {NIPS}, vol. 33, 2020.",
    "chen2022fedrod": f"H.-Y. Chen and W.-L. Chao, “On bridging generic and personalized federated learning for image classification,” in {ICLR}, 2022.",
    "houlsby2019": "N. Houlsby et al., “Parameter-efficient transfer learning for NLP,” in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 2790–2799.",
    "zhang2023fedpetuning": "Z. Zhang et al., “FedPETuning: When federated learning meets the parameter-efficient tuning methods of pre-trained language models,” in *Findings Assoc. Comput. Linguistics (ACL)*, 2023.",
    "nguyen2023": f"J. Nguyen, J. Wang, K. Malik, M. Sanjabi, and M. Rabbat, “Where to begin? On the impact of pre-training and initialization in federated learning,” in {ICLR}, 2023.",
    "pan2010": "S. J. Pan and Q. Yang, “A survey on transfer learning,” *IEEE Trans. Knowl. Data Eng.*, vol. 22, no. 10, pp. 1345–1359, Oct. 2010.",
    "yu2020pcgrad": f"T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, “Gradient surgery for multi-task learning,” in {NIPS}, vol. 33, 2020, pp. 5824–5836.",
    "raghu2019": f"M. Raghu, C. Zhang, J. Kleinberg, and S. Bengio, “Transfusion: Understanding transfer learning for medical imaging,” in {NIPS}, vol. 32, 2019.",
    "alain2016": "G. Alain and Y. Bengio, “Understanding intermediate layers using linear classifier probes,” 2016, *arXiv:1610.01644*.",
    "kumar2022": f"A. Kumar, A. Raghunathan, R. Jones, T. Ma, and P. Liang, “Fine-tuning can distort pretrained features and underperform out-of-distribution,” in {ICLR}, 2022.",
    "dietterich1998": "T. G. Dietterich, “Approximate statistical tests for comparing supervised classification learning algorithms,” *Neural Comput.*, vol. 10, no. 7, pp. 1895–1923, Oct. 1998.",
    "demsar2006": "J. Demšar, “Statistical comparisons of classifiers over multiple data sets,” *J. Mach. Learn. Res.*, vol. 7, pp. 1–30, Jan. 2006.",
    "wasserstein2016": "R. L. Wasserstein and N. A. Lazar, “The ASA statement on p-values: Context, process, and purpose,” *Amer. Statistician*, vol. 70, no. 2, pp. 129–133, 2016.",
    "eckart1936": "C. Eckart and G. Young, “The approximation of one matrix by another of lower rank,” *Psychometrika*, vol. 1, no. 3, pp. 211–218, Sep. 1936.",
    "kather2019": "J. N. Kather et al., “Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study,” *PLoS Med.*, vol. 16, no. 1, Art. no. e1002730, Jan. 2019.",
    "tschandl2018": "P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” *Sci. Data*, vol. 5, Art. no. 180161, 2018.",
    "kermany2018": "D. S. Kermany et al., “Identifying medical diagnoses and treatable diseases by image-based deep learning,” *Cell*, vol. 172, no. 5, pp. 1122–1131, Feb. 2018.",
    "acevedo2020": "A. Acevedo, A. Merino, S. Alférez, Á. Molina, L. Boldú, and J. Rodellar, “A dataset of microscopic peripheral blood cell images for development of automatic recognition systems,” *Data Brief*, vol. 30, Art. no. 105474, Jun. 2020.",
    "ljosa2012": "V. Ljosa, K. L. Sokolnicki, and A. E. Carpenter, “Annotated high-throughput microscopy image sets for validation,” *Nature Methods*, vol. 9, no. 7, p. 637, Jul. 2012.",
    "bilic2023": "P. Bilic et al., “The liver tumor segmentation benchmark (LiTS),” *Med. Image Anal.*, vol. 84, Art. no. 102680, Feb. 2023.",
    "liu2022deepdrid": "R. Liu et al., “DeepDRiD: Diabetic retinopathy—grading and image quality estimation challenge,” *Patterns*, vol. 3, no. 6, Art. no. 100512, Jun. 2022.",
    "aldhabyani2020": "W. Al-Dhabyani, M. Gomaa, H. Khaled, and A. Fahmy, “Dataset of breast ultrasound images,” *Data Brief*, vol. 28, Art. no. 104863, Feb. 2020.",
})
