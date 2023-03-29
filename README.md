# Robust Private Neural Network

## Literature Attack Methods
- Bandit
- NES
- Square attack: a query-efficient black-box adversarial attack via random search
  - [Paper](https://arxiv.org/abs/1912.00049), [Code](https://github.com/max-andr/square-attack)
- Simple Black-box Adversarial Attacks
  - [Paper](https://arxiv.org/abs/1905.07121), [Code](https://github.com/cg563/simple-blackbox-attack)
- Audit and Improve Robustness of Private Neural Networks on Encrypted Data
  - [Paper](https://arxiv.org/abs/2209.09996)
- signSGD via Zeroth-Order Oracle
  - [Paper](https://openreview.net/forum?id=BJe-DsC5Fm)

## Literature PNN Methods
- Security Protocols
  - FHE
    - Fully homomorphic encryption using ideal lattices
    - Allow executing any Boolean circuit over encrypted data
    - SEAL library
  - MPC
  - Garbled Circuits (GC)
  - Quantization
    - The networks on MNIST, CIFAR10, and medical datasets are quantized into 8 bits, 10 bits, and 16 bits respectively.
  - CryptoNets (ICML 2016)
    - Convert learned neural networks to CryptoNets
  - Delphi: A Cryptographic Inference Service for Neural Networks
    - [Paper](https://www.usenix.org/conference/usenixsecurity20/presentation/mishra), [Code](https://github.com/mc2-project/delphi)
- Constrained Nonlinearity (ONLY for MPC)
  - DeepReDuce: ReLU Reduction for Fast Private Inference
    - [Paper](https://arxiv.org/abs/2103.01396), Pretrained models available
  - CryptoNAS: Private Inference on a ReLU Budget
    - [Paper](https://arxiv.org/abs/2006.08733)
  - Selective Network Linearization for Efficient Private Inference
    - [Paper](https://proceedings.mlr.press/v162/cho22a.html), [Code](https://github.com/NYU-DICE-Lab/selective_network_linearization)
- Polynomial Approximations
  - On polynomial approximations for privacy-preserving and verifiable relu networks
    - https://arxiv.org/pdf/2011.05530.pdf
  - Sisyphus: A Cautionary Tale of Using Low-Degree Polynomial Activations in Privacy-Preserving Deep Learning
    - The 3rd Privacy-Preserving Machine Learning Workshop 2021 (PPML)
    - [Paper](https://arxiv.org/pdf/2107.12342.pdf), [Code](https://github.com/kvgarimella/sisyphus-ppml)
  - AESPA: Accuracy Preserving Low-degree Polynomial Activation for Fast Private Inference
    - https://arxiv.org/pdf/2201.06699.pdf
  - PolyKervNets: Activation-free Neural Networks For Efficient Private Inference (1st IEEE on SaTML 2023 Accepted)
    - https://openreview.net/forum?id=OGzt9NKC0lO
    - KNN (CVPR 2019)

## Objective
- Clean Accuracy*
- Larger ASR (Attack Success Rate)
- Fewer Queries
- Smaller $\ell_2$ perturbation norm: Imperceptible

### References
```bib
@article{xue2022audit,
  title={Audit and Improve Robustness of Private Neural Networks on Encrypted Data},
  author={Xue, Jiaqi and Xu, Lei and Chen, Lin and Shi, Weidong and Xu, Kaidi and Lou, Qian},
  journal={arXiv preprint arXiv:2209.09996},
  year={2022}
}

@inproceedings{guo2019simple,
  title={Simple black-box adversarial attacks},
  author={Guo, Chuan and Gardner, Jacob and You, Yurong and Wilson, Andrew Gordon and Weinberger, Kilian},
  booktitle={International Conference on Machine Learning},
  pages={2484--2493},
  year={2019},
  organization={PMLR}
}

@inproceedings{andriushchenko2020square,
  title={Square attack: a query-efficient black-box adversarial attack via random search},
  author={Andriushchenko, Maksym and Croce, Francesco and Flammarion, Nicolas and Hein, Matthias},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part XXIII},
  pages={484--501},
  year={2020},
  organization={Springer}
}

@inproceedings{jha2021deepreduce,
  title={DeepReDuce: Relu reduction for fast private inference},
  author={Jha, Nandan Kumar and Ghodsi, Zahra and Garg, Siddharth and Reagen, Brandon},
  booktitle={International Conference on Machine Learning},
  pages={4839--4849},
  year={2021},
  organization={PMLR}
}

@inproceedings {delphi,
  author = {Pratyush Mishra and Ryan Lehmkuhl and Akshayaram Srinivasan and Wenting Zheng and Raluca Ada Popa},
  title = {Delphi: A Cryptographic Inference Service for Neural Networks},
  booktitle = {29th USENIX Security Symposium (USENIX Security 20)},
  year = {2020},
  isbn = {978-1-939133-17-5},
  pages = {2505--2522},
  url = {https://www.usenix.org/conference/usenixsecurity20/presentation/mishra},
  publisher = {USENIX Association},
  month = aug,
}
```
