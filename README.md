# Robust Private Neural Network

## Literature Attack Methods
- Square attack: a query-efficient black-box adversarial attack via random search
  - [Paper](https://arxiv.org/abs/1912.00049), [Code](https://github.com/max-andr/square-attack)
- Simple Black-box Adversarial Attacks
  - [Paper](https://arxiv.org/abs/1905.07121), [Code](https://github.com/cg563/simple-blackbox-attack)
- Audit and Improve Robustness of Private Neural Networks on Encrypted Data
  - [Paper](https://arxiv.org/abs/2209.09996)

## Literature PNN Methods
- DeepReDuce [DeepReDuce: ReLU Reduction for Fast Private Inference]
  - [Paper](https://arxiv.org/abs/2103.01396), Pretrained models available
- Delphi
- Quantization
- The networks on MNIST, CIFAR10, and medical datasets are quantized into 8 bits, 10 bits,
and 16 bits respectively.

## Objective
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
```
