# CaTs and DAGs 🐱🐶

### Integrating Directed Acyclic Graphs with Transformers and Fully-Connected Neural Networks for Causally Constrained Predictions

Vowels, M.J., Rochat, M., Akbari, S. (2024)  https://arxiv.org/abs/2410.14485

---

### Overview

Artificial Neural Networks (ANNs), including fully-connected networks and transformers, are known for their powerful function approximation capabilities. They have achieved remarkable success in various domains such as computer vision and natural language processing. However, one limitation of these models is their inability to inherently respect causal structures. This limitation often results in challenges like covariate shift and difficulties in model interpretability, which in turn can undermine their robustness in real-world applications.

In this repository, we introduce two novel model families:  
- **Causal Transformers (CaTs)**  
- **Causal Fully-Connected Neural Networks (CFCNs)**

These models are designed to operate under predefined causal constraints, as dictated by a Directed Acyclic Graph (DAG). By integrating causal structure directly into the architecture, these models combine the strengths of traditional neural networks with improved robustness, reliability, and interpretability.

---


### Citation

If you find this work useful, please cite:

Vowels, M.J., Rochat, M., and Akbari, S. (2024) CaTs and DAGs: Integrating Directed Acyclic Graphs with Transformers and Fully-Connected Neural Networks for Causally Constrained Predictions. arXiv preprint. arXiv:2410.14485


```
@misc{vowels2024catsdagsintegratingdirected,
      title={CaTs and DAGs: Integrating Directed Acyclic Graphs with Transformers and Fully-Connected Neural Networks for Causally Constrained Predictions}, 
      author={Matthew J. Vowels and Mathieu Rochat and Sina Akbari},
      year={2024},
      eprint={2410.14485},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.14485}, 
}
```