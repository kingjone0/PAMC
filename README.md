# PAMC: PPO-Driven Adaptive Metric Collaboration for Decentralized Personalized Federated Learning

Official implementation of **PAMC (PPO-Driven Adaptive Metric Collaboration)**, a decentralized personalized federated learning framework that adaptively learns collaboration preferences under partial and dynamically evolving client observations.

PAMC formulates decentralized personalized collaboration as a **Partially Observable Markov Decision Process (POMDP)**. Each client characterizes candidate collaborators using three complementary descriptors:

- **Model Similarity**
- **Feature Complementarity**
- **Logits Complementarity**

A PPO agent dynamically adjusts the relative importance of these descriptors. The resulting collaboration scores are used for staleness-aware stochastic neighbor selection, followed by self-retained personalized aggregation and local adaptation.

---

## Overview

In decentralized personalized federated learning (DPFL), clients communicate directly with peers without relying on a central server. Under non-IID data, however, the usefulness of different collaborators changes across clients and training stages, while each client only observes partial and potentially stale information.

PAMC addresses this problem through a closed-loop collaboration process:

1. **Observation Modeling**  
   Each client constructs its local observation from its learning state, cached neighbor information, and collaboration history.

2. **Multi-Descriptor Collaboration Modeling**  
   Candidate collaborators are characterized from parameter-, representation-, and prediction-level perspectives using model similarity, feature complementarity, and logits complementarity.

3. **PPO-Driven Metric Adaptation**  
   PPO outputs a simplex-constrained metric-weight vector that dynamically determines the relative importance of the three descriptors.

4. **Staleness-Aware Neighbor Selection**  
   Collaboration scores are adjusted according to cached-information staleness and converted into sampling probabilities for stochastic neighbor selection.

5. **Self-Retained Personalized Aggregation**  
   Selected models are weighted according to collaboration quality and local data support, while each client retains part of its own model to preserve personalization.

6. **Reward Feedback**  
   The collaboration utility obtained after aggregation is used as feedback for updating the PPO policy.

---

## Repository Structure

A typical project structure is shown below:

```text
DFL/
├── main_dbac7.py                  # Main training entry
│
├── fedfl/
│   ├── dbac7/
│   │   ├── dbac7.py               # PAMC training workflow
│   │   ├── client.py              # Client state, descriptors and collaboration
│   │   └── rlAgent.py             # Dirichlet policy and PPO agent
│   │
│   └── dbac5/
│       └── dfl_trainer.py         # Local model training and evaluation
│
├── model/
│   ├── cnn.py
│   ├── vgg.py
│   ├── lenet5.py
│   └── trainer.py
│
├── data/
│   └── dataprocess/
│       ├── cifar10/
│       ├── cifar100/
│       ├── mnist/
│       └── emnist/
│
├── utils/
│   ├── attack.py
│   └── slogits_visualizer.py
│
└── LOG/                           # Training logs
