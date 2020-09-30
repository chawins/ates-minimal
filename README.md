# Improving Adversarial Robustness Through Progressive Hardening

## Abstract
Adversarial training (AT) has become a popular choice for training robust networks. However, it tends to sacrifice clean accuracy heavily in favor of robustness, and with a large perturbation, it can cause models to learn a trivial solution, always predicting the same class. To address the above concerns, we propose Adversarial Training with Early Stopping (ATES), guided by principles from curriculum learning that emphasizes on starting "easy" and gradually ramping up on the "difficulty" of training. ATES is derived from our formulation for curriculum learning in the adversarial setting which introduces an additional curriculum constraint to the normal adversarial loss. To satisfy this constraint, we apply early stopping on the adversarial example generation step when a specified level of difficulty is reached. ATES stabilizes network training even for a large perturbation norm and allows the network to operate at a better clean accuracy versus robustness trade-off curve compared to AT. This leads to a significant improvement in both clean accuracy and robustness compared to AT, TRADES, and the other baselines.

## Requirements
- pytorch >= 1.4
- torchvision >= 0.5.0
- numpy == 1.18.1
- pyyaml == 5.3.1
- foolbox == 3.0.2
- numba == 0.48.0
- autoattack == 0.1

## Usage
- Edit the parameters listed in `minimal_test_aa.py` to choose the dataset (MNIST, CIFAR-10, CIFAR-100).
- Run `python minimal_test_aa.py`. The weight should be automatically downloaded. Otherwise, you can also manually download them using the links in `minimal_test_aa.py`.
- Result will be saved in the log file.
