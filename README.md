# Improving Adversarial Robustness Through Progressive Hardening

## Abstract
Adversarial training (AT) has become a popular choice for training robust networks. However, it tends to sacrifice clean accuracy heavily in favor of robustness, and with a large perturbation, it can cause models to learn a trivial solution, always predicting the same class. To address the above concerns, we propose Adversarial Training with Early Stopping (ATES), guided by principles from curriculum learning that emphasizes on starting "easy" and gradually ramping up on the "difficulty" of training. ATES is derived from our formulation for curriculum learning in the adversarial setting which introduces an additional curriculum constraint to the normal adversarial loss. To satisfy this constraint, we apply early stopping on the adversarial example generation step when a specified level of difficulty is reached. ATES stabilizes network training even for a large perturbation norm and allows the network to operate at a better clean accuracy versus robustness trade-off curve compared to AT. This leads to a significant improvement in both clean accuracy and robustness compared to AT, TRADES, and the other baselines.

## Requirements
- pytorch == 1.4
- torchvision == 0.5.0
- numpy == 1.18.1
- pyyaml == 5.3.1
- foolbox == 3.0.2
- numba == 0.48.0

## File Organization
- There are multiple scripts to run the training and the testing. Main portion of code is in `./lib`.
- The naming of the scripts is simply `SCRIPT_DATASET.py` with the YAML config file under the same name. `DATASET` includes `mnist` and `cifar` which combines both CIFAR-10 and CIFAR-100.
- Scripts
  - `train_DATASET.py`: main script for training AT, TRADES, Dynamic AT and ATES models. The options and hyperparameters can be set in `train_DATASET.yml`.s
  - `test_DATASET.py`: test a trained network under one attack (PGD or BB).
  - `test_script_DATASET.py`: run the three attacks used in the evaluation (20-PGD, 100-PGD, BB) on a specified network.
  - `save_metrics_DATASET.py`: Compute the three difficulty scores (perturbation norm, the convergence score, and the ideal difficulty metric) a subset of training samples given a network's weights saved at the end of each epoch.
- Library
  - `lib/adv_model.py`: wrapper Pytorch Module for AT, TRADES, Dynamic AT and ATES.
  - `lib/pgd_attack.py`: implements PGD attack.
  - `lib/mnist_model.py`: implements MNIST models.
  - `lib/cifar10_model.py`: implements ResNet.
  - `lib/wideresnet.py`: implements WideResNet.
  - `lib/dataset_utils.py`: handles dataset loading.
  - `lib/utils.py`: other utility functions (e.g., quantization, get logger, etc.)
- External Code
  - CAT18: `./CAT18` (taken from https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT)
  - CAT20: `./CAT20` (obtained privately)
- Weights
  - MNIST: `exp1` means epsilon of 0.3, `exp2` means epsilon of 0.45.
  - CIFAR-10/100: `exp1` means ResNet, `exp2` means WideResNet.

## Usage
- We use YAML config files (.yml) for both training and testing.
- Training
  - To use ATES, set `early_stop: True`.
  - To use TRADES, set `loss_func: 'trades'`.
  - To use Dynamic AT, set `use_fosc: True`.
  - Other options are self-explanatory.
- Testing
  - Only need to specify name of the model to test.
  - To use BB attack, set `bb: True` (we use the default parameters which can be changed in the test scripts).
