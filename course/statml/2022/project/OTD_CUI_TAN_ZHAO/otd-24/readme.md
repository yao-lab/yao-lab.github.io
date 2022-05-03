## Code for "Open Set Debiasing"
We mainly refer the code of Umang Gupta, Aaron Ferber, Bistra Dilkina, and Greg Ver Steeg. “Controllable Guarantees for Fair Outcomes via Contrastive Information Estimation.” In: Thirty-Fifth AAAI Conference on Artificial

- The model file for OTIS is located in `src/models/fcrl_model.py` and
  the trainer code is in `src/trainers/fcrl.py`.
  You can see the architectures in `src/arch/<data>/<data>_fcrl.py` where
  data is adult for UCI Adult and health for Heritage Health.
- Both trainer and model are loaded via `src/scripts/main.py`. Other models can
  also be run by calling `main.py`. See shell folder to get example commands
  for other methods.

