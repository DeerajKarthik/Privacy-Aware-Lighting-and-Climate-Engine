# Grid design (fast mode; written before the controller grid was run)
**Core grid** (`configs/grid_fast.yaml`): sites {ac1, ac3, ac4, scone, tokyo, stair01} x seasons {summer, monsoon, winter; one representative week each, Surat 2019 weather replayed with TEST-split occupancy days} x seeds {0,1,2} x methods:
B0, B1, B2 (central), B3 (local), B4 (FedAvg), B5 (DP-FedAvg, eps_train in {0.5,2,8}), PALACE (DP-FedPer, eps_train in {0.5,2,8}; APF on at eps_act=8; preference posterior on).
**One-factor-at-a-time ablations** around the PALACE default (eps_train=2, eps_act=8): eps_act in {0.5, 2}; APF off (= eps_act=inf); personalisation off; simulator perturbation +-30 % (Rao, Ca, COP_ref, window area, DF jointly in the adverse/benign direction).
**Adversary logs**: validation-split replays of B1 and PALACE (APF off; eps_act 0.5, 2, 8) as adversary training data; test-split cells as adversary test data.
**Not run (cut for time, logged in DECISIONS D017/D021):** second climate (New Delhi data downloaded but unused), PPO baseline B6, clients 3/6/9 ablation, forecaster horizon ablation, k-anonymous rung R4, EnergyPlus/Sinergym, seeds 3-4 for DP/controller runs, full-year headline runs, dwell-time/top-K APF add-ons.
**Fast config**: 1 week per season; the PALACE hidden synthetic occupant per (site, seed) is shared by all methods (paired).
