
Warm-start the pixel SAC agent from a scripted driver, tune it once, pick one policy, then study how changes to the reward and the world change its driving (speed vs. recklessness). Everything runs through `sweep.py`; each step below is one or two commands. `TRAINING_METHODS.md` explains the methods behind each step.

## How the pieces fit

| File | Job |
| --- | --- |
| `environment.py` | The race: actions `[acceleration, steer, brake, drift, nitro]`, observations, reward. Every experiment setting is a named option (`DEFAULT_OPTIONS`), changed with `--env key=value`. |
| `SAC_CNN_Pixels.py` | The agent: CNN + GRU + actor, twin critics with target copies, automatic α, image augmentation. |
| `bc_pretrain.py` | Warm start: records the path follower, clones it into the pixel policy, refines it with DAgger. |
| `train.py` | Trains one run, optionally from a checkpoint (`--init-from`) with demonstrations in replay (`--demos`). Saves a checkpoint every 25k steps plus `final.pt`. |
| `evaluate.py` | Races a policy on 20 fixed seeds; one CSV row per race (finish, win, stuck, time, speed, off-road, steering jerk, yaw rate, crashes). |
| `sweep.py` | Runs each step of this plan: train → evaluate → summarize. Skips finished work, so after a stop you rerun the same command. |
| `render_agent.py` | 1920x1080 MP4 of a checkpoint's race. |

## Which policy each step uses

"Policy" means the pixel agent from `SAC_CNN_Pixels.py`, saved as a checkpoint. Evaluations and videos always drive with its mean action (no exploration noise).

| Step | Policy | Trained or only evaluated? |
| --- | --- | --- |
| 1. Sanity check | No learned policy: scripted behaviors (stand still, reverse, random actions, the path follower) | Neither: only scores the reward |
| 2. Smoke test | Tiny throwaway runs in `runs/quick/` | Trained + evaluated, then discarded |
| 3. Warm start | `runs/bc/policy.pt`: the path follower cloned into the pixel policy | Pretrained (supervised) + evaluated |
| 4. Phase 1 | One SAC policy per γ and seed in `runs/phase1/<gamma>_s<seed>/`, each **starting from** `runs/bc/policy.pt`. Evaluated at 50k, 100k and 150k steps, plus the `random` and `full_throttle` baselines | Trained + evaluated |
| 5. Pick | None trained: you choose **one checkpoint** (`FIXED_POLICY`) and its settings (`FIXED`) | Neither |
| 6. Phase 2a | One policy per change in `runs/phase2/<change>/`, each **fine-tuned from** `FIXED_POLICY` for 50k steps; `fixed` is fine-tuned with no change | Trained + evaluated, each in its own setting |
| 7. Phase 2b | `FIXED_POLICY` itself | Only evaluated, under each world change |
| 8. Videos | Any checkpoint, usually `FIXED_POLICY` and the phase 2a policies | Only rendered |

## The reward

| Term | Value | Option name |
| --- | --- | --- |
| Progress | +0.1 per meter on the road; off-road meters count half | `progress_reward`, `off_road_factor` |
| Time | −0.03 per step | `step_penalty` |
| Leading | +0.02 per step in 1st place | `lead_reward` |
| Finishing / winning | +5 for finishing, +15 more for finishing 1st | `finish_bonus`, `win_bonus` |
| Getting stuck | −10, ends the run (30 steps without progress) | `stuck_penalty` |

Off-road is a factor on progress (instead of a flat penalty per step) and getting stuck is a failure because otherwise standing still scored better than an untrained agent's driving, and the agent learned to park. `python sweep.py sanity` checks this after any reward change.

## Step 0: Set up (once per machine)

```bash
git fetch origin
git switch main
git pull
python -m venv .venv                 # first time only
```

Activate the virtual environment in **every new terminal**, before any other command:

```powershell
.venv\Scripts\Activate.ps1           # PowerShell
```

```bash
source .venv/Scripts/activate        # Git Bash
```

Then install and check the GPU:

```bash
pip install -r requirements.txt
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

- `GPU: False` usually means the venv isn't active: a bare `python` can be the system Python with a CPU-only PyTorch. `sweep.py` runs everything with the Python that started it, so without the venv every run silently trains on the CPU (about 6× slower).
- If it's `False` with the venv active on an NVIDIA machine, install the CUDA build of PyTorch from pytorch.org (RTX 50-series needs CUDA 12.8 or newer).
- If PowerShell won't run `Activate.ps1`, call the venv's Python directly instead: `.venv\Scripts\python.exe sweep.py ...`.
- Don't resize the game window while training: the frames change size and training crashes.

## Step 1: Check the reward (1 minute)

```bash
python sweep.py sanity
```

It must print `OK: quitting < random driving < a lap`. Rerun it whenever a reward weight changes.

## Step 2: Smoke test the pipeline (about 10 minutes)

```bash
python sweep.py bc --quick
python sweep.py phase1 --quick --only gamma_0.99,random --seeds 0
python sweep.py summary phase1 --quick
```

Tiny runs under `runs/quick/`, to confirm everything works on this machine before starting hours of training.

## Step 3: Warm start by behavior cloning (about 10 minutes)

```bash
python sweep.py bc
```

Records 20 path-follower races (with steering noise, so it also shows recoveries), trains the pixel policy to copy them, then runs 3 DAgger rounds: the student drives, the teacher labels every frame the student reaches, and the student retrains on everything. Saves `runs/bc/policy.pt` and `runs/bc/demos.npz`, and evaluates the clone as the `bc` row of phase 1's summary.

What to expect: the printout shows each recorded race. In testing, the student finished every DAgger race and every test race at the teacher's pace (about 600 steps, 60 s, against novices). If the `bc` row shows a low finish rate, rerun with more teacher races or DAgger rounds (the `BC` dictionary in `sweep.py`).

## Step 4: Phase 1, choose γ and training length (about 18 hours, easy to split)

Every run starts from the cloned policy, with the demonstrations in replay and 2,000 critic-only updates first. All use the novice opponent. Each γ runs with seeds 0 and 1, so luck shows up as the gap between seeds. Training length comes from each run's 50k/100k/150k checkpoints, with no extra runs.

| Runs | γ | Approx. time |
| --- | --- | --- |
| `gamma_0.98_s0`, `gamma_0.98_s1` | 0.98 (shorter horizon) | 3 h each |
| `gamma_0.99_s0`, `gamma_0.99_s1` | 0.99 (default) | 3 h each |
| `gamma_0.995_s0`, `gamma_0.995_s1` | 0.995 (longer horizon) | 3 h each |

```bash
python sweep.py phase1 --dry-run   # prints every command first
python sweep.py phase1
```

To split across three machines, give each person one γ. Everyone needs the same `runs/bc/`: run step 3 once and share that folder. Then copy everyone's `runs/eval/phase1/*.csv` into one folder before the summary.

```bash
python sweep.py phase1 --only gamma_0.98,random,full_throttle   # person A
python sweep.py phase1 --only gamma_0.99                        # person B
python sweep.py phase1 --only gamma_0.995                       # person C
```

Reward weights are not tuned here: they decide what "good driving" means, so they are phase 2's perturbations.

## Step 5: Pick the fixed policy

```bash
python sweep.py summary phase1
```

- **Choose γ** by averaging its two seeds: finish rate first, then win rate, then finish time. A difference smaller than the gap between seeds is luck.
- **Choose the length** from the 50k/100k/150k rows: the earliest checkpoint that's about as good as the best.
- **Pick one checkpoint** of that γ and length (the better seed), and write it and its settings into `sweep.py`, then commit:

```python
FIXED_POLICY = "runs/phase1/gamma_0.995_s1/step_100000.pt"
FIXED = {**BASE, "gamma": 0.995}
```

Share that checkpoint file with everyone running phase 2.

## Step 6: Phase 2a, fine-tune under each change (about 1 hour per run)

```bash
python sweep.py phase2-train
```

| Run | Change | Question |
| --- | --- | --- |
| `fixed` | none | The reference, with the same 50k steps of extra training as every change |
| `step_0.01`, `step_0.05` | `step_penalty` | Does time pressure make it faster, or more reckless? |
| `offroad_0`, `offroad_1` | `off_road_factor` | Does it cut corners when grass is cheap? |
| `win_45` | `win_bonus` | Does it take more risks when winning matters more? |
| `no_brakes` | `no_brakes=True` | How does it handle corners without brakes? |

Each run is evaluated in the setting it was fine-tuned in.

## Step 7: Phase 2b, robustness of the fixed policy (no training)

```bash
python sweep.py phase2-eval
```

| Evaluation | Change |
| --- | --- |
| `same` | none |
| `expert` | expert opponent (also changes our kart's speed) |
| `karts_4` | 3 opponents instead of 1 |
| `noise_0.2` | Gaussian noise on acceleration and steer |
| `delay_2` | actions reach the kart 2 steps late |
| `kart_gnu` | a different kart |
| `track_zengarden` | an unseen track |

## Step 8: Results

```bash
python sweep.py summary phase2
python render_agent.py --checkpoint runs/phase2/fixed/final.pt --out videos/fixed.mp4
```

Upload `runs/eval/phase1/` and `runs/eval/phase2/` (each has per-race CSVs and a `summary.csv`) to the shared Drive for the Overleaf write-up.

## Changing the plan

Every experiment is a line in the dictionaries at the top of `sweep.py` (`BC`, `BASE`, `PHASE1`, `SEEDS`, `FINETUNE`, `PHASE2_TRAIN`, `PHASE2_EVAL`). Any env setting works as a change: `track`, `n_karts`, `kart`, the reward weights, `action_noise`, `action_delay`, `no_brakes`. The same settings work by hand:

```bash
python train.py --run-dir runs/my_test --init-from runs/bc/policy.pt --env step_penalty=0.05
python evaluate.py runs/my_test/final.pt --env step_penalty=0.05 --name my_test
```

## Things to know

- **Difficulty changes our kart's speed too.** The same path-following controller laps in about 60 s against the novice and 41 s against the expert, so only compare times within one difficulty. Against the expert even that controller never wins, so expect a win rate near 0.
- **Seeds matter more than races.** The race seed barely changes a fixed policy's race, so 20 races per checkpoint is plenty; the second training seed is what tells you whether a difference is real.
- **A stopped training run restarts from scratch** (the replay buffer isn't saved). Evaluations pick up where they stopped.
- **Older checkpoints don't compare.** Anything trained with 2 actions, the old reward, or the old agent (e.g. `runs/full_training_01`) won't load or isn't comparable.
- **Timing** was measured on one RTX 5070 laptop: about 2 hours per 100k training steps, a few minutes per 20-race evaluation, and about 10 minutes for the warm start.
