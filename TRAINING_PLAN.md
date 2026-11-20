# Training Plan

Tune the pixel SAC agent once, freeze its settings, then study how changes to the reward and the world change its driving (speed vs. recklessness). Everything runs through `sweep.py`; each step below is one or two commands.

## How the pieces fit

| File | Job |
| --- | --- |
| `environment.py` | The race: actions `[acceleration, steer, brake, drift, nitro]`, observations, reward. Every experiment setting is a named option (`DEFAULT_OPTIONS`), changed with `--env key=value`. |
| `train.py` | Trains one run. Saves `step_025000.pt`, `step_050000.pt`, … every 25k steps plus `final.pt`, and logs `episodes.csv` / `updates.csv`. |
| `evaluate.py` | Races a policy on 50 fixed seeds and writes one CSV row per race (finish, win, stuck, time, speed, off-road, steering jerk, yaw rate, crashes). |
| `sweep.py` | Runs a whole phase: train → evaluate → summarize. Skips finished work, so after a stop you rerun the same command. |
| `render_agent.py` | 1920x1080 MP4 of a checkpoint's race. |

## Which policy each step uses

"Policy" means the pixel SAC agent from `SAC_CNN_Pixels.py` (CNN + GRU + actor), trained from scratch by `train.py` and saved as a checkpoint. Evaluations and videos always drive with its mean action (no exploration noise).

| Step | Policy | Trained or only evaluated? |
| --- | --- | --- |
| 1. Sanity check | No learned policy: scripted behaviors (stand still, reverse, random actions, a hand-written path follower) driving from the state vector | Neither: only scores the reward |
| 2. Smoke test | Tiny throwaway SAC runs in `runs/quick/` (600 steps) | Trained + evaluated, then discarded |
| 3. Phase 1 | One new SAC policy per run in `runs/phase1/<run>/`. Evaluated: `baseline`'s `step_100000.pt`, `step_200000.pt` and `step_300000.pt`, every other run's `final.pt` (100k). Plus the `random` and `full_throttle` baselines for reference. | Trained + evaluated |
| 4. Pick settings | None: you choose **settings** (α, γ, T, training length), not a checkpoint | Neither |
| 5. Phase 2a | New SAC policies trained from scratch with `FIXED`: `runs/phase2/fixed/final.pt` (the reference) and one per change, `runs/phase2/<change>/final.pt` | Trained + evaluated, each in the setting it was trained in |
| 6. Phase 2b | One policy only: `runs/phase2/fixed/final.pt` | Only evaluated, under each world change |
| 7. Videos | Any checkpoint; usually `runs/phase2/fixed/final.pt` and the phase 2a policies | Only rendered |

Phase 2 retrains its reference (`fixed`) instead of reusing a phase 1 checkpoint, because `FIXED` can combine winners that no single phase 1 run had (e.g. α = 0.05 and 200k steps). If `FIXED` matches one phase 1 run exactly, copy that run to save the training time; `sweep.py` sees it's done and skips it:

```bash
cp -r runs/phase1/alpha_0.05 runs/phase2/fixed
```

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
git switch training-plan
pip install -r requirements.txt
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

If `GPU: False` on an NVIDIA machine, install the CUDA build of PyTorch from pytorch.org (RTX 50-series needs CUDA 12.8 or newer). Don't resize the game window while training: the frames change size and training crashes.

## Step 1: Check the reward (1 minute)

```bash
python sweep.py sanity
```

It must print `OK: quitting < random driving < a lap`. Rerun it whenever a reward weight changes.

## Step 2: Smoke test the pipeline (about 5 minutes)

```bash
python sweep.py phase1 --quick --only baseline,random
python sweep.py summary phase1 --quick
```

Tiny runs under `runs/quick/`, to confirm everything works on this machine before starting hours of training.

## Step 3: Phase 1, tune how well it learns (about 16 hours, can be split)

Each run changes one knob from the baseline. All use the novice opponent (the only level where a good driver can win) and seed 0. Reward weights are not tuned here: they decide what "good driving" means, so they are phase 2's perturbations.

| Run | Change from baseline | Approx. time |
| --- | --- | --- |
| `baseline` | none, trained 300k steps; its 100k/200k/300k checkpoints answer "how long to train" | 6 h |
| `alpha_0.05`, `alpha_0.1` | entropy weight α (baseline 0.2) | 2 h each |
| `gamma_0.995`, `gamma_0.998` | discount γ (baseline 0.99): how far ahead the agent looks | 2 h each |
| `seq_16` | GRU sequence length T (baseline 8) | 2 h |

```bash
python sweep.py phase1 --dry-run   # prints every command first
python sweep.py phase1
```

To split it across two machines, give each person a set of runs, then copy everyone's `runs/eval/phase1/*.csv` into one folder before the summary:

```bash
python sweep.py phase1 --only baseline,random,full_throttle                       # person A
python sweep.py phase1 --only alpha_0.05,alpha_0.1,gamma_0.995,gamma_0.998,seq_16   # person B
```

## Step 4: Pick the fixed settings

```bash
python sweep.py summary phase1
```

Compare every 100k run against `baseline@100k`. Pick by finish rate first, then win rate, then finish time. With 50 races, gaps of a few percent are noise. For a close call, repeat the top two with another seed:

```bash
python sweep.py phase1 --only baseline,alpha_0.05 --seed 1
```

Then write the winning **settings** into `FIXED` in `sweep.py` and commit. Phase 2 trains new policies with them; the phase 1 checkpoints aren't carried over. For example:

```python
FIXED = {**BASE, "alpha": 0.05, "total_steps": 200_000}
```

## Step 5: Phase 2a, retrain under each change (how decisions adapt)

```bash
python sweep.py phase2-train
```

| Run | Change | Question |
| --- | --- | --- |
| `fixed` | none | The reference |
| `step_0.01`, `step_0.05` | `step_penalty` | Does time pressure make it faster, or more reckless? |
| `offroad_0`, `offroad_1` | `off_road_factor` | Does it cut corners when grass is cheap? |
| `win_45` | `win_bonus` | Does it take more risks when winning matters more? |
| `no_brakes` | `no_brakes=True` | How does it handle corners without brakes? |

Each run is evaluated in the setting it was trained in.

## Step 6: Phase 2b, robustness of the fixed policy (no retraining)

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

## Step 7: Results

```bash
python sweep.py summary phase2
python render_agent.py --checkpoint runs/phase2/fixed/final.pt --out videos/fixed.mp4
```

Upload `runs/eval/phase1/` and `runs/eval/phase2/` (each has per-race CSVs and a `summary.csv`) to the shared Drive for the Overleaf write-up.

## Changing the plan

Every experiment is a line in the dictionaries at the top of `sweep.py` (`PHASE1`, `PHASE2_TRAIN`, `PHASE2_EVAL`). Any env setting works as a change: `track`, `n_karts`, `kart`, the reward weights, `action_noise`, `action_delay`, `no_brakes`. The same settings work by hand:

```bash
python train.py --run-dir runs/my_test --env step_penalty=0.05 --env track=zengarden
python evaluate.py runs/my_test/final.pt --env track=zengarden --name my_test_zengarden
```

## Things to know

- **Difficulty changes our kart's speed too.** The same path-following controller laps in about 60 s against the novice and 41 s against the expert, so only compare times within one difficulty. Against the expert even that controller never wins, so expect a win rate near 0.
- **A stopped training run restarts from scratch** (the replay buffer isn't saved). Evaluations pick up where they stopped.
- **Results vary between runs.** Use the 50-race evaluations and a second seed for close calls, not a single training curve.
- **Older checkpoints don't compare.** Anything trained with 2 actions or the old reward (e.g. `runs/full_training_01`) won't load or isn't comparable.
- **Timing** was measured on one RTX 5070 laptop: about 2 hours per 100k training steps, and roughly 5–10 minutes per 50-race evaluation.
