# Mykel Kart

AA228/CS238 final project at Stanford: an autonomous driving agent for [SuperTuxKart](https://supertuxkart.net) that sees only raw camera pixels, and a study of how its decisions change when the game's dynamics are perturbed.

**Team:** Alex Burns, Matthew Lu, Lars Osterberg

## Approach
- Driving is modeled as a POMDP: the agent observes only RGB frames, never the game's internal state.
- A CNN encodes each frame, and a GRU carries a belief state across frames.
- The policy is trained with Soft Actor-Critic (SAC), optionally warm-started by behavior cloning of the game's built-in AI.
- Perturbations (tire friction, acceleration, no brakes, rewards for hitting opponents) test how the learned behavior adapts.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
On Windows, the `pysupertuxkart2` wheel ships without its DLLs. Install [SuperTuxKart 1.5](https://supertuxkart.net/Download); `stk_setup.py` loads the DLLs from `C:\Program Files\SuperTuxKart 1.5`.

## Quick check
```bash
python verify_initial_execution.py
```
Runs a 100-step race on `lighthouse` and saves each camera frame to `frames/`. The first run downloads the game data.
