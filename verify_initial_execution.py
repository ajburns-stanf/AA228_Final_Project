# === Verify Initial Execution of STK ===

# -- Imports --
from PIL import Image  # Adds image processing capabilities

from stk_setup import setup

setup()
from pathlib import Path

import pystk2

# -- Definitions --
FRAMES_DIR = Path(__file__).parent / "frames"
SAVE_FRAMES = True  # Whether to save frames as PNGs
N_STEPS = 100  # Number of steps to run in the simulation
N_KARTS = 2  # Total number of karts


# === Run Simulation for 100 Steps ===
# NOTE: Example Quick Start from https://pystk2.readthedocs.io/en/latest/start.html
if __name__ == "__main__":
    # -- Initialize the rendering engine (once per process) --
    config = pystk2.GraphicsConfig.hd()
    config.screen_width = 800
    config.screen_height = 600
    pystk2.init(config)  # initialize the defined config

    # -- Setup the race (one per process) --
    # NOTE: kart 0 is driven by our actions, the other N_KARTS - 1 are STK's AI
    config = pystk2.RaceConfig()
    config.num_kart = N_KARTS  # Total number of karts (default is 1)
    config.track = "lighthouse"  # Specify the track

    # -- One PlayerConfig per kart, so ours is kart 0 and the only one with a camera --
    config.players[0].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL
    config.players[0].camera_mode = pystk2.PlayerConfig.CameraMode.ON
    for _ in range(N_KARTS - 1):
        opponent = pystk2.PlayerConfig()
        opponent.controller = pystk2.PlayerConfig.Controller.AI_CONTROL
        opponent.camera_mode = pystk2.PlayerConfig.CameraMode.OFF
        config.players.append(opponent)

    race = pystk2.Race(config)

    # -- Create the directory to save frames --
    FRAMES_DIR.mkdir(exist_ok=True)

    # -- Step through the simulation --
    race.start()
    for i in range(N_STEPS):
        # - Implement policy -
        action = pystk2.Action(
            steer=0.0, acceleration=1.0
        )  # placeholder: full throttle straight ahead
        race_ended = race.step(action)

        # - Save frames if enabled -
        if SAVE_FRAMES:
            for cam, data in enumerate(race.render_data):
                Image.fromarray(data.image).save(
                    FRAMES_DIR / f"step{i:03d}_cam{cam}.png"
                )

            print("Saved frames to", FRAMES_DIR)

    print("Finished", N_STEPS, "steps, race_ended =", race_ended)

    # -- Clean up the simulation --
    race.stop()
    del race
    pystk2.clean()
