# === Create Swappable Env between SAC on State and SAC on Image ===
"""obs="state" and obs="pixels" build the same race, actions, reward. Only the observation type differs.
- Action (both obs types):  [acceleration 0..1, steer -1..1, brake, drift, nitro] (on/off 0..1, pressed above 0.5)
- obs="state":              18 floats (listed in KartEnv._state)
- obs="pixels":             camera frame, uint8 array of shape (height, width, 3)
- Experiments change settings by name: make_env("pixels", 2, step_penalty=0.05, track="zengarden") (see DEFAULT_OPTIONS)"""

# === Imports ===
from collections import deque

from stk_setup import setup

setup()  # ensure DLL is properly imported for windows
import gymnasium as gym
import numpy as np
import pystk2

# === Definitions ===
TRACK = "lighthouse"
DIFFICULTY = 2  # opponent AI: 0 novice, 1 intermediate, 2 expert (STK default)
N_KARTS = 2  # total karts
FRAME_SIZE = (128, 96)  # (width, height) of the pixel observation
MAX_STEPS = 1500  # steps before a run is cut off
STUCK_STEPS = 30  # steps without progress before a run is cut off
N_LOOKAHEAD = 5  # upcoming points in the state
PRESS_THRESHOLD = 0.5  # brake, drift and nitro count as pressed above this value

# -- Reward weights --
PROGRESS_REWARD = 0.1  # per meter gained along the track
STEP_PENALTY = 0.03  # per step: pushes the agent to finish faster
OFF_ROAD_FACTOR = 0.5  # off-road meters count this much: worse than the road, never worse than stopping
LEAD_REWARD = 0.02  # per step spent in 1st place
FINISH_BONUS = 5.0  # for finishing the race in any position
WIN_BONUS = 15.0  # extra for finishing in 1st place
STUCK_PENALTY = 10.0  # getting stuck ends the run as a failure, so quitting never beats trying

# -- Settings experiments can change by name (make_env(..., key=value) or train.py/evaluate.py --env key=value) --
DEFAULT_OPTIONS = {
    "track": TRACK,
    "n_karts": N_KARTS,
    "kart": "",  # our kart model, e.g. "gnu"; "" = STK's default (tux)
    "progress_reward": PROGRESS_REWARD,
    "step_penalty": STEP_PENALTY,
    "off_road_factor": OFF_ROAD_FACTOR,
    "lead_reward": LEAD_REWARD,
    "finish_bonus": FINISH_BONUS,
    "win_bonus": WIN_BONUS,
    "stuck_penalty": STUCK_PENALTY,
    "action_noise": 0.0,  # std of Gaussian noise added to acceleration and steer
    "action_delay": 0,  # steps between choosing an action and it reaching the kart
    "no_brakes": False,  # ignore the brake action
}


def rotation(q):
    """World-to-kart rotation matrix from the kart's quaternion (kart frame: x right, y up, z forward).
    Needed to determine positions on map into positions relative to the cart."""
    w, x, y, z = q
    return np.array(
        [
            [
                x * x + w * w - y * y - z * z,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
            ],
            [
                2 * w * z + 2 * x * y,
                w * w - x * x + y * y - z * z,
                2 * y * z - 2 * w * x,
            ],
            [
                2 * x * z - 2 * w * y,
                2 * w * x + 2 * y * z,
                w * w - x * x - y * y + z * z,
            ],
        ]
    )


# === Environment ===
class KartEnv(gym.Env):
    def __init__(self, obs_type, difficulty=DIFFICULTY, **options):
        """Initialize the kart enviornment/SAC parameters."""
        self.obs = obs_type
        self.difficulty = difficulty
        self.race = None

        # -- Experiment Settings: defaults, overridden by name --
        unknown = sorted(set(options) - set(DEFAULT_OPTIONS))
        if unknown:
            raise ValueError(f"Unknown env options {unknown}; valid: {sorted(DEFAULT_OPTIONS)}")
        self.opt = {**DEFAULT_OPTIONS, **options}

        # -- Setup Graphics Depending on Observation Type --
        if self.obs == "state":
            graphics = pystk2.GraphicsConfig.none()  # no rendering: much faster
        elif self.obs == "pixels":
            graphics = pystk2.GraphicsConfig.hd()
            graphics.screen_width, graphics.screen_height = FRAME_SIZE
        else:
            raise ValueError(f"Invalid obs_type: {obs_type}")
        pystk2.init(graphics)  # initialize graphics

        # -- Initialize Action Space --
        self.action_space = gym.spaces.Box(
            low=np.array([0, -1, 0, 0, 0], np.float32),
            high=np.array([1, 1, 1, 1, 1], np.float32),
        )  # action: [acceleration 0..1, steer -1..1, brake, drift, nitro] (on/off 0..1)

        # -- Initialize Observation Space --
        if self.obs == "state":
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
            )  # 18 floats for state observation (defined in _state)
        elif self.obs == "pixels":
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8
            )  # pixel observation: (height, width, 3)

    def reset(self, seed=None, options=None):
        """Reset the environment and start a new race."""
        super().reset(seed=seed)
        if self.race is not None:  # Stop Current Race if Race is Running
            self.race.stop()
            self.race = None

        # -- Setup Race Configuration --
        config = pystk2.RaceConfig()
        config.num_kart = self.opt["n_karts"]  # total karts
        config.track = self.opt["track"]  # specify track
        config.difficulty = self.difficulty  # opponent AI skill
        config.laps = 1  # one lap per run (STK's default is 3)
        config.seed = int(
            self.np_random.integers(2**16)
        )  # race randomness follows reset(seed=...)

        # -- Setup Player Configurations --
        config.players[0].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL
        config.players[0].camera_mode = (
            pystk2.PlayerConfig.CameraMode.ON
            if self.obs == "pixels"
            else pystk2.PlayerConfig.CameraMode.OFF
        )
        if self.opt["kart"]:
            config.players[0].kart = self.opt["kart"]  # kart model, e.g. "gnu"

        # Add Opponent Karts (AI Controlled)
        for _ in range(self.opt["n_karts"] - 1):
            opponent = pystk2.PlayerConfig()
            opponent.controller = pystk2.PlayerConfig.Controller.AI_CONTROL
            opponent.camera_mode = pystk2.PlayerConfig.CameraMode.OFF
            config.players.append(opponent)

        # -- Start the Race --
        self.race = pystk2.Race(config)
        self.race.start()
        self.world = pystk2.WorldState()
        self.world.update()

        # -- Track centerline: segment start/end points, widths, next segment --
        track = pystk2.Track()
        track.update()
        self.path = np.asarray(track.path_nodes)  # (n_segments, 2, 3)
        self.width = np.asarray(track.path_width)[:, 0]
        self.next = [s[0] for s in track.successors]

        # -- Skip the intro and countdown: the race clock starts at "Go" --
        while self.world.time == 0:
            self.race.step()
            self.world.update()

        self.distance = self._distance()
        self.steps = self.stuck = 0
        self.pending = deque(
            [np.zeros(5, np.float32)] * self.opt["action_delay"]
        )  # delayed actions waiting to reach the kart (no throttle at first)
        return self._obs(), {}

    def step(self, action):
        """Take a step in the environment with the given action. Then return info needed for SAC."""
        # -- Perturb the Controls if the Experiment Asks: noise, delay, no brakes --
        action = np.array(action, dtype=np.float32)
        if self.opt["action_noise"] > 0:
            action[:2] += self.np_random.normal(0, self.opt["action_noise"], 2)
            action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.opt["action_delay"] > 0:
            self.pending.append(action)
            action = self.pending.popleft()  # the action chosen action_delay steps ago

        # -- Take the step and evaluate the world now --
        brake, drift, nitro = (
            bool(a > PRESS_THRESHOLD) for a in action[2:5]
        )  # on/off actions are pressed above PRESS_THRESHOLD
        brake = brake and not self.opt["no_brakes"]
        self.race.step(
            pystk2.Action(
                acceleration=float(action[0]),
                steer=float(action[1]),
                brake=brake,
                drift=drift,
                nitro=nitro,
            )
        )
        self.world.update()
        kart = self.world.players[0].kart  # Obtain agent state
        self.steps += 1  # Increment step counter

        # -- Compute Reward: progress (less off-road), time, being in the lead, finishing, winning --
        distance = self._distance()  # Compute distance along track
        progress = distance - self.distance  # Compute progress since last step
        self.distance = distance  # store for next step to track progress

        on_road = kart.is_on_road  # Check if kart is on the road
        finished = kart.has_finished_race  # Check if kart has finished the race
        leading = kart.position == 1  # race rank, updated live (1 = first)

        progress_reward = self.opt["progress_reward"] * progress
        if progress > 0 and not on_road:
            progress_reward *= self.opt["off_road_factor"]  # forward meters off the road count less
        reward = (
            progress_reward
            - self.opt["step_penalty"]
            + self.opt["lead_reward"] * leading
            + (self.opt["finish_bonus"] + self.opt["win_bonus"] * leading) * finished
        )

        # -- Check if run should be stopped due to being stuck or exceeding max steps --
        self.stuck = (
            self.stuck + 1 if progress < 0.1 else 0
        )  # stuck if distance does not change between current and prior step
        stuck = self.stuck >= STUCK_STEPS
        if stuck:
            reward -= self.opt["stuck_penalty"]  # getting stuck is a failure, never a cheap way out
        terminated = finished or stuck  # both end the run for good: no future value
        truncated = (
            not terminated and self.steps >= MAX_STEPS
        )  # out of time: cut off, but the run still had a future

        # -- Return current observation, reward, terminated status, truncated status, and additional info --
        return (
            self._obs(),
            float(reward),
            bool(terminated),
            truncated,
            {
                "distance": distance,
                "position": kart.position,
                "finish_time": kart.finish_time if finished else None,
                "finished": bool(finished),  # terminated also covers getting stuck
                "stuck": stuck,
            },
        )

    def _distance(self):
        """Computes the distance along the track (Clamped at zero)"""
        d_along_track = self.world.players[0].kart.overall_distance
        return max(d_along_track, 0.0)

    def _obs(self):
        """Return observation for the given observation type"""
        if self.obs == "state":
            return self._state()
        elif self.obs == "pixels":
            return np.array(
                self.race.render_data[0].image
            )  # copy: STK reuses the buffer\
        else:
            raise ValueError(f"Invalid obs_type: {self.obs}")

    def _state(self):
        """Custom defined state vector from knowledge of true state from simulator. Used to verify SAC functionality and ability
        to train an agent in this environment. The 18 states chosen are:
        - Velocity (3 floats, normalized by 30 m/s)
        - Yaw Rate (1 float) (turning rate of the kart)
        - Centerline Offset (1 float, normalized by half track width)
        - Track Width (1 float, normalized by 20 m)
        - Heading Error (1 float, in radians)
        - Next 5 Centerline Points (10 floats, x and z coordinates in kart frame)
        - On Road (1 float, 1 if on road, 0 if off road)"""
        kart = self.world.players[0].kart
        R = rotation(kart.rotation)

        def to_kart(points):  # world points -> kart frame
            return (points - np.asarray(kart.location)) @ R.T

        # -- Compute segment related information for the kart's current position on the track --
        node = max(kart.node, 0)  # which segment of the track the kart is on
        start, end = to_kart(
            self.path[node]
        )  # current segment start and end points in kart frame
        direction = end - start  # kart desired direction vector along the track segment
        closest = (
            start - np.dot(start, direction) / np.dot(direction, direction) * direction
        )  # closest point on the track segment to the kart in kart frame

        # -- Compute the next N_LOOKAHEAD points along the track in kart frame --
        # Describes where the road goes over the next 35-40 meters.
        points, ix = [], node
        for _ in range(N_LOOKAHEAD):
            x, _, z = to_kart(self.path[ix, 1])
            points += [x, z]
            ix = self.next[ix]

        # -- Compute the state vector and return it as a numpy array of floats --
        state = np.array(
            [
                *np.asarray(kart.velocity_lc) / 30,  # velocity (3 floats)
                kart.angular_velocity[1],  # yaw rate (1 float)
                closest[0] / (self.width[node] / 2),  # centerline offset (1 float)
                self.width[node] / 20,  # track width (1 float)
                np.arctan2(direction[0], direction[2]),  # heading error (1 float)
                *np.array(points)
                / 50,  # next N_LOOKAHEAD centerline points (10 floats)
                kart.is_on_road,  # on road (1 float)
            ],
            dtype=np.float32,
        )
        return state

    def close(self):
        """Close the environment and clean up resources after a run."""
        if self.race is not None:
            self.race.stop()
            self.race = None
        pystk2.clean()


def make_env(obs="state", difficulty=DIFFICULTY, **options):
    """Entry point for the training scripts (options: see DEFAULT_OPTIONS)"""
    return KartEnv(obs, difficulty, **options)
