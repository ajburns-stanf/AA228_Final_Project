# === Create Swappable Env between SAC on State and SAC on Image ===
"""obs="state" and obs="pixels" build the same race, actions, reward. Only the observation type differs.
- Action (both obs types):  [acceleration 0..1, steer -1..1]
- obs="state":              18 floats (listed in KartEnv._state)
- obs="pixels":             camera frame, uint8 array of shape (height, width, 3)"""

# === Imports ===
from stk_setup import setup

setup()  # ensure DLL is properly imported for windows
import gymnasium as gym
import numpy as np
import pystk2

# === Definitions ===
TRACK = "lighthouse"
N_KARTS = 2  # total karts
FRAME_SIZE = (128, 96)  # (width, height) of the pixel observation
MAX_STEPS = 1500  # steps before a run is cut off
STUCK_STEPS = 30  # steps without progress before a run is cut off
N_LOOKAHEAD = 5  # upcoming points in the state


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
    def __init__(self, obs_type):
        """Initialize the kart enviornment/SAC parameters."""
        self.obs = obs_type
        self.race = None

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
            low=np.array([0, -1], np.float32), high=np.array([1, 1], np.float32)
        )  # action: [acceleration 0..1, steer -1..1]

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
        config.num_kart = N_KARTS  # total karts
        config.track = TRACK  # specify track
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

        # Add Opponent Karts (AI Controlled)
        for _ in range(N_KARTS - 1):
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
        return self._obs(), {}

    def step(self, action):
        """Take a step in the environment with the given action. Then return info needed for SAC."""
        # -- Take the step and evaluate the world now --
        self.race.step(
            pystk2.Action(acceleration=float(action[0]), steer=float(action[1]))
        )
        self.world.update()
        kart = self.world.players[0].kart  # Obtain agent state
        self.steps += 1  # Increment step counter

        # -- Compute Reward: 0.1 per meter gained, -0.01 per step, -0.1 off-road, +10 for finishing --
        distance = self._distance()  # Compute distance along track
        progress = distance - self.distance  # Compute progress since last step
        self.distance = distance  # store for next step to track progress

        on_road = kart.is_on_road  # Check if kart is on the road
        finished = kart.has_finished_race  # Check if kart has finished the race

        reward = 0.1 * progress - 0.01 - 0.1 * (not on_road) + 10 * finished

        # -- Check if run should be stopped due to being stuck or exceeding max steps --
        self.stuck = (
            self.stuck + 1 if progress < 0.1 else 0
        )  # stuck if distance does not change between current and prior step
        truncated = (
            self.stuck >= STUCK_STEPS or self.steps >= MAX_STEPS
        )  # boolean true/false to signal that run has been stopped early

        # -- Return current observation, reward, finished status, truncated status, and additional info (distance) --
        return (
            self._obs(),
            float(reward),
            bool(finished),
            truncated,
            {"distance": distance},
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


def make_env(obs="state"):
    """Entry point for the training scripts"""
    return KartEnv(obs)
