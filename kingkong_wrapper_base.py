import gymnasium as gym

# RAM: Y gracza pod 0x21; drabina 0x64 (0 = poza, 64 = na drabinie).
PLAYER_Y_ADDR = 0x21
LADDER_RAM_ADDR = 0x64
LADDER_ON_VALUE = 64


class KingKongHeightWrapperBase(gym.Wrapper):
    """Shared base for King Kong (ALE) reward-shaping wrappers.

    Tracks best height Y and no-progress step counts; reads player Y and ladder
    state from RAM. Subclasses implement the actual reward logic.
    """

    def __init__(self, env):
        super().__init__(env)
        self.highest_y = None
        self.steps_without_progress = 0

    def _get_current_y(self):
        ram = self.unwrapped.ale.getRAM()
        return int(ram[PLAYER_Y_ADDR])

    def _get_ladder_state(self):
        ram = self.unwrapped.ale.getRAM()
        return int(ram[LADDER_RAM_ADDR])

    def _initialize_tracking(self, current_y):
        if self.highest_y is None:
            self.highest_y = current_y
            self.steps_without_progress = 0

    def step(self, action):
        raise NotImplementedError(
            "Use a concrete subclass (e.g. KingKongHeightWrapper1)."
        )

    def reset(self, **kwargs):
        self.highest_y = None
        self.steps_without_progress = 0
        return self.env.reset(**kwargs)
