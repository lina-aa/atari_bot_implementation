from kingkong_wrapper_base import (
    KingKongHeightWrapperBase,
    PLAYER_Y_ADDR,
    LADDER_RAM_ADDR,
    LADDER_ON_VALUE,
)


class KingKongHeightWrapper6(KingKongHeightWrapperBase):
    """Wrapper v7: simpler signal — per-step Y delta reward, extra when climbing a ladder, big one-shot
    bonus when leaving the ladder (64→0) after enough net climb.

    Simpler than v6; Time and death penalties apply.
    """

    UP_REWARD_COEF = 0.2
    LADDER_UP_REWARD_COEF = 0.15
    LADDER_EXIT_MIN_CLIMB = 12
    LADDER_EXIT_BONUS = 20.0
    TIME_PENALTY = 0.02
    DEATH_PENALTY = 10.0

    def __init__(self, env):
        super().__init__(env)
        self.prev_y = None
        self.lives = None
        self.prev_ladder_state = 0
        self.ladder_start_y = None

    def reset(self, **kwargs):
        self.highest_y = None
        self.steps_without_progress = 0
        self.prev_y = None
        self.lives = None
        self.prev_ladder_state = 0
        self.ladder_start_y = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.unwrapped.ale.getRAM()
        current_y = int(ram[PLAYER_Y_ADDR])
        current_ladder = int(ram[LADDER_RAM_ADDR])
        on_ladder = current_ladder == LADDER_ON_VALUE
        current_lives = info.get("lives", 0)

        if self.prev_y is None:
            self.prev_y = current_y
            self.lives = current_lives
            self.prev_ladder_state = current_ladder
            self.highest_y = current_y
            return obs, reward, terminated, truncated, info

        delta_up = self.prev_y - current_y

        if current_y > 0:
            reward += self.UP_REWARD_COEF * delta_up
            if on_ladder and delta_up > 0:
                reward += self.LADDER_UP_REWARD_COEF * delta_up

        if on_ladder and self.prev_ladder_state != LADDER_ON_VALUE:
            self.ladder_start_y = current_y
        elif (not on_ladder) and self.prev_ladder_state == LADDER_ON_VALUE:
            if self.ladder_start_y is not None:
                climbed = self.ladder_start_y - current_y
                if climbed == self.LADDER_EXIT_MIN_CLIMB:
                    reward += self.LADDER_EXIT_BONUS
            self.ladder_start_y = None

        if current_lives < self.lives:
            reward -= self.DEATH_PENALTY
            self.ladder_start_y = None
        self.lives = current_lives
        self.prev_ladder_state = current_ladder

        reward -= self.TIME_PENALTY

        if self.highest_y is None or current_y < self.highest_y:
            self.highest_y = current_y

        self.prev_y = current_y
        return obs, reward, terminated, truncated, info
