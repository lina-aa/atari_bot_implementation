from kingkong_wrapper_base import (
    KingKongHeightWrapperBase,
    PLAYER_Y_ADDR,
    LADDER_RAM_ADDR,
    LADDER_ON_VALUE,
)


class KingKongHeightWrapper5(KingKongHeightWrapperBase):
    """Wrapper v6: dense per-step delta-Y reward, ladder terms, height milestones, 
    large Y-jump (floor) bonus, death and time penalties.

    Stabilizes climbing with many small signals instead of record-only reward; needs tuning and
    often two-stage training (terminal_on_life_loss on/off).
    """

    DELTA_CLIMB_COEF = 0.05

    LADDER_BONUS_COEF = 0.5
    LADDER_FLAT_BONUS = 0.1

    MILESTONE_STEP = 8
    MILESTONE_BONUS = 2.0

    LEVEL_UP_BONUS = 50.0
    LEVEL_UP_DELTA = 30

    DEATH_PENALTY = 5.0
    TIME_PENALTY = 0.002

    Y_JUMP_GUARD = 30

    def __init__(self, env):
        super().__init__(env)
        self.prev_y = None
        self.lives = None
        self.furthest_y = None

    def reset(self, **kwargs):
        self.highest_y = None
        self.steps_without_progress = 0
        self.prev_y = None
        self.lives = None
        self.furthest_y = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.unwrapped.ale.getRAM()
        current_y = int(ram[PLAYER_Y_ADDR])
        on_ladder = int(ram[LADDER_RAM_ADDR]) == LADDER_ON_VALUE
        current_lives = info.get("lives", 0)

        if self.prev_y is None:
            self.prev_y = current_y
            self.lives = current_lives
            self.furthest_y = current_y
            self.highest_y = current_y
            return obs, reward, terminated, truncated, info

        delta = current_y - self.prev_y
        delta_up = self.prev_y - current_y

        if delta > self.LEVEL_UP_DELTA and current_lives == self.lives:
            reward += self.LEVEL_UP_BONUS
            self.furthest_y = current_y
        elif current_y > 0 and abs(delta_up) < self.Y_JUMP_GUARD:
            reward += self.DELTA_CLIMB_COEF * delta_up
            if on_ladder and delta_up > 0:
                reward += self.LADDER_BONUS_COEF * delta_up + self.LADDER_FLAT_BONUS
            while self.furthest_y is not None and current_y <= self.furthest_y - self.MILESTONE_STEP:
                self.furthest_y -= self.MILESTONE_STEP
                reward += self.MILESTONE_BONUS

        if current_lives < self.lives:
            reward -= self.DEATH_PENALTY
            self.furthest_y = current_y
        self.lives = current_lives

        reward -= self.TIME_PENALTY

        if self.highest_y is None or current_y < self.highest_y:
            self.highest_y = current_y

        self.prev_y = current_y
        return obs, reward, terminated, truncated, info
