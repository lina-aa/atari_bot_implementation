from kingkong_wrapper_base import (
    KingKongHeightWrapperBase,
    LADDER_ON_VALUE,
)


class KingKongHeightWrapper3(KingKongHeightWrapperBase):
    """Wrapper v4: height records, a small per-step “idle” penalty, ladder bonus, death penalty.

    A small negative per-step cost encourages moving; climbing works as before; on-ladder
    steps get an extra boost. Losing a life is penalized strongly.
    """

    def __init__(self, env):
        super().__init__(env)
        self.lives = None

    def reset(self, **kwargs):
        self.highest_y = None
        self.lives = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_y = self._get_current_y()
        current_lives = info.get("lives", 0)

        if self.highest_y is None:
            self.highest_y = current_y
            self.lives = current_lives

        reward -= 0.001

        if current_y < self.highest_y and current_y > 0:
            bonus = (self.highest_y - current_y) * 0.1
            reward += bonus
            self.highest_y = current_y

            if self._get_ladder_state() == LADDER_ON_VALUE:
                reward += 0.05

        if current_lives < self.lives:
            reward -= 2.0
            self.lives = current_lives

        return obs, reward, terminated, truncated, info
