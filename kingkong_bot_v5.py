from kingkong_wrapper_base import (
    KingKongHeightWrapperBase,
    PLAYER_Y_ADDR,
    LADDER_RAM_ADDR,
    LADDER_ON_VALUE,
)


class KingKongHeightWrapper4(KingKongHeightWrapperBase):
    """Wrapper v5: stronger height bonus, penalties for time without progress, ladder session and lives.

    After rewarded climbing, leaving the ladder (0x64 vale changes from 64 to 0) with clear upward 
    displacement yields a large one-shot bonus; death is penalized. 
    Aims to reward moving up the ladder, not only jumping in place.
    """

    def __init__(self, env):
        super().__init__(env)
        self.lives = None
        self.prev_ladder_state = 0
        self.ladder_start_y = None
        self.steps_since_progress = 0

    def reset(self, **kwargs):
        self.highest_y = None
        self.lives = None
        self.prev_ladder_state = 0
        self.ladder_start_y = None
        self.steps_since_progress = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.unwrapped.ale.getRAM()

        current_y = int(ram[PLAYER_Y_ADDR])
        current_ladder = int(ram[LADDER_RAM_ADDR])
        current_lives = info.get("lives", 0)

        if self.highest_y is None:
            self.highest_y = current_y
            self.lives = current_lives

        self.steps_since_progress += 1
        if self.steps_since_progress < 400:
            reward -= 0.01
        else:
            reward -= 0.1

        if current_y < self.highest_y and current_y > 0:
            bonus = (self.highest_y - current_y) * 0.2
            reward += bonus
            self.highest_y = current_y
            self.steps_since_progress = 0

        if current_ladder == LADDER_ON_VALUE and self.prev_ladder_state == 0:
            self.ladder_start_y = current_y
        elif current_ladder == 0 and self.prev_ladder_state == LADDER_ON_VALUE:
            if self.ladder_start_y is not None:
                if current_y < self.ladder_start_y - 10:
                    reward += 5.0
            self.ladder_start_y = None

        self.prev_ladder_state = current_ladder

        if current_lives < self.lives:
            reward -= 2.0
            self.lives = current_lives

        return obs, reward, terminated, truncated, info
