from kingkong_wrapper_base import (
    KingKongHeightWrapperBase,
    PLAYER_Y_ADDR,
    LADDER_RAM_ADDR,
    LADDER_ON_VALUE,
)


class KingKongHeightWrapper7(KingKongHeightWrapperBase):
    """Wrapper v8: extends v7 with escalating penalty for no upward progress, a separate “ladder loiter”
    penalty, penalty for jump without useful Y gain.

    Reduces top-of-ladder oscillation and idle behavior;
    """

    UP_REWARD_COEF = 0.25
    LADDER_UP_COEF = 0.20

    LADDER_EXIT_MIN_CLIMB = 12
    LADDER_EXIT_BONUS = 25.0

    STALL_GRACE = 60
    STALL_COEF = 0.002
    STALL_MAX = 0.4
    LADDER_STALL_MULT = 2.0

    LADDER_LOITER_GRACE = 40
    LADDER_LOITER_COEF = 0.005
    LADDER_LOITER_MAX = 0.5

    JUMP_Y_THRESHOLD = 12
    JUMP_PENALTY = 0.4

    TIME_PENALTY = 0.005
    DEATH_PENALTY = 10.0

    def __init__(self, env):
        super().__init__(env)
        self.prev_y = None
        self.lives = None
        self.prev_ladder_state = 0
        self.ladder_start_y = None
        self.steps_on_ladder = 0
        self._jump_actions = None

    def _get_jump_actions(self):
        if self._jump_actions is None:
            try:
                meanings = self.unwrapped.get_action_meanings()
                self._jump_actions = {i for i, m in enumerate(meanings) if "FIRE" in m}
            except Exception:
                self._jump_actions = set()
        return self._jump_actions

    def reset(self, **kwargs):
        self.highest_y = None
        self.steps_without_progress = 0
        self.prev_y = None
        self.lives = None
        self.prev_ladder_state = 0
        self.ladder_start_y = None
        self.steps_on_ladder = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        is_jump = action in self._get_jump_actions()

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
            self.steps_on_ladder = 1 if on_ladder else 0
            return obs, reward, terminated, truncated, info

        delta_up = self.prev_y - current_y

        if current_y > 0:
            reward += self.UP_REWARD_COEF * delta_up
            if on_ladder and delta_up > 0:
                reward += self.LADDER_UP_COEF * delta_up

        if on_ladder and self.prev_ladder_state != LADDER_ON_VALUE:
            self.ladder_start_y = current_y
            self.steps_on_ladder = 0
        elif not on_ladder and self.prev_ladder_state == LADDER_ON_VALUE:
            if self.ladder_start_y is not None:
                climbed = self.ladder_start_y - current_y
                if climbed >= self.LADDER_EXIT_MIN_CLIMB:
                    reward += self.LADDER_EXIT_BONUS
            self.ladder_start_y = None
            self.steps_on_ladder = 0

        if on_ladder:
            self.steps_on_ladder += 1

        if on_ladder and self.steps_on_ladder > self.LADDER_LOITER_GRACE:
            loiter_excess = self.steps_on_ladder - self.LADDER_LOITER_GRACE
            loiter_penalty = min(self.LADDER_LOITER_MAX, loiter_excess * self.LADDER_LOITER_COEF)
            reward -= loiter_penalty

        if delta_up > 0:
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

        if self.steps_without_progress > self.STALL_GRACE:
            stall_excess = self.steps_without_progress - self.STALL_GRACE
            stall_penalty = min(self.STALL_MAX, stall_excess * self.STALL_COEF)
            if on_ladder:
                stall_penalty *= self.LADDER_STALL_MULT
            reward -= stall_penalty

        if is_jump and not on_ladder and current_y > 0:
            if delta_up < self.JUMP_Y_THRESHOLD:
                reward -= self.JUMP_PENALTY

        if current_lives < self.lives:
            reward -= self.DEATH_PENALTY
            self.ladder_start_y = None
            self.steps_on_ladder = 0
        self.lives = current_lives

        reward -= self.TIME_PENALTY

        if self.highest_y is None or current_y < self.highest_y:
            self.highest_y = current_y

        self.prev_ladder_state = current_ladder
        self.prev_y = current_y
        return obs, reward, terminated, truncated, info
