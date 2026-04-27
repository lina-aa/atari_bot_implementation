import gymnasium as gym

from kingkong_wrapper_base import (
    LADDER_ON_VALUE,
    LADDER_RAM_ADDR,
    PLAYER_Y_ADDR,
)

def _on_ladder(ladder_ram_value: int) -> bool:
    return ladder_ram_value == LADDER_ON_VALUE

class KingKongHeightWrapper8(gym.Wrapper):

    PROGRESS_COEF = 0.25
    LADDER_EXIT_BONUS = 20.0
    MAX_STEPS_WITHOUT_PROGRESS = 300
    STALL_TRUNCATE_PENALTY = 1.0
    DOWNWARD_LADDER_PENALTY = 2.0

    def __init__(self, env):
        super().__init__(env)
        self.best_y = None
        self.steps_since_progress = 0
        self.previous_ladder_ram = None
        self.previous_y = None
        self.ladder_entry_y = None
        self.best_ladder_exit_y = None

    def reset(self, **kwargs):
        self.best_y = None
        self.steps_since_progress = 0
        self.previous_ladder_ram = None
        self.previous_y = None
        self.ladder_entry_y = None
        self.best_ladder_exit_y = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        y, current_ladder_ram = self._read_player_y_and_ladder()

        if self._should_penalize_downward_on_ladder(
            self.previous_ladder_ram, current_ladder_ram, y
        ):
            reward -= self.DOWNWARD_LADDER_PENALTY

        reward += self._ladder_edge_shaping_reward(
            self.previous_ladder_ram, current_ladder_ram, y
        )
        self.previous_ladder_ram = current_ladder_ram

        reward += self._height_progress_shaping_reward(y)

        if self._stall_limit_reached():
            reward -= self.STALL_TRUNCATE_PENALTY

        self.previous_y = y
        return obs, reward, terminated, truncated, info

    def _read_player_y_and_ladder(self):
        ram = self.unwrapped.ale.getRAM()
        y = int(ram[PLAYER_Y_ADDR])
        current_ladder_ram = int(ram[LADDER_RAM_ADDR])
        return y, current_ladder_ram

    def _should_penalize_downward_on_ladder(
        self, previous_ladder_ram, current_ladder_ram, y_after_step
    ):
        if previous_ladder_ram is None or self.previous_y is None:
            return False
        if not _on_ladder(previous_ladder_ram) or not _on_ladder(current_ladder_ram):
            return False
        # Higher Y = further down the screen; penalize each downward step on a ladder.
        return y_after_step > self.previous_y

    def _ladder_edge_shaping_reward(
        self, previous_ladder_ram, current_ladder_ram, y_after_step
    ):
        """Reward leaving a ladder after a net climb, only once per new best exit height."""
        if previous_ladder_ram is None:
            return 0.0

        was_on_ladder = _on_ladder(previous_ladder_ram)
        is_on_ladder = _on_ladder(current_ladder_ram)

        if not was_on_ladder and is_on_ladder:
            self.ladder_entry_y = y_after_step
            return 0.0

        if was_on_ladder and not is_on_ladder:
            bonus = self._maybe_ladder_exit_bonus(y_after_step)
            self.ladder_entry_y = None
            return bonus

        return 0.0

    def _maybe_ladder_exit_bonus(self, exit_y):
        if self.ladder_entry_y is None:
            return 0.0

        climbed_up = exit_y < self.ladder_entry_y
        if not climbed_up:
            return 0.0

        exit_is_new_best = (
            self.best_ladder_exit_y is None or exit_y < self.best_ladder_exit_y
        )
        if not exit_is_new_best:
            return 0.0

        self.best_ladder_exit_y = exit_y
        self.steps_since_progress = 0
        return self.LADDER_EXIT_BONUS

    def _height_progress_shaping_reward(self, y):
        """Dense reward only when Y improves on the episode; updates stall."""
        if self.best_y is None:
            self.best_y = y
            self.steps_since_progress = 0
            return 0.0

        if y < self.best_y:
            gain = self.best_y - y
            self.best_y = y
            self.steps_since_progress = 0
            return self.PROGRESS_COEF * gain

        self.steps_since_progress += 1
        return 0.0

    def _stall_limit_reached(self):
        return self.steps_since_progress >= self.MAX_STEPS_WITHOUT_PROGRESS
