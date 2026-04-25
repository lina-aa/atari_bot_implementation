from kingkong_wrapper_base import KingKongHeightWrapperBase


class KingKongHeightWrapper1(KingKongHeightWrapperBase):
    """Wrapper v2: rewards only new height records (smaller Y means higher on screen).

    When the player climbs higher than the previous best, adds a bonus proportional
    to vertical gain.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_y = self._get_current_y()
        self._initialize_tracking(current_y)

        if current_y < self.highest_y and current_y > 0:
            bonus = (self.highest_y - current_y) * 0.1
            reward += bonus

            self.highest_y = current_y

        return obs, reward, terminated, truncated, info
