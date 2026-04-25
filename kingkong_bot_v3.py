from kingkong_wrapper_base import KingKongHeightWrapperBase


class KingKongHeightWrapper2(KingKongHeightWrapperBase):
    """Wrapper v3: v2 wrapper plus a penalty for long periods without progress.

    If Y does not improve for many steps, a negative reward nudges the policy away
    from standing still instead of climbing.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_y = self._get_current_y()
        self._initialize_tracking(current_y)

        if current_y < self.highest_y and current_y > 0:
            bonus = (self.highest_y - current_y) * 0.1
            reward += bonus

            self.highest_y = current_y
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

            if self.steps_without_progress > 500:
                reward -= 0.05

        return obs, reward, terminated, truncated, info
