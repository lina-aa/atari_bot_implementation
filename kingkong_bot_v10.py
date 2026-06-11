from kingkong_wrapper_base import (
    KingKongHeightWrapperBase,
    PLAYER_Y_ADDR,
    LADDER_RAM_ADDR,
    LADDER_ON_VALUE,
)

PLAYER_X_ADDR = 0x24  # potwierdzony empirycznie (0x20 zawsze=22, nie śledzi gracza)


LADDER_MAP: dict[int, list[int]] = {
    # floor_Y: [ladder_x1, ladder_x2, ...]  — uzupełnij z ram_scanner.py (klawisz S)
    92: [82, 218],   # piętro 1, lewa ściana (potwierdzone)
    82: [22],   # piętro 2, lewa ściana (potwierdzone) — brakuje środkowej drabiny
    70: [98, 220],
    58: [22],
}
SCREEN_CENTER_X = 80  # fallback gdy piętro nieznane


class KingKongHeightWrapper10(KingKongHeightWrapperBase):
    """Wrapper v10: połączenie v10 (nawigacja X) z mechanikami v9 (anti-farming).

    Z v10:
    - APPROACH_REWARD: naprowadza agenta do drabiny po osi X
    - on_new_floor + EXPLORATION_REWARD: tryb eksploracji po wejściu na nowe piętro

    Z v9:
    - PROGRESS_COEF: dense reward TYLKO za nowy rekord wysokości — eliminuje exploit oscylacji
    - LADDER_EXIT_BONUS: duży bonus za wyjście z drabiny wyżej niż wejście,
      ale tylko raz na nowy rekord — nie da się farmić
    - steps_since_progress: miernik stagnacji oparty na prawdziwym postępie,
      nie na samym ruchu
    """

    # Nagrody
    APPROACH_REWARD = 1.5
    PROGRESS_COEF = 0.5            # nowy rekord Y
    LADDER_EXIT_BONUS = 15.0       # net-climb przez drainę + nowy rekord wyjścia
    EXPLORATION_REWARD = 0.2       # ruch boczny podczas szukania nowej drabiny
    ON_LADDER_BONUS = 0.3

    # Kary
    SIDESTEPPING_PENALTY = 0.1
    DESCEND_PENALTY = 6.0
    FALL_PENALTY = 1.5
    LADDER_IDLE_PENALTY = 0.2
    LADDER_IDLE_THRESHOLD = 15
    STALL_PENALTY = 0.1            # Zwiększone: wymusza eksplorację
    MAX_STEPS_WITHOUT_PROGRESS = 150  # Zmniejszone: szybciej karze stagnację
    TIME_PENALTY = 0.01
    DEATH_PENALTY = 10.0

    def __init__(self, env):
        super().__init__(env)
        self.prev_x = None
        self.prev_y = None
        self.lives = None
        self.ladder_x = None
        self.was_on_ladder = False
        self.ladder_idle_steps = 0

        # v9: anti-farming
        self.best_y = None
        self.steps_since_progress = 0
        self.ladder_entry_y = None
        self.best_ladder_exit_y = None

        # v10: tryb eksploracji nowego piętra
        self.on_new_floor = False
        # X drabiny znalezionej przez eksplorację — niezależny od RAM[0x65]
        # który może wskazywać na drainę z poprzedniego piętra.
        # gracz wchodzi na drainę w trybie on_new_floor (gracz IS na drabinie → jego X = X drabiny).
        self.floor_ladder_x = None

    def reset(self, **kwargs):
        self.prev_x = None
        self.prev_y = None
        self.lives = None
        self.ladder_x = None
        self.was_on_ladder = False
        self.ladder_idle_steps = 0
        self.best_y = None
        self.steps_since_progress = 0
        self.ladder_entry_y = None
        self.best_ladder_exit_y = None
        self.on_new_floor = False
        self.floor_ladder_x = None
        return super().reset(**kwargs)

    def _nearest_ladder_x(self, player_x, player_y):
        """Zwraca X najbliższej drabiny z LADDER_MAP dla aktualnego piętra (tolerancja ±8px).
        Fallback: SCREEN_CENTER_X gdy piętro nieznane — agent eksploruje środek.
        """
        for floor_y, ladder_xs in LADDER_MAP.items():
            if abs(floor_y - player_y) <= 8:
                return min(ladder_xs, key=lambda x: abs(x - player_x))
        return SCREEN_CENTER_X

    def _ladder_edge_reward(self, on_ladder, current_x, current_y):
        """Nagroda za wejście/wyjście z drabiny (v9).

        Wejście: zapamiętuje Y wejścia; czyści tryb eksploracji.
        Wyjście: LADDER_EXIT_BONUS tylko jeśli net-climb i nowy rekord wyjścia.
        """
        reward = 0.0

        if not self.was_on_ladder and on_ladder:
            self.ladder_entry_y = current_y
            if self.on_new_floor:
                self.floor_ladder_x = current_x
                self.on_new_floor = False

        elif self.was_on_ladder and not on_ladder:
            if (self.ladder_entry_y is not None and
                    current_y < self.ladder_entry_y):
                # Net climb: wyszedł wyżej niż wszedł
                if (self.best_ladder_exit_y is None or
                        current_y < self.best_ladder_exit_y):
                    self.best_ladder_exit_y = current_y
                    reward += self.LADDER_EXIT_BONUS
                    self.on_new_floor = True
                    self.floor_ladder_x = None  # Resetuj — na nowym piętrze szukamy nowej drabiny
                    self.steps_since_progress = 0
            self.ladder_entry_y = None

        return reward

    def _height_progress_reward(self, current_y):
        """Dense reward tylko za nowy rekord Y (v9 logic).

        Nagroda proporcjonalna do poprawy. Zero za powrót na znane wysokości.
        Eliminuje exploit polegający na oscylowaniu w górę/dół na drabinie.
        """
        if self.best_y is None:
            self.best_y = current_y
            self.steps_since_progress = 0
            return 0.0

        if current_y < self.best_y:
            gain = self.best_y - current_y
            self.best_y = current_y
            self.steps_since_progress = 0
            return self.PROGRESS_COEF * gain

        self.steps_since_progress += 1
        return 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.unwrapped.ale.getRAM()

        try:
            current_x = int(ram[PLAYER_X_ADDR])
            current_y = int(ram[PLAYER_Y_ADDR])
            current_ladder = int(ram[LADDER_RAM_ADDR])
            ladder_x = self._nearest_ladder_x(current_x, current_y)
            self.ladder_x = ladder_x
        except (IndexError, ValueError):
            return obs, reward, terminated, truncated, info

        on_ladder = current_ladder == LADDER_ON_VALUE
        current_lives = info.get("lives", 0)

        if self.prev_x is None:
            self.prev_x = current_x
            self.prev_y = current_y
            self.lives = current_lives
            return obs, reward, terminated, truncated, info

        #* ================= DRABINA: WEJŚCIE / WYJŚCIE =================

        reward += self._ladder_edge_reward(on_ladder, current_x, current_y)

        #* ================= RUCH X =================

        delta_x = current_x - self.prev_x

        if not self.on_new_floor:
            # Naprowadzanie do drabiny.
            # staramy się nawigowaniu do drabiny z poprzedniego piętra gdy jest nieaktualny.
            target_x = self.floor_ladder_x if self.floor_ladder_x is not None else ladder_x
            dist = abs(current_x - target_x)
            prev_dist = abs(self.prev_x - target_x)
            if dist < prev_dist:
                reward += self.APPROACH_REWARD * (prev_dist - dist)
        else:
            # Eksploracja: nagradzaj dowolny ruch boczny
            if abs(delta_x) > 0:
                reward += self.EXPLORATION_REWARD

        #* ================= RUCH Y =================

        delta_y = self.prev_y - current_y  # mniejsze Y = wyżej

        if on_ladder:
            reward += self.ON_LADDER_BONUS  # zachęta do wchodzenia i zostania na drabinie
            if delta_y > 0:
                self.ladder_idle_steps = 0
            elif delta_y < 0:
                reward -= self.DESCEND_PENALTY * abs(delta_y)
                self.ladder_idle_steps += 1
            else:
                self.ladder_idle_steps += 1
        else:
            if self.was_on_ladder and current_y > self.prev_y:
                reward -= self.FALL_PENALTY
            self.ladder_idle_steps = 0

        self.was_on_ladder = on_ladder

        #* ================= WYSOKOŚĆ =================

        reward += self._height_progress_reward(current_y)

        #* ================= KARY =================

        if abs(delta_x) > 5 and not on_ladder:
            reward -= self.SIDESTEPPING_PENALTY * abs(delta_x)

        if self.ladder_idle_steps > self.LADDER_IDLE_THRESHOLD:
            reward -= self.LADDER_IDLE_PENALTY

        if self.steps_since_progress >= self.MAX_STEPS_WITHOUT_PROGRESS:
            reward -= self.STALL_PENALTY

        if current_lives < self.lives:
            reward -= self.DEATH_PENALTY
            self.steps_since_progress = 0

        reward -= self.TIME_PENALTY

        #* ================= UPDATE =================

        self.prev_x = current_x
        self.prev_y = current_y
        self.lives = current_lives

        if self.highest_y is None or current_y < self.highest_y:
            self.highest_y = current_y

        return obs, reward, terminated, truncated, info
