from kingkong_wrapper_base import (
    KingKongHeightWrapperBase,
    PLAYER_Y_ADDR,
    LADDER_RAM_ADDR,
    LADDER_ON_VALUE,
)

# Spróbujemy znaleźć X - zwykle to 0x20 w King Kongu
PLAYER_X_ADDR = 0x20
LADDER_X_ADDR = 0x65  # X drabiny


class KingKongHeightWrapper9(KingKongHeightWrapperBase):
    """Wrapper v10: Smart ladder navigation coach
    
    Nagrody za:
    1. Zbliżanie się do drabiny (odległość X)
    2. Wspinanie na drabinie (odległość Y)
    3. Być na drabinie i wspinać się
    
    Kary za:
    1. Włóczenie się na boki bez sensu
    2. Stanie w miejscu zbyt długo
    """

    # Współczynniki reward'ów
    APPROACH_REWARD = 0.5      # Zbliżanie się do drabiny
    CLIMB_REWARD = 2.0        # Wspinanie na drabinie (zwiększone)
    LADDER_BONUS = 0.3         # Bonus za bycie na drabinie
    SIDESTEPPING_PENALTY = 0.1 # Kara za ruchy na boki
    DESCEND_PENALTY = 3.0      # ← NOWE: Kara za schodzenie w dół na drabinie
    FALL_PENALTY = 1.5         # ← NOWE: Kara za spadnięcie z drabiny
    TIME_PENALTY = 0.01        # Kara za każdy krok
    STALL_PENALTY = 0.05       # Kara za stanie w miejscu
    DEATH_PENALTY = 10.0       # Kara za śmierć
    CONSECUTIVE_CLIMB_BONUS = 0.4  # ← NOWE: Bonus za kilka kroków wspinania z rzędu

    def __init__(self, env):
        super().__init__(env)
        self.prev_x = None
        self.prev_y = None
        self.lives = None
        self.on_ladder_steps = 0  # Ile kroków jest na drabinie
        self.ladder_x = None       # Gdzie jest drabina
        self.idle_counter = 0      # Licznik bezczynności
        self.was_on_ladder = False # ← NOWE: Czy był na drabinie poprzednio
        self.consecutive_climbs = 0  # ← NOWE: Licznik konsekutywnych wspinań

    def reset(self, **kwargs):
        self.prev_x = None
        self.prev_y = None
        self.lives = None
        self.on_ladder_steps = 0
        self.ladder_x = None
        self.idle_counter = 0
        self.was_on_ladder = False
        self.consecutive_climbs = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.unwrapped.ale.getRAM()

        try:
            current_x = int(ram[PLAYER_X_ADDR])
            current_y = int(ram[PLAYER_Y_ADDR])
            current_ladder = int(ram[LADDER_RAM_ADDR])
            ladder_x = int(ram[LADDER_X_ADDR]) if self.ladder_x is None else self.ladder_x
        except (IndexError, ValueError):
            return obs, reward, terminated, truncated, info

        on_ladder = current_ladder == LADDER_ON_VALUE
        current_lives = info.get("lives", 0)

        # Inicjalizacja
        if self.prev_x is None:
            self.prev_x = current_x
            self.prev_y = current_y
            self.lives = current_lives
            self.ladder_x = ladder_x
            return obs, reward, terminated, truncated, info

        # ============ LOGIKA REWARD'ÓW ============

        # 1. Nagroda za zbliżanie się do drabiny (zależy od X)
        dist_to_ladder = abs(current_x - self.ladder_x)
        prev_dist_to_ladder = abs(self.prev_x - self.ladder_x)
        
        if dist_to_ladder < prev_dist_to_ladder:
            # Zbliżył się do drabiny
            reward += self.APPROACH_REWARD * (prev_dist_to_ladder - dist_to_ladder)
            self.idle_counter = 0
        elif dist_to_ladder == prev_dist_to_ladder:
            # Stoi w miejscu
            self.idle_counter += 1
        else:
            # Oddala się od drabiny - kara
            self.idle_counter += 1

        # 2. Nagroda za wspinanie (Y delta)
        delta_y = self.prev_y - current_y  # Mniejsze Y = wyżej
        
        if on_ladder:
            if delta_y > 0:
                # Na drabinie i wspina się! ⬆️
                reward += self.CLIMB_REWARD * delta_y
                self.consecutive_climbs += 1
                # Bonus za serię wspinań
                if self.consecutive_climbs > 3:
                    reward += self.CONSECUTIVE_CLIMB_BONUS * self.consecutive_climbs
                reward += self.LADDER_BONUS
                self.idle_counter = 0
            elif delta_y < 0:
                # Na drabinie ale SCHODZI! ⬇️ - KARA!
                reward -= self.DESCEND_PENALTY * abs(delta_y)
                self.consecutive_climbs = 0
                self.on_ladder_steps += 1
            else:
                # Na drabinie ale stoi w miejscu
                self.consecutive_climbs = 0
                self.on_ladder_steps += 1
                reward += self.LADDER_BONUS * 0.3
        else:
            # Zszedł z drabiny
            if self.was_on_ladder and current_y > self.prev_y:
                # ← NOWE: Był na drabinie i teraz jest NIŻEJ (spadł)
                reward -= self.FALL_PENALTY
            self.consecutive_climbs = 0
            self.on_ladder_steps = 0
        
        self.was_on_ladder = on_ladder

        # 3. Kara za chaotyczne ruchy na boki
        delta_x = abs(current_x - self.prev_x)
        if delta_x > 5 and not on_ladder:  # Duży skok na boki poza drabną
            reward -= self.SIDESTEPPING_PENALTY * delta_x

        # 4. Kara za całkowita bezczynność
        if self.idle_counter > 100:
            reward -= self.STALL_PENALTY
            self.idle_counter = 0

        # 5. Kara za śmierć
        if current_lives < self.lives:
            reward -= self.DEATH_PENALTY
            self.idle_counter = 0

        # 6. Bazowa kara za czas
        reward -= self.TIME_PENALTY

        # Update stanu
        self.prev_x = current_x
        self.prev_y = current_y
        self.lives = current_lives

        if self.highest_y is None or current_y < self.highest_y:
            self.highest_y = current_y

        return obs, reward, terminated, truncated, info
