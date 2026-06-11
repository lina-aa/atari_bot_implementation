import sys
import gymnasium as gym
import ale_py
import time
import os
from pynput import keyboard

# Użycie: python ram_scanner.py [mode]   np. python ram_scanner.py 1
GAME_MODE = int(sys.argv[1]) if len(sys.argv) > 1 else 0

PLAYER_X_ADDR  = 0x24   # 36  — potwierdzony: zmienia się przy ruchu poziomym
PLAYER_Y_ADDR  = 0x21   # 33  — potwierdzony
LADDER_ST_ADDR = 0x64   # 100 — potwierdzony (0 / 64)
LADDER_ON_VALUE = 64

X_CANDIDATES = [0x24]   # potwierdzony adres X

WATCH_ADDRS = {
    0x24: "player_X",
    0x21: "player_Y",
    0x64: "ladder",
}

diff_mode      = False
show_map       = False
save_snapshot  = False
prev_ram       = None

# Zebrane pozycje drabin: { floor_y: set(ladder_x) }
ladder_map: dict[int, set] = {}

# Zbiór aktualnie wciśniętych klawiszy ruchu
pressed = set()

def compute_action() -> int:
    """Wylicza akcję ALE z aktualnie wciśniętych klawiszy.

    Akcje King Kong:
      0=NOOP  1=FIRE  2=UP  3=RIGHT  4=LEFT  5=DOWN
      10=UPFIRE  11=RIGHTFIRE  12=LEFTFIRE
    """
    up    = keyboard.Key.up    in pressed
    down  = keyboard.Key.down  in pressed
    left  = keyboard.Key.left  in pressed
    right = keyboard.Key.right in pressed
    fire  = keyboard.Key.space in pressed

    if fire and right: return 11  # skok w prawo
    if fire and left:  return 12  # skok w lewo
    if fire and up:    return 10  # skok w górę
    if fire:           return 1
    if up:             return 2
    if down:           return 5
    if right:          return 3
    if left:           return 4
    return 0

current_action = 0

def on_press(key):
    global current_action, diff_mode, show_map, save_snapshot
    try:
        if key in (keyboard.Key.up, keyboard.Key.down,
                   keyboard.Key.left, keyboard.Key.right,
                   keyboard.Key.space):
            pressed.add(key)
            current_action = compute_action()
        elif hasattr(key, 'char'):
            if   key.char == 'd': diff_mode     = not diff_mode
            elif key.char == 's': show_map      = True
            elif key.char == 'r': save_snapshot = True
    except Exception:
        pass

def on_release(key):
    global current_action
    pressed.discard(key)
    current_action = compute_action()

def register_ladder(x, y):
    """Zapamiętaj pozycję drabiny. Grupuje po floor_y z tolerancją ±4px."""
    for floor_y in ladder_map:
        if abs(floor_y - y) <= 4:
            ladder_map[floor_y].add(x)
            return
    ladder_map[y] = {x}

def print_ladder_map():
    """Drukuje gotowy słownik Python do wklejenia w wrapperze."""
    print("\n" + "=" * 55)
    print("# Wklej do kingkong_bot_v10.py:")
    print("LADDER_MAP = {")
    for floor_y in sorted(ladder_map):
        xs = sorted(ladder_map[floor_y])
        print(f"    {floor_y}: {xs},   # piętro Y≈{floor_y}")
    print("}")
    print("=" * 55 + "\n")

def render_header(xs: dict, y: int, on_ladder: bool, step: int) -> str:
    status = "\033[92m NA DRABINIE \033[0m" if on_ladder else "             "
    x_str = "  ".join(f"0x{addr:02X}={val:3d}" for addr, val in xs.items())
    return (
        f"Krok:{step:6d} | Akcja:{current_action} | Y={y:3d}  ladder={'64' if on_ladder else ' 0'}  {status}\n"
        f"Kandydaci X:  {x_str}\n"
        f"Zebrane: {sum(len(v) for v in ladder_map.values())} drabin | "
        f"[D] diff={'ON' if diff_mode else 'OFF'}  [S] mapa  [R] snapshot"
    )

def render_diff(ram, prev):
    lines = [f"     {'  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F'}"]
    for row in range(8):
        base = row * 16
        cells = []
        for col in range(16):
            addr = base + col
            val  = ram[addr]
            changed = prev is not None and val != prev[addr]
            named   = addr in WATCH_ADDRS
            if changed and diff_mode:
                cells.append(f"\033[91m{val:02X}\033[0m")
            elif named:
                cells.append(f"\033[93m{val:02X}\033[0m")
            else:
                cells.append(f"{val:02X}")
        cells_str = " ".join(cells)
        label = "  ← " + "  ".join(
            f"{WATCH_ADDRS[base+c]}" for c in range(16) if (base+c) in WATCH_ADDRS
        )
        lines.append(f"0x{base:02X}  {cells_str}{label if label.strip() else ''}")
    return "\n".join(lines)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

env = gym.make("ALE/KingKong-v5", render_mode="human", full_action_space=True, mode=GAME_MODE)
print(f"Tryb gry: mode={GAME_MODE}  (zmień: python ram_scanner.py 0/1/2/3)")
obs, _ = env.reset()

prev_on_ladder = False
step = 0

while True:
    obs, reward, terminated, truncated, info = env.step(current_action)
    ram = env.unwrapped.ale.getRAM()

    x         = int(ram[PLAYER_X_ADDR])
    y         = int(ram[PLAYER_Y_ADDR])
    on_ladder = int(ram[LADDER_ST_ADDR]) == LADDER_ON_VALUE

    # Wykryj wejście na drainę i zapamiętaj pozycję
    if on_ladder and not prev_on_ladder:
        register_ladder(x, y)
        print(f"\033[92m  >> Drabina: X={x}  Y={y}\033[0m")

    if step % 4 == 0:
        os.system("clear")
        print(render_header({PLAYER_X_ADDR: x}, y, on_ladder, step))
        if diff_mode:
            print()
            print(render_diff(ram, prev_ram))

    if show_map:
        print_ladder_map()
        show_map = False

    if save_snapshot:
        fname = f"ram_snapshot_{step}.txt"
        with open(fname, "w") as f:
            f.write(f"Krok {step}\n")
            for addr in range(128):
                label = WATCH_ADDRS.get(addr, "")
                f.write(f"0x{addr:02X} ({addr:3d}): {ram[addr]:3d}  {label}\n")
        print(f"Zapisano: {fname}")
        save_snapshot = False

    prev_on_ladder = is_on_ladder = on_ladder
    prev_ram = ram.copy()
    step += 1
    time.sleep(0.05)

    if terminated or truncated:
        obs, _ = env.reset()
        prev_on_ladder = False
