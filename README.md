## Atari game bot implementation for AI in game development PWr course
### Authors:
- Karolina Bajer
- Zuzanna Gorczyca
- Natalia Tymczyszyn

------------------------------------------------------------------

The objective of this project is to create a bot for Atari game.

KingKong was chosen as inital game to learn how RL works.

Bot versions in this repo:
1. v1 - basic bot version; learned only to jump over bombs (200 000 timesteps across 4 parallel environments)
2. v2 - bot with a early custom wrapper that gave additional reward for getting higher; only stood in the corner, didn't do much (200 000 timesteps across 4 parallel environments)
3. v3 - bot with upgraded wrapper; learned to use magicbombs to jump higher (200 000 timesteps + 2 000 000 additional timesteps across 4 parallel environments)


RAM adress  for the player y positon was checked using ram_map_y.py. It was identified as '0x21 (33)'.

### Milestones:
- 24.03: learn the bot to play chosen Atari game. Next milestones to be disscussed on the meeting.
- 5.05: make the bot win the game, currently it can do basic moves but it only loses
- 26.05: check if custom CNN architecture for PPO("CnnPolicy") would give better results