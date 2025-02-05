from gymnasium.envs.registration import register

register(
    id = 'Maze_env/MazeRunner-v0',
    entry_point = 'Maze_env.env:maze_env'
)