from gymnasium.envs.registration import register

register(
    id = 'Maze_env/BasicMaze-v0',
    entry_point = 'Maze_env.env:BasicMaze'
)

register(
    id = 'Maze_env/MazeRunner-v0',
    entry_point = 'Maze_env.env:MazeRunner'
)

register(
    id = 'Maze_env/MonsterMaze-v0',
    entry_point = 'Maze_env.env:MonsterMaze'
)