from Maze_env.env.maze_runner import MazeRunner
from Maze_env.env.mazes import BasicMaze

# IN PROGRESS:
# Need to finish this while adding in appropriate methods to deal with 
# the monster killing other agents

# Need to change the _get_vision__ in the original to deal with seeing agents
# from different object types;  like it can see runners it wants to eat, but ignores
# other monsters for example, or even makes them a different color
# so it learns to ignore them
# Add in the appripriate info to be used in the rewards function
# Need to define the rewards function and what would push the monster 
# to want to go towards its prey.  
class MonsterMaze(BasicMaze):
    def __init__(self, maze, len_game=1000, num_objects={'runners':1,'monsters':1},
                  vision_len={'runners':1,'monsters':3}, type_of_objects={'runners','monsters'}, 
                  objectives={'monsters':'kill'}, action_type='full', 
                 render_mode='rgb_array', obs_type='spatial', init_pos={}, 
                 start_dist={}, dist_paradigm='radius', 
                 collision_rules={}, colormap={'runners':'tab10', 'monsters':'reds'}):
        super().__init__(maze, len_game, num_objects, vision_len, 
                         type_of_objects, objectives, 
                         action_type, render_mode, obs_type, init_pos, 
                         start_dist, dist_paradigm, collision_rules, colormap)
        

    
        