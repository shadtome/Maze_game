from Maze_env.reward_functions.basic import BasicRewardFun 


class basicGame:
    def __init__(self):
        self.name = 'Basic Game'
        self.rewards_dist = {'agents':BasicRewardFun()}
        self.maze_environment = 'Maze_env/BasicMaze-v0'
        self.collision_rules = {}
        self.objectives = {}
        self.type_of_objects = ['agents']
        self.reward_wrapper = None

    def add_wrapper(self,env):
        if self.reward_wrapper!=None:
            return self.reward_wrapper(env, self.rewards_dist)
        else:
            return env