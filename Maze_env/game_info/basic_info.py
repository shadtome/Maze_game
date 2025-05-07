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
        self.colormap = {'agent','tab10'}

    def update_game_info(self, **kwargs):
        """Update multiple attributes dynamically\n.
        Attributes:\n
        name\n
        rewards_dist\n
        maze_environment\n
        collision_rules\n
        objectives\n
        type_of_objects\n
        reward_wrapper\n
        colormap\n"""
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
                

    def add_wrapper(self,env):
        if self.reward_wrapper!=None:
            for rw in self.reward_wrapper:
                env = rw(env,self.rewards_dist)
            return env
        else:
            return env