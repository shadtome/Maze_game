from torch.utils.data import Dataset
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.generation import generators
from maze_dataset.plotting import MazePlot
import matplotlib.pyplot as plt

class Maze_dataset(Dataset):
    def __init__(self,num_mazes,shape = (10,10),maze_type = 'dfs', **kwargs):
        """Used to generate mazes
            num_mazes: the number of mazes to generate
            shape: shape of the maze (h,w)
            type: the type of maze generation.
                    - 'dfs': depth first search
                        - lattice_dim
                        - accessible_cells
                        - max_tree_depth
                        - do_forks
                    - 'wilson': wilson's algorithm
                    - 'percolation': percolation maze generation
                        - p
                        - lattice_dim 
                    - 'prim': 
                        - lattice_dim
                        - accessible_cells
                        - max_tree_depth
                        - do_forks"""
        self.shape = shape
        self.maze_type = maze_type
        self.mazes = [self.generate_maze(shape, **kwargs) for _ in range(num_mazes)]
        

    def generate_maze(self,shape, **kwargs):
        """Generate mazes based on the type of maze generation
        shape: shape of the maze (h,w)"""
        if self.maze_type == 'dfs':
            maze = LatticeMazeGenerators.gen_dfs(
                grid_shape = shape,
                lattice_dim=2,
                accessible_cells=None,
                max_tree_depth=None,
                start_coord=None,
                do_forks=True
            )
            return maze
        if self.maze_type == 'wilson':
            maze = LatticeMazeGenerators.gen_wilson(
                grid_shape=shape
            )
            return maze
        if self.maze_type == 'percolation':
            maze = LatticeMazeGenerators.gen_percolation(
                grid_shape=shape,
                p=1,
                lattice_dim = 2,
                start_coord=None,
            )
            return maze
        if self.maze_type == 'prim':
            maze = LatticeMazeGenerators.gen_prim(
                grid_shape=shape,
                lattice_dim=2,
                accessible_cells=None,
                max_tree_depth=None,
                do_forks=True,
                start_coord=None
            )

    def __len__(self):
        return len(self.mazes)
    
    def __getitem__(self,idx):
        return self.mazes[idx]
    

    def show_maze(self,idx):
        MazePlot(self.__getitem__(idx)).plot()

        plt.show()