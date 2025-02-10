# Maze_game

## The Game and Set Up
Suppose we are sitting in a corn maze and we can "sense" our goal at the end and we want to get to that goal.
But, there are other agents that started at different locations with their own goal locations.  You want to get to your location first before others do, but others can attack you.  So hence, it is a hunger games version in a maze.

Below we have an image of such a maze with circles as the agents and the squares are the goals with corresponding colors.

![maze set up](image.png)

## Training Data
To train our agents, we used the local information around each agent, based on if there is nothing, a wall, another agent, and their own goal.  It is represented as a rgb image with different gradients of the colors to represent how far it is and if its harder to see.  For an example, the following is a animation of their perspective.

- Put html animation here

