# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np
from collections import defaultdict
from util import Counter
from game import Grid
import numpy as np

# QLearnAgent
#
class QLearnAgent(Agent):

######################################################################################
################                    Initialization                   ################
######################################################################################
    # Constructor, called when we start running the
    def __init__(self, alpha=0.0001, epsilon=0, gamma=.5, numTraining = 2000):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # random.seed(1)

################ Parameters Controlling Learning ################

        # Initialized learning boolean (for functions which only (de)activate when learning is done)
        self.learning = True

        # Training episode counter
        self.training_episodes_won = 0
        self.training_episodes_lost = 0

        # Paying Initialization
        self.played = False

        # Build map
        self.map_built = False

        # Initial weights
        self.weights = [0, 0, 0]

        # Initialize rewards
        self.r = 0
        self.old_score = 0




######################################################################################
################     Get Wall positions & Map Graph                   ################
######################################################################################

    # Function to extract the coordinates of the walls of the map
    def wall_pos(self, state):
        walls = state.getWalls()
        w_pos = []
        index_x = 0
        index_y = 0
        for i in walls:
            for n in i:
                if n is True:
                    w_pos.append((index_x, index_y)) # Append tuple. (List of tuples)
                index_y +=1
            index_x += 1
            index_y = 0

        return w_pos

    # Function to extract a dictionary of vertices and children for possible paths in the maze
    def map_graph(self, w_pos):
        # Create defaultdict object
        map_graph = defaultdict(list)

        index_x = 0
        index_y = 0
        # Iterate over the entire grid
        for y in range(7):
            for x in range(7):
                if (index_x, index_y) not in w_pos: # If key is not a wall, proceed
                    # Look at moves up, down, right, and left and store in a list
                    down = (index_x, index_y - 1)
                    right = (index_x + 1, index_y)
                    left = (index_x - 1, index_y)
                    up = (index_x, index_y + 1)
                    directions = [down, right, left, up]
                    # Determine if any of those moves are walls
                    for i in directions:
                        # If they aren't, append them to the dictionary
                        if i not in w_pos:
                            map_graph[(index_x, index_y)].append(i)
                index_x += 1
            index_y += 1
            index_x = 0

        return map_graph



######################################################################################
################         DBS - Credit goes to Edd Mann                   ################
######################################################################################
# http://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
# Slight modification made in fixing of set differences (lines 5-6 in function)
    def bfs_paths(self, graph, start, goal):
        queue = [(start, [start])]
        while queue:
            (vertex, path) = queue.pop(0)
            l = set(graph[vertex])
            m = (set(path))
            for next in l - m:
                if next == goal:
                    yield path + [next]
                else:
                    queue.append((next, path + [next]))


######################################################################################
################         Move to Integer Conversion                   ################
######################################################################################
    # Function to transform directions to integers
    def MoveToInt(self, Direction):
        x = 0

        if Direction == 'East':
            x = 0
        elif Direction == 'South':
            x = 1
        elif Direction == 'West':
            x = 2
        elif Direction == 'North':
            x = 3
        else: x = 4

        return x


######################################################################################
################         Ineteger to Move Conversion                   ################
######################################################################################
    # Function to transform integers to directions
    def IntToMove(self, Int):
        x = 0

        if Int == 0:
            x = Directions.EAST
        elif Int == 1:
            x = Directions.SOUTH
        elif Int == 2:
            x = Directions.WEST
        elif Int == 3:
            x = Directions.NORTH
        else: x = Directions.STOP

        return x



######################################################################################
################        Acsessor functions for episodesSoFars         ################
######################################################################################
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts




######################################################################################
################ Parameters Controlling State Information (S')        ################
######################################################################################
    def Distance_to_Ghost(self, state, ghost, p_pos, graph):

        g_pos = []

        for i in ghost:
            # Need to iterate over the tuple to get the integers. Only one ghost.
            for n in i:
                # Add ghost positions in ghost list. One ghost so only necessary to store
                # the two integers
                g_pos.append(n)
            g_pos = tuple(g_pos) # Transform to tuple

            if cmp(g_pos, p_pos) != 0: # If tuple not identical
                d_ghost = len(list(self.bfs_paths(graph, p_pos, g_pos))[0]) - 1
            else: d_ghost = 0

            final_distance = np.log10(d_ghost + 0.1)
            if d_ghost >= 4:
                final_distance = 4

            return final_distance


    def Distance_to_food(self, state, food, p_pos, graph, ghost):
        ############## Food Information ##############
        # Obtaining the food coordinates
        f_pos = []
        index_x = 0
        index_y = 0
        for i in food:
            for n in i:
                if n is True:
                    f_pos.append([index_x, index_y])
                index_y +=1
            index_x += 1
            index_y = 0

        # Food remaining
        f_remaining = len(f_pos)

        ############## Distance Information ##############

        # Calculating the distance to food
        distances_food = []
        if f_pos:
            for i in f_pos:
                if cmp(tuple(i), p_pos) != 0:
                    d_food = len(list(self.bfs_paths(graph, p_pos, tuple(i)))[0]) -1
                else:
                    d_food = 0
                distances_food.append(d_food) # Not used in state info

        min_distance = min(distances_food)

        final_distance = (float(1)/float(0.9 + min_distance))

        ## Ghosts ##

        g_pos = []

        for i in ghost:
            # Need to iterate over the tuple to get the integers. Only one ghost.
            for n in i:
                # Add ghost positions in ghost list. One ghost so only necessary to store
                # the two integers
                g_pos.append(n)
            g_pos = tuple(g_pos) # Transform to tuple

            if cmp(g_pos, p_pos) != 0: # If tuple not identical
                d_ghost = len(list(self.bfs_paths(graph, p_pos, g_pos))[0]) - 1
            else: d_ghost = 0

            if d_ghost <= 6:
                final_distance = final_distance - 0.5

        return final_distance





        ############## General State Information ##############
    def tunnel(self, state, p_pos):
        # Is pacman in a tunnel?
        counter = 0
        x = 0

        # Calculate coordinates for up, down, right, and left of pacman
        down = (p_pos[0], p_pos[1] - 1)
        right = (p_pos[0] + 1, p_pos[1])
        left = (p_pos[0] - 1, p_pos[1])
        up = (p_pos[0], p_pos[1] + 1)
        directions = [down, right, left, up]
        # Calculate the wall coordinate locations
        walls = self.wall_pos(state)

        # If pacman has a wall up, down, left, or right of him, +1 counter
        for i in directions:
            if i in walls:
                counter += 1
        # If he has 3 walls around him, he's in a dangerous area. Append 1, else 0
        if counter == 3:
            x = 1
        else: x = 0

        ############## RETURN STATE INFORMATION ##############
        # Return State information

        return x




######################################################################################
################    Parameters Controlling Reward information (R')    ################
######################################################################################
    # Function for current reward (reward signal).
    # Comprised of subtracting the new score with the old score at every move
    def reward_signal(self, current_score, old_score):
        reward = float(current_score) - old_score


        return reward



######################################################################################
################    Parameters Controlling Legal Moves                ################
######################################################################################
    # Function to retrieve available actions in form of integer
    def AvailableActions(self, legal_):
        available_actions = []

        for i in legal_:
            available_actions.append(self.MoveToInt(i))

        return available_actions



######################################################################################
################    Get Action - Q-Learning Implementation            ################
######################################################################################
    def getAction(self, state):

        ############## Build network map ##############
        # This builds a network of positions pacman can move to based on a state.
        # Required for search algorithms. This is specific to this maze.
        if not self.map_built: # If map isn't built, build it
            self.w_pos = self.wall_pos(state) # Get wall locations
            self.map_graph = self.map_graph(self.w_pos) # Get network
            self.map_built = True # Turn off map building




        # Update weights
        if self.played: # Have we played yet?
            if self.learning:

                ############## Get Reward of state ##############
                self.r = self.reward_signal(state.getScore(), self.old_score)
                self.old_score = state.getScore() # Udate the old score for next time


                ############## Extract legal moves ##############
                # Get legal actions & remove STOP
                legal = state.getLegalPacmanActions()
                if Directions.STOP in legal:
                    legal.remove(Directions.STOP)

                # Convert available actions in form of int
                available_actions = self.AvailableActions(legal)


                ############## Calculate Max Q(s', a') ##############
                # Coordinates for grids around pacman
                south = (state.getPacmanPosition()[0], state.getPacmanPosition()[1] - 1)
                east = (state.getPacmanPosition()[0] + 1, state.getPacmanPosition()[1])
                west = (state.getPacmanPosition()[0] - 1, state.getPacmanPosition()[1])
                north = (state.getPacmanPosition()[0], state.getPacmanPosition()[1] + 1)
                directions = [south, east, west, north]

                possible_directions = [] # List with +1 moves to examine in legal
                direction_integer = [] # Mapping grid examined with direction

                # Examine available actions and store grids to look at for pacman
                if 0 in available_actions:
                    possible_directions.append(east)
                    direction_integer.append(0)
                if 1 in available_actions:
                    possible_directions.append(south)
                    direction_integer.append(1)
                if 2 in available_actions:
                    possible_directions.append(west)
                    direction_integer.append(2)
                if 3 in available_actions:
                    possible_directions.append(north)
                    direction_integer.append(3)

                # Stores Q(s', a') values, and their respective function scores
                Q_values = []
                f1_score = []
                f2_score = []

                # Looking at possible actions and compute Q values
                for i in possible_directions:
                    f1 = self.Distance_to_food(state, state.getFood(), i, self.map_graph, state.getGhostPositions())
                    f2 = self.Distance_to_Ghost(state, state.getGhostPositions(), i, self.map_graph)

                    f1_score.append(f1)
                    f2_score.append(f2)

                    Q = self.weights[0] + self.weights[1]*f1 + self.weights[2]*f2
                    Q_values.append(Q)


                # Choose the best action
                index = Q_values.index(max(Q_values))
                action = self.IntToMove(direction_integer[index])


                ############## Weight Updates ##############
                # UPDATE THE WEIGHTS

                difference = self.r + self.gamma * max(Q_values) - self.Qsa


                self.weights[0] = self.weights[0] + self.alpha * difference
                self.weights[1] = self.weights[1] + self.alpha * difference * f1_score[index]
                self.weights[2] = self.weights[2] + self.alpha * difference * f2_score[index]

                # Save the chosen action's previous function scores
                self.Qsa = max(Q_values) # This will be Q(s,a) after the move has been made

                # Save f values in case of death
                self.f1_death = f1_score[index]
                self.f2_death = f2_score[index]


        else:

            ############## Extract legal moves ##############
            # Get legal actions & remove STOP
            legal = state.getLegalPacmanActions()
            if Directions.STOP in legal:
                legal.remove(Directions.STOP)

            # Convert available actions in form of int
            available_actions = self.AvailableActions(legal)

            ############## Calculate Max Q(s', a') ##############
            # Coordinates for grids around pacman
            south = (state.getPacmanPosition()[0], state.getPacmanPosition()[1] - 1)
            east = (state.getPacmanPosition()[0] + 1, state.getPacmanPosition()[1])
            west = (state.getPacmanPosition()[0] - 1, state.getPacmanPosition()[1])
            north = (state.getPacmanPosition()[0], state.getPacmanPosition()[1] + 1)
            directions = [south, east, west, north]

            possible_directions = [] # List with +1 moves to examine in legal
            direction_integer = [] # Mapping grid examined with direction

            # Examine available actions and store grids to look at for pacman
            if 0 in available_actions:
                possible_directions.append(east)
                direction_integer.append(0)
            if 1 in available_actions:
                possible_directions.append(south)
                direction_integer.append(1)
            if 2 in available_actions:
                possible_directions.append(west)
                direction_integer.append(2)
            if 3 in available_actions:
                possible_directions.append(north)
                direction_integer.append(3)

            # Stores Q(s', a') values, and their respective function scores
            Q_values = []
            f1_score = []
            f2_score = []

            # Looking at possible actions and compute Q values
            for i in possible_directions:

                f1 = self.Distance_to_food(state, state.getFood(), i, self.map_graph, state.getGhostPositions())
                f2 = self.Distance_to_Ghost(state, state.getGhostPositions(), i, self.map_graph)

                f1_score.append(f1)
                f2_score.append(f2)

                Q = self.weights[0] + self.weights[1]*f1 + self.weights[2]*f2
                Q_values.append(Q)

            # Choose the best action
            index = Q_values.index(max(Q_values))
            action = self.IntToMove(direction_integer[index])


            # Save the function values
            self.Qsa = max(Q_values) # This will be Q(s,a) after the move has been made


            # Begin learning
            if self.learning:
                self.played = True

        # Exploration function
        if util.flipCoin(self.epsilon) and self.learning:
            choices = range(0, len(Q_values))
            index = random.choice(choices)
            action = self.IntToMove(direction_integer[index])
            self.Qsa = Q_values[index] # This will be Q(s,a) after the move has been made
        print self.weights

        return action






######################################################################################
################             Death & Learning                         ################
######################################################################################
    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):


        # print "A game just ended!"
        # Did pacman win or lose during training?
        if state.getScore() > 0:
            print "Pacman won!!!"
            self.training_episodes_won += 1
        else:
            print "Pacman has died."
            self.training_episodes_lost += 1



        ############## Update last state-action value ##############
        if state.getScore() < 0:
            self.r = -50
        else:
            self.r = 50
        difference = self.r - self.Qsa

        self.weights[0] = self.weights[0] + self.alpha * difference
        self.weights[1] = self.weights[1] + self.alpha * difference * self.f1_death
        self.weights[2] = self.weights[2] + self.alpha * difference * self.f2_death


        # Refresh Rewards
        self.r = 0
        self.old_score = 0

        # Refresh Qsa
        self.Qsa = 0


        ############## Re-Initialize all learning parameters ##############
        # Reinitializing boolean for starting condition
        self.played = False


        print self.weights
        ############## Book-keeping for learning ##############
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        print self.getEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)

            # Register that learning is complete. Leave on "True" to include continued exploration VS exploitation. Leave on "False" for agent to select Greedy actions based on Q-values
            self.learning = False
            # Switch off alpha adjustment (if on). Must remain on "False" at all times.
            self.alpha_adjustment = False

            # Number of training episodes won and lost
            print "Pacman has won %s training episodes \n" %self.training_episodes_won
            print "Pacman has lost %s training episodes\n" %self.training_episodes_lost
