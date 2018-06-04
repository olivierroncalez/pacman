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

# QLearnAgent
#
class QLearnAgent(Agent):

######################################################################################
################                    Initialization                   ################
######################################################################################
    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=1, numTraining = 2000):
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
        # Large reward for exploration
        self.L_Reward = 510
        # Number of times to visit each state (triggers large reward if not met)
        self.Ne = 1
        # Table of Q values indexed by state and action. New states/actions will = 0 initially
        self.Q = {}
        # Counter to keep track of the state-actions
        self.Nsa = Counter()
        # Previous state, action, reward
        self.s = None
        self.a = None
        self.r = None

        # Initialize boolean to make the first move
        self.played = False
        # Initialized learning boolean (for functions which only (de)activate when learning is done)
        self.learning = True
        # Adjust alpha based on Nsa? If set to True, uses the specific lamba alpha adjustment.
        self.alpha_adjustment = True
        self.adjusted_alpha = lambda n: 1./(1+n)
        # Training episode counter
        self.training_episodes_won = 0
        self.training_episodes_lost = 0

        # Build map. Required to build the network map only once.
        self.map_built = False

        # Exploration method used (am I using function exploration, or epsilon?)
        # Only turn one on at a time. Final solution uses function exploration
        self.function_exploration = True
        self.epsilon_exploration = False




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


    # Function to extract a dictionary of nodes and leaves for possible paths in the maze
    def map_graph(self, w_pos):
        # Create defaultdict object
        map_graph = defaultdict(list)

        index_x = 0
        index_y = 0
        # Iterate over the entire grid
        for y in range(7): # These numbers are specific to the dimensions of the small grid
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
################         DBS - Credit goes to Edd Mann (referenced in report)                ################
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
    # Function to prepare state-action key (state signal).
    # Not all of the information found in this function is used. Any usable information is appeneded to the empty list called "state_list" which is what is returned at the end

    # Parameters are used to extract the specific information within the function.
    # Inputs are pacman's and the ghost's positions, the food grid, the game graph (constructed),
    # and the general state.
    def key(self, pacman, ghost, food, graph, state):

        # This initializes empty lists to store integer information

        # Main list which comprises the key representing a particular state
        state_list = list()
        # Ghost position list
        g_pos = list()
        # Pacman position list
        p_pos = list()
        # food positions
        f_pos = list()


        ############## Pacman Information ##############
        # Getting pacman positions
        for i in pacman:
            # Add Pacman's positions
            p_pos.append(i)
        p_pos = tuple(p_pos) # Turn into tuple



        ############## Ghost Information ##############
        for i in ghost:
            # Need to iterate over the tuple to get the integers. Only one ghost.
            for n in i:
                # Add ghost positions in ghost list. One ghost so only necessary to store
                # the two integers
                g_pos.append(n)
            g_pos = tuple(g_pos) # Transform to tuple
            state_list.append(g_pos) # ADDING TO STATE INFORMATION



        ############## Food Information ##############
        # Obtaining the food coordinates
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
        state_list.append(f_remaining) # ADDING TO STATE INFORMATION



        ############## Distance Information ##############


        ### DISTANCE TO GHOST (not used) ###
        # Calulating grid distance to ghost (shortest distance) BFS algorithm w/ game graph (not sued)
        dist_to_ghost = len(list(self.bfs_paths(graph, p_pos, g_pos))[0])-1


        ### DISTANCE TO FOOD (not used) ###
        distances_food = []
        if f_pos:
            for i in f_pos:
                d_food = len(list(self.bfs_paths(graph, p_pos, tuple(i)))[0]) -1
                distances_food.append(d_food) # Not used in state info


        ############## Location Information ##############

        ### GHOST DIRECTION ###

        # Is the ghost coming from the east, north, west, or south? List is in that order.
        possible_directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        dir_ghost_0 = list(self.bfs_paths(graph, p_pos, g_pos))[0][0] # Where is the ghost now
        dir_ghost_1 = list(self.bfs_paths(graph, p_pos, g_pos))[0][1] # Where could the ghost move next if it were to go to the shortest path towards you
        direction_g = tuple(np.array(dir_ghost_1) - np.array(dir_ghost_0)) # Which way would the ghost need to move if he were to take the shortest path to pacman? See possible_directions.
        index = 0
        # Check which direction the ghost would need to move if following closest distance.
        for i in possible_directions:
            if i == direction_g:
                state_list.append(index)
            index += 1
            # Results in an integer between 0-3 based on the ghost's movement towards pacman.



        ### FOOD DIRECTION ###

        ## In which direction should pacman move for the closest food?
        # Get index of min distance to food
        i_food = distances_food.index(min(distances_food))
        # Find coordinate location of min food
        closest_food = f_pos[i_food]
        # Calculate direction of first step towards food if following shortest path
        # Note that this follows the same idea between estimating the ghost's directions. This time however, food is static, so pacman would definitevely know where to move.

        # Find the next coordinate pacman should go to if he takes the shortest path to food
        dir_loc_food = list(self.bfs_paths(graph, p_pos, tuple(closest_food)))[0][1]
        # Compute the direction required to move closer to food
        loc_food = tuple(np.array(p_pos) - np.array(dir_loc_food))
        # Append the direction of movement suggested to move closer to food
        index = 0
        for i in possible_directions:
            if i == loc_food:
                state_list.append(index)
            index += 1


        ############## General State Information ##############

        ### TUNNEL INFORMATION ###
        # Is pacman in a tunnel?
        counter = 0
        # Calculate coordinates for up, down, right, and left of pacman
        down = (p_pos[0], p_pos[1] - 1)
        right = (p_pos[0] + 1, p_pos[1])
        left = (p_pos[0] - 1, p_pos[1])
        up = (p_pos[0], p_pos[1] + 1)
        directions = [down, right, left, up]
        # Calculate the wall coordinate locations
        walls = self.wall_pos(state)

        # If pacman has a wall up, down, left, or right of him, add +1 to counter
        for i in directions:
            if i in walls:
                counter += 1
        # If he has 3 walls around him, he's in a dangerous area. Append 1, else 0
        if counter == 3:
            state_list.append(1)
        else: state_list.append(0)



        ############## RETURN STATE INFORMATION ##############
        # Return State information

        return tuple(state_list)




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




        ############## Extract legal moves ##############
        # Get legal actions & remove STOP
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Convert available actions in form of int
        available_actions = self.AvailableActions(legal)




        ############## Extract State information S' ##############
        # Get CURRENT state info S' (in form of key)
        self.s_cur = self.key(state.getPacmanPosition(), state.getGhostPositions(), state.getFood(), self.map_graph, state)




        ############## Update Q-Table ##############
        # If this is not the first action we make in a game. If it is, skip to "else"
        if self.played:
            # If this is the first time we have seen that state, initialize key-value pairs of all possible actions to 0. If not the dictionary will be empty and not function. Allows us to add states as we see them.
            for i in available_actions:
                if self.Q.get((self.s_cur, i)) == None:
                    self.Q[self.s_cur, i] = 0


            # Get the current reward (R')
            self.r_cur = self.reward_signal(state.getScore(), self.old_score)


            # Update old score
            self.old_score = state.getScore()


            # Increment the state/action pair that we were in previously. (Nsa += 1)
            self.Nsa[(self.s, self.a)] += 1


            # Calculate alpha adjustment based on Nsa (if activated)
            if self.alpha_adjustment:
                self.alpha = self.adjusted_alpha(self.Nsa[(self.s, self.a)])
            else: # Use regular alpha if not active
                self.alpha = self.alpha


            # Update the Q Table for previous state/action pair
            self.Q[(self.s, self.a)] = self.Q[(self.s, self.a)] + self.alpha * (self.r + self.gamma * max(self.Q[(self.s_cur, i)] for i in available_actions) - self.Q[(self.s, self.a)])



        else:
            # This code is only run once at the beginning of each game.

            # Initialize the current reward for starting
            self.r_cur = state.getScore()
            self.old_score = state.getScore() # "Old score" is the same as current score at t = 0.
            # Initialize playing state. We will not come here again until the new game.
            self.played = True
            # Ensure dictionary is not empty for current starting position and available actions.
            # They are initialized to 0.
            for i in available_actions:
                if self.Q.get((self.s_cur, i)) == None:
                    self.Q[self.s_cur, i] = 0


        ############## Update S, R ##############
        # Adjust state, and reward. We have already updated the Q table so this will only be relevant at the next table update (given we survived an extra move)
        self.s = self.s_cur
        self.r = self.r_cur


        ############## Chosing argmax Q(s', a') and updating A ##############

        self.scores = [] # Will keep track of all rewards for each action in legal

        ############## If using function exploration
        if self.function_exploration:
            ## Adjust action (need the arg max Q(s', a'))
            # Obtaining the action which maximizes the rewards. Examine all possible actions
            # and store their Q-values in a list
            for i in available_actions:
                # If the state I can go to hasn't been visited enough time, incentivise it properly using the large reward. Agent must also be in a state of learning
                if (self.Nsa[(self.s_cur, i)] < self.Ne) and self.learning:
                    self.scores.append(self.L_Reward)
                # If it has, get the true calculated utility of that state
                else:
                    self.scores.append(self.Q[(self.s_cur, i)])

            # Verify that the number of scores which are equal to max score. This will be used to make a random choice if we have several unseen state-action pairs.
            counter = 0 # Serves as a counter and index for max score
            max_score_index = []
            for i in self.scores:
                if i == max(self.scores):
                    max_score_index.append(counter)
                counter += 1

            # Extract the index for the highest score. Either randomly when there is more than one max score, or the first element in the list when there is only 1. This is needed to map the score back with the action which produced it.
            if max_score_index > 1:
                max_ = random.choice(max_score_index)
            else:
                max_ = max_score_index[0]


            # Map the index corresponding to the highest score back to its respective action in available_actions
            self.a = available_actions[max_]
            # Convert int action to actual action and return move.
            action = self.IntToMove(self.a)






        ############## If using epsilon exploration (not used)
        if self.epsilon_exploration:
            for i in available_actions:
                self.scores.append(self.Q[(self.s_cur, i)])

            # If less than epsilon, and we're learning, make a random choice
            if util.flipCoin(self.epsilon) and self.learning:
                self.a = random.choice(available_actions)
            else:
                max_ = self.scores.index(max(self.scores))
                self.a = available_actions[max_]
            action = self.IntToMove(self.a)





        ############## Return Action Arg Max Q(S', A') ##############
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
            print self.scores
            self.training_episodes_won += 1
        else:
            print "Pacman has died."
            print self.scores
            self.training_episodes_lost += 1




        ############## Update last state-action value ##############
        # Adding reward to the state. Last known state and action before move (discounted according to the usual update rules)
        self.Q[self.s, self.a] = self.Q[self.s, self.a] + self.alpha * (self.r + self.gamma * state.getScore() - self.Q[self.s, self.a])

        # Incrementing last seen state/action pair.
        self.Nsa[(self.s, self.a)] += 1


        ############## Re-Initialize all learning parameters ##############
        # Reinitializing boolean for starting condition
        self.played = False

        # Clearing the last action reward, and state for fresh learning experience
        self.r = None
        self.a = None
        self.s = None



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
            # Printing to get a better idea of how long training lasts (for my sanity)
            print self.Q

            # Register that learning is complete. Leave on "True" to include continued exploration VS exploitation after learning. Leave on "False" for agent to select Greedy actions based on Q-values after learning
            self.learning = False
            # Switch off alpha adjustment (if on). Must remain on "False" at all times.
            self.alpha_adjustment = False

            # Number of training episodes won and lost
            print "Pacman has won %s training episodes \n" %self.training_episodes_won
            print "Pacman has lost %s training episodes\n" %self.training_episodes_lost
