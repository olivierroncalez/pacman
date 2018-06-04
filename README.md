# Pacman
Modified agents built to play the pacman game using the Berkley API. Both a classifier-based approach as well as a QLearner and Function Approximation agent were used. 

## Getting Started

The Pacman code that is used was developed at UC Berkeley for their AI course. The homepage for the Berkeley AI Pacman projects is here:
http://ai.berkeley.edu/

The code only supports Python 2.7.

## Installing
Download the desired Python files and set your working directory to the folder using the command line. 

## Running Pacman agents

### Classifier Agent
Once you have set your working directory, type:

python pacman.py --pacman classifierAgent


and watch it run. This agent takes in a training data of binary states of length 25 with a labelled decision based on past player actions. The classifier uses custom coded Naive Bayes with cross validation metrics to evaluate performance. Code is also compared to scikit learn packages for comparative purposes. Note that as the state values are binary, this agent does not perform well and never wins. This is normal. You can read my interpretation of the agent in the associated project report. 

You can view the file good-moves.txt to see the reraining data. To create more custom data, run the code:

python pacman.py --p TraceAgent

which will write the new data to moves.txt.


### QLearning and Function Approximation Agents
Once you have set your working directory, type:

python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid 

for the QLearner, or type 

python pacman.py -p ApproxAgent -x 100 -n 110 -l smallGrid 

for function approximation agent. 

Both agents should obtain 100% success rate on the small grid. There are far too many possible state-action pairs to run QLearner on the medium or large grid, and function approximation has not yet been optimized for the large grid. Continued work will occur for the function approximation agent in order to optimize it in the future. Subscribe to updates to see any new developments. 

Note that the QLearner is trained on 2000 runs (although convergence generally occurs after 200-300 runs). Function approximation agent is trained on 100 runs. The terminal printouts for the QLearner agent are the state-action reward pairs, while for the function approximation agent they reflect the weights assigned to the linear function. 

A report was made detailing the process of the QLearner agent. Please reference for more information.
