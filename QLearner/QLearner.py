"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

# Mark Trinquero
# ML4T - Project 3, Deliverable 3


# RESOURCES CONSULTED:
# http://artint.info/html/ArtInt_265.html
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html
# https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node25.html
# http://www2.hawaii.edu/~chenx/ics699rl/grid/rl.html


# NOTES:
# Experience Tuple =  <s, a, s_prime, r>
# This provides one data point for the value of Q(s,a)
# This is equal to the current reward + discounted estimated future return
# The formula for computing Q for any state-action pair <s, a>, given an experience tuple <s, a, s', r>, is:

import math
import numpy as np
import random as rand
import pandas as pd




class QLearner(object):

    # WHAT IS Q ?
    # Q can be represented by a function or a table (in this class we will represent Q as a table)
    # Q represents the value of taking action a in state s (immediate + discounted reward)

    # Q TABLE
    # Q is a table of rewards for aviable actions/states
    # Q[s,a] represents: the TOTAL reward for action "a" at state "s"
    # The total reward = (the immediate reward) + (the discounted future reward)

    # POLICY 
    # PIE[s] = the policy/action for state s, leverages Q table to select best action
    # PIE[s] = argmax_a(Q[s,a])    --> this finds the 'a' that maximizes the total reward at that state


    #----------------------------------------------------------
    #---------------    Q-LEARNER CONSTRUCTOR    --------------
    #----------------------------------------------------------

    # The constructor QLearner() should reserve space for keeping track of Q[s, a] for the number of states and actions. 
    # It should initialize Q[] with uniform random values between -1.0 and 1.0. 
    # vhttp://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html

    # CONSTRUCTOR INPUT ARGUMENTS
    # 1) num_states (integer)       the number of states to consider
    # 2) num_actions (integer)      the number of actions available
    # 3) alpha (float)              the learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    # 4) gamma (float)              the discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    # 5) rar (float)                random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    # 6) radr (float)               random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    # 7) dyna (integer)             conduct this number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    # 8) verbose (boolean)          if True, your class is allowed to print debugging statements, if False, all printing is prohibited.


    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False): #Submission mode
        # verbose = True): #DEBUG MODE


        # Initilize QTable(states, actions) with random vales U(-1,1)
        # INT_QTABLE = 2 * np.random.random((num_states, num_actions)) - 1
        INT_QTABLE = np.random.uniform( low = -1, high = 1, size = (num_states, num_actions))
        self.q = INT_QTABLE

        # TODO: Initilize DynaTable

        # Set initial state and action locations
        self.s = 0
        self.a = 0

        # Initilize input params
        self.num_actions = num_actions      #number of availiable actions (4)
        self.num_states = num_states        #number of states in QTable  
        self.alpha = alpha                  #learning rate
        self.gamma = gamma                  #discount rate for value of future rewards
        self.rar = rar                      #probability of selectiong a random action at each stem
        self.radr = radr                    #the random action decay rate
        self.dyna = dyna                    # BOO DYNA!!
        self.verbose = verbose              #for debugging













    #--------------------------------------------------------------------------------------
    #-----------    Q-LEARNER QUERYSETSTATE   - Used to set up the initial state  ---------
    #--------------------------------------------------------------------------------------
    
    # querysetstate(s) A special version of the query method that sets the state to s, 
    # and returns an integer action according to the same rules as query(), 
    # but it does not execute an update to the Q-table. 

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        # if self.verbose: print "s =", s,"a =",action
        # Set initial state
        self.s = s

        # Get random number for decay comparison
        random_num = rand.random()  # random float between 0-1
        # Determine if learner should select random action or not
        if (random_num < self.rar):
            action = np.random.randint(0, self.num_actions) # random action
            self.rar = self.rar * self.radr #decay threshold for random decision
        else:
            action = np.argmax(self.q[s, :]) #best action at given state
            self.rar = self.rar * self.radr #decay threshold for random decision


        self.a = action     # update current action
        
        if self.verbose: print "s =", s,"a =",action
        return action









    #----------------------------------------------------
    #---------------    Q-LEARNER QUERY    --------------
    #----------------------------------------------------

    # query(s_prime, r) is the core method of the Q-Learner. 

    # It should keep track of the last state s and the last action a, 
    # then use the new information s_prime and r to update the Q table. 
    # The learning instance, or experience tuple is <s, a, s_prime, r>. 

    # query() should return an integer, which is the next action to take. 
    # Note that it should choose a random action with probability rar, 
    # and that it should update rar according to the decay rate radr at each step. 

    # QUERY INPUT ARGUMENTS
    # s_prime (integer) the the new state.
    # r (float) a real valued immediate reward.

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new reward
        @returns: The selected action (integer value of the next action to take)
        """

        # get current action
        action = self.a

        # WHERE THE MAGIC HAPPENS :)
        # stright from Q - learning lecture
        # UPDATE THE Q TABLE for the state/action changes (including discounted future reward)
        self.q[self.s,action] =( ((1 - self.alpha) * self.q[self.s, action]) + self.alpha * (r + self.gamma * self.q[s_prime, np.argmax(self.q[s_prime,:])]) )
        
        # UPDATE NEW STATE 
        self.s = s_prime
        # Get new action based on updated state
        action = self.querysetstate(self.s)
        self.a = action

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action






if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
