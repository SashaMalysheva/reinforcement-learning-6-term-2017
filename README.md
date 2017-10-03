Considering the attached lion-and-cows gridworld domain. S is the start 
position of the lion, and G is the goal position. Positions A through F 
are cows that the lion wants to pick up and bring to G (which is a 
terminal state, i.e. the episode ends when the lion reaches that 
position). The more cows the lion brings to G, the better. 

Implementation of an abstraction of this domain in the form of an abstract MDP. 
Then use the value function computed on this abstract MDP to shape the 
reward of Q learning on the full (i.e. detailed low-level) MDP, using 
potential-based reward shaping. Compare the performance of this approach 
with regular Q learning. 
