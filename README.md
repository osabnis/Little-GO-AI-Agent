# Little-AI-Go-Agent
This repository contains the code for a Little-Go AI agent developed for a programming assignment. The goal of the assignment is to create an AI agent that can play a simplified version of the game of Go, called Little-Go, on a 5x5 board.

### Overview
The assignment involves implementing AI techniques for Search, Game Playing, and Reinforcement Learning.  
The game of Go is an abstract strategy board game for two players, Black and White.  
The board is a 5x5 grid with intersections where stones can be placed.  
Stones are placed on the board, and the objective is to surround more territory than the opponent.  

### Game Rules
The game is played based on two simple rules: Liberty (No-Suicide) and KO.
- Liberty Rule: Stones must have at least one open point (liberty) directly adjacent or be part of a group with liberty.
- KO Rule: Prohibits immediate recapture of a captured stone.

AI Players
Your agent plays against other AI players, including Random Player, Greedy Player, Aggressive Player, Alphabeta Player, and QLearningPlayer.
Players are graded based on their performance in tournaments against these AI agents.

### Grading
The assignment has two stages.
In the first stage, your agent plays against basic AI players for grading.
In the second stage, your agent competes against advanced AI players, including a Q-Learning Player and a Championship Player.
Grading is based on win rates against these players, and points are awarded accordingly.

### Implementation
This AI Agent is built using Python. 
The agent reads the game state from input.txt and outputs its move to output.txt.
The AI techniques include Minimax with alpha-beta pruning.

### Usage
To run this agent, run the "build.sh" file.
Your agent (my_player3.py) will play games against other AI players - and at the end of the game, you will know your result!

### References
Wikipedia - Go (Game)
Wikipedia - Rules of Go
