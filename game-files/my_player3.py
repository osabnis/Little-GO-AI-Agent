# IMPORTING PACKAGES
import copy
import os
import time
import numpy as np

# GLOBAL VARIABLES
BOARD_SIZE = 5
NO_STONE = 0
BLACK_STONE = 1
WHITE_STONE = 2
SEARCH_DEPTH = 4
AI_SEARCH_DEPTH = 1
KOMI_PENALTY = BOARD_SIZE / 2
MIN_VALUE = -np.inf
MAX_VALUE = np.inf


class IOFunctions:
    """
    This class holds the functions related to Input and Output Functionalities
    """

    def __init__(self):
        self.cwd = os.getcwd()

    def read_input_file(self):
        """
        Read the file - input.txt
        :return: An array of strings containing the input data
        """
        path_to_file = os.path.join(self.cwd + "/input.txt")
        if not os.path.exists(path_to_file):
            raise "Input File not Found!"
        else:
            input_file = open(path_to_file, "r")
            input_file_data = input_file.read().splitlines()
            input_file.close()
            return input_file_data

    @staticmethod
    def parse_input(input_data_array):
        """
        Reads the input data array and parses it into color, previous state and current state
        :param input_data_array: The array of data from the input file
        :return: The color of play, previous state and current state of the go board
        """
        previous_state_array = []
        for x in input_data_array[1:6]:
            inter_prev_array = []
            for position in x:
                inter_prev_array.append(int(position))
            previous_state_array.append(inter_prev_array)
        current_state_array = []
        for x in input_data_array[6:]:
            inter_curr_array = []
            for position in x:
                inter_curr_array.append(int(position))
            current_state_array.append(inter_curr_array)
        return int(input_data_array[0]), previous_state_array, current_state_array

    def write_output_file(self, move_to_play):
        """
        Writes the move to be played to a file - output.txt
        :param move_to_play: The move to play
        :return: Nothing
        """
        path_to_file = os.path.join(self.cwd + "/output.txt")
        output_file = open(path_to_file, "w")
        if move_to_play == "PASS":
            output_file.write("PASS")
        else:
            output_file.write(str(move_to_play[0]) + "," + str(move_to_play[1]))
        output_file.close()


class GOBoardFunctions:
    """
    This class holds the functions required for the manipulation of the GO Board
    """

    def __init__(self):
        """
        Function called during initialization of the class
        """
        self.size = BOARD_SIZE
        self.dead_pieces = []
        self.komi = KOMI_PENALTY
        self.color = -1
        self.previous_state = [[]]
        self.current_state = [[]]

    def setup(self, color, previous_state, current_state):
        """
        Sets up the basic variables of the class - received from the input
        :param color: The color we are playing as
        :param previous_state: The previous state of the board when we last played
        :param current_state: The current state of the board after AI move
        :return: Nothing - updates class variables
        """
        self.color = color
        self.previous_state = previous_state
        self.current_state = current_state

    def captured_pieces(self, color, previous_state, current_state):
        """
        From the previous state, we find what are the stones missing to see what was captured by the AI
        :param color: The color we are playing as
        :param previous_state: The previous state of the board when we last played
        :param current_state: The current state of the board after AI move
        :return: Nothing - updates class variables
        """
        for row in range(self.size):
            for column in range(self.size):
                if previous_state[row][column] == color and current_state[row][column] != color:
                    self.dead_pieces.append((row, column))

    def compare_states(self, previous_state, current_state):
        """
        Compares the states of the board and returns a boolean variable to show if the state has changed. Used to detect "PASS" moves.
        :param previous_state: The previous state of the board when we last played
        :param current_state: The current state of the board after the AI move
        :return: Boolean - if the states have changed, False, else True
        """
        for row in range(self.size):
            for column in range(self.size):
                if previous_state[row][column] != current_state[row][column]:
                    return False
        return True

    def find_dead_stones(self, board_state, color):
        """
        Finds the stones who lose liberties and can be classified as dead stones
        :param board_state: Current state of the board
        :param color: Color we are playing as
        :return: A list that holds the coordinates of the dead stones
        """
        dead_stones = []
        for row in range(len(board_state)):
            for column in range(len(board_state)):
                if board_state[row][column] == color and not self.calculate_stone_liberty(board_state, row, column):
                    dead_stones.append((row, column))
        return dead_stones

    def del_dead_stones(self, color):
        """
        A function to delete the dead stones from the board
        :param color: Color we are playing as
        :return: A list that holds the coordinates of the dead stones
        """
        dead_stones = self.find_dead_stones(self.current_state, color)
        if not dead_stones:
            return []
        self.remove_dead_stones_from_board(dead_stones)
        return dead_stones

    def remove_dead_stones_from_board(self, positions):
        """
        A function that clears the dead stones and changes the coordinate to an empty position
        :param positions: A list of the coordinates of the dead stones
        :return: Nothing - updates the class variables
        """
        for piece in positions:
            self.current_state[piece[0]][piece[1]] = NO_STONE
        self.current_state = self.current_state

    def stone_count(self, color):
        """
        A function to count the number of stones that belong to a particular color on the board
        :param color: Which color do you want to count the number of stones for
        :return: Number of stones found for a particular color
        """
        no_of_stones = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.current_state[i][j] == color:
                    no_of_stones += 1
        return no_of_stones

    @staticmethod
    def find_neighbors(board_state, row, column):
        """
        A function to calculate the coordinates of the neighbors of a particular stone on the board
        :param board_state: The current state of the board
        :param row: The row number of the stone
        :param column: The column number of the stone
        :return: A list of valid neighbors for the stone
        """
        neighbors = []
        if row > 0:
            neighbors.append((row - 1, column))
        if len(board_state) - 1 > row:
            neighbors.append((row + 1, column))
        if column > 0:
            neighbors.append((row, column - 1))
        if len(board_state) - 1 > column:
            neighbors.append((row, column + 1))
        return neighbors

    def find_allied_neighbors(self, board_state, row, column):
        """
        A function to find if the neighbors of a particular stone are allies or enemies
        :param board_state: The current state of the board
        :param row: The row number of the stone
        :param column: The column number of the stone
        :return: A list of neighbors that are allies
        """
        neighbors = self.find_neighbors(board_state, row, column)
        allied_neighbors = []
        for stone in neighbors:
            if board_state[stone[0]][stone[1]] == board_state[row][column]:
                allied_neighbors.append(stone)
        return allied_neighbors

    def dfs(self, board_state, row, column):
        """
        A function that uses Depth First Search to find connected allied components from a particular stone
        :param board_state: The current state of the board
        :param row: The row number of the stone
        :param column: The column number of the stone
        :return: A list of all neighboring allies
        """
        stack = [(row, column)]
        allied_neighbors = []
        while stack:
            piece = stack.pop()
            allied_neighbors.append(piece)
            neighbor_allies = self.find_allied_neighbors(board_state, piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in allied_neighbors:
                    stack.append(ally)
        return allied_neighbors

    def calculate_stone_liberty(self, board_state, row, column):
        """
        A function to calculate if a particular stone has liberties
        :param board_state: The current state of the board
        :param row: The row number of the stone
        :param column: The column number of the stone
        :return: Boolean - if the stone has liberties or not
        """
        allied_neighbors = self.dfs(board_state, row, column)
        for x in allied_neighbors:
            neighbors = self.find_neighbors(board_state, x[0], x[1])
            for y in neighbors:
                if board_state[y[0]][y[1]] == NO_STONE:
                    return True
        return False

    def is_position_valid(self, board_state, row, column, color):
        """
        A function to check if the position is valid by checking if the position has liberties, is open and is not a repeat position
        :param board_state: The current state of the board
        :param row: The row number of the position
        :param column: The column number of the position
        :param color: The color we are playing as
        :return: Boolean - if the position is valid or not
        """
        if board_state[row][column] != NO_STONE or not 0 <= row < len(board_state) or not 0 <= column < len(board_state):
            return False
        sample = copy.deepcopy(self)
        sample_board = sample.current_state
        sample_board[row][column] = color
        sample.current_state = sample_board
        if sample.calculate_stone_liberty(sample.current_state, row, column):
            return True
        sample.del_dead_stones(3 - color)
        if not sample.calculate_stone_liberty(sample.current_state, row, column):
            return False
        else:
            if self.dead_pieces and self.compare_states(self.previous_state, sample.current_state):
                return False
        return True


class HelperFunctions:
    """
    A class that holds some auxiliary functions for the playing the game
    """

    def __init__(self, go_class):
        """
        Function called during initialization of the class
        """
        self.go = go_class

    def find_my_stones(self, board_state, color):
        """
        A function to find the coordinates of all the stones of a particular color
        :param board_state: The current state of the board
        :param color: The color we are playing as
        :return: A list of coordinates where the stones are
        """
        positions = []
        for row in range(self.go.size):
            for column in range(self.go.size):
                if board_state[row][column] == color:
                    positions.append((row, column))
        return positions

    def evaluation_function(self, color):
        """
        A function that evaluates the state of the board based on the number of stones
        :param color: The color we are playing as
        :return: A score for the particular state
        """
        white_stone_count = self.go.stone_count(2)
        white_stone_score = white_stone_count + self.go.komi
        black_stone_score = self.go.stone_count(1)
        if color == BLACK_STONE:
            score = black_stone_score - white_stone_score
            return score
        if color == WHITE_STONE:
            score = white_stone_score - black_stone_score
            return score

    def find_empty_positions_on_board(self, color):
        """
        A function that finds the valid empty positions on the board
        :param color: The color we are playing as
        :return: A list of positions that are empty and valid
        """
        empty_positions = []
        for row in range(self.go.size):
            for column in range(self.go.size):
                if self.go.is_position_valid(self.go.current_state, row, column, color):
                    empty_positions.append((row, column))
        return empty_positions

    def play_stone(self, board_state, row, column, color):
        """
        A function that plays a stone to a particular coordinate
        :param board_state: The current state of the board
        :param row: The row number of the stone
        :param column: The column number of the stone
        :param color: The color we are playing as
        :return: The new state of the board
        """
        if (row, column) in self.find_empty_positions_on_board(color):
            self.go.previous_state = copy.deepcopy(board_state)
            board_state[row][column] = color
            self.go.current_state = board_state
            return board_state
        else:
            return board_state


class AlphaBetaPruningFunctions:
    """
    This class holds the functions related to Alpha Beta Pruning
    """

    def __init__(self, go_class):
        """
        Function called during initialization of the class
        """
        self.helper = HelperFunctions(go_class)

    def min_level_value_calculator(self, board_state, color, depth, alpha, beta, running_time):
        """
        A function that acts as the value calculator for the MIN STEP in Alpha Beta Pruning
        :param board_state: The current state of the board
        :param color: The color we are playing as
        :param depth: The depth limit to search till
        :param alpha: The alpha value
        :param beta: The beta value
        :param running_time: A variable to keep track of run-time so we can stop if it crosses the 10 second time limit per move
        :return: The best move at MIN step and the MIN value
        """
        min_value = MAX_VALUE
        possible_moves = self.helper.find_empty_positions_on_board(color)
        end = time.time()
        if len(possible_moves) == 0 or depth == 0 or end - running_time > 9:
            return (-1, -1), self.helper.evaluation_function(color)
        else:
            for i in possible_moves:
                board_to_pass_each_time = copy.deepcopy(board_state)
                new_board = self.helper.play_stone(board_to_pass_each_time, i[0], i[1], color)
                self.helper.go.del_dead_stones(3 - color)
                if color == BLACK_STONE:
                    new_move, new_score = self.max_level_value_calculator(new_board, WHITE_STONE, depth - 1, alpha,
                                                                          beta, running_time)
                else:
                    new_move, new_score = self.max_level_value_calculator(new_board, BLACK_STONE, depth - 1, alpha,
                                                                          beta, running_time)
                if new_score < min_value:
                    min_value = new_score
                    best_move = i
                beta = min(new_score, beta)
                if beta <= alpha:
                    break
            return best_move, min_value

    def max_level_value_calculator(self, board_state, color, depth, alpha, beta, running_time):
        """
        A function that acts as the value calculator for the MAX step in Alpha Beta Pruning
        :param board_state: The current state of the board
        :param color: The color we are playing as
        :param depth: The depth limit to search till
        :param alpha: The alpha value
        :param beta: The beta value
        :param running_time: A variable to keep track of run-time so we can stop if it crosses the 10 second time limit per move
        :return: The best move at the MAX STEP and the MAX value
        """
        end = time.time()
        max_value = MIN_VALUE
        moves = self.helper.find_empty_positions_on_board(color)
        bad_moves = []
        if depth == 4:
            for i in moves:
                self.helper.go.current_state[i[0]][i[1]] = color
                ai_moves = self.helper.find_empty_positions_on_board(3 - color)
                for j in ai_moves:
                    self.helper.go.current_state[j[0]][j[1]] = 3 - color
                    dead_stones = self.helper.go.find_dead_stones(self.helper.go.current_state, color)
                    self.helper.go.current_state[j[0]][j[1]] = 0
                    if i in dead_stones and i not in bad_moves:
                        bad_moves.append(i)
                self.helper.go.current_state[i[0]][i[1]] = 0
            for x in bad_moves:
                if x in moves:
                    moves.remove(x)
        if len(moves) == 0 or depth == 0 or end - running_time > 9:
            return (-1, -1), self.helper.evaluation_function(color)
        else:
            for i in moves:
                board_to_pass_each_time = copy.deepcopy(board_state)
                new_board = self.helper.play_stone(board_to_pass_each_time, i[0], i[1], color)
                self.helper.go.del_dead_stones(3 - color)
                if color == BLACK_STONE:
                    new_move, new_score = self.min_level_value_calculator(new_board, WHITE_STONE, depth - 1, alpha,
                                                                          beta, running_time)
                else:
                    new_move, new_score = self.min_level_value_calculator(new_board, BLACK_STONE, depth - 1, alpha,
                                                                          beta, running_time)
                if new_score > max_value:
                    max_value = new_score
                    best_move = i
                alpha = max(new_score, alpha)
                if beta <= alpha:
                    break
            return best_move, max_value

    def choose_best_move(self, board_state, search_depth, color):
        """
        The main function that starts the alpha beta pruning recursion step
        :param board_state: The current state of the board
        :param search_depth: The search depth you must go till
        :param color: The color we are playing as
        :return: The X and Y coordinates of the best move as well as the score at the top of the tree
        """
        start = time.time()
        best_move, score = self.max_level_value_calculator(board_state, color, search_depth, MIN_VALUE, MAX_VALUE,
                                                           start)
        return best_move[0], best_move[1], score


class MyAgent:
    """
    This class holds the agent itself and its required functions
    """

    def __init__(self, go_class):
        """
        Function called during initialization of the class
        """
        self.go = go_class
        self.helper = HelperFunctions(go)
        self.minimax = AlphaBetaPruningFunctions(go)
        self.io = IOFunctions()

    @staticmethod
    def check_special_moves(move_list):
        """
        A function to check if the important positions are already played on the board - to maximize the area captured and maximize "eyes" on the board
        :param move_list: The current valid moves for the board
        :return: The first special move that is playable
        """
        special_positions = [(2, 2), (1, 1), (1, 3), (3, 1), (3, 3), (2, 0), (2, 4), (0, 2), (4, 2)]
        for special_position in special_positions:
            if special_position in move_list:
                return special_position

    def find_empty_positions_on_board(self):
        """
        A function to find all the empty positions on the board
        :return: A list of empty positions
        """
        empty_positions = []
        for row in range(self.go.size):
            for column in range(self.go.size):
                if self.go.current_state[row][column] == NO_STONE:
                    empty_positions.append((row, column))
        return empty_positions

    def find_kill_moves(self, spaces, color):
        """
        A function that creates a dictionary that holds the moves and the number of AI stones they can kill
        :param spaces: A list of empty valid positions available on the board
        :param color: The color we are playing as
        :return: A sorted dictionary that holds the moves and the number of the stones they kill
        """
        no_of_killable_stones = dict()
        for i in spaces:
            self.go.current_state[i[0]][i[1]] = color
            dead_stones = self.go.find_dead_stones(self.go.current_state, 3 - color)
            self.go.current_state[i[0]][i[1]] = NO_STONE
            if len(dead_stones) >= 1:
                no_of_killable_stones[i] = len(dead_stones)
        no_of_killable_stones = sorted(no_of_killable_stones, key=no_of_killable_stones.get, reverse=True)
        return no_of_killable_stones

    def check_kill_move_result(self, move_list, color):
        """
        A function that takes all the kill moves that returns the best possible move without repeating the previous state
        :param move_list: A list of moves to try from
        :param color: The color we are playing as
        :return: The best valid move
        """
        sample_board = copy.deepcopy(self.go.current_state)
        for one_move in move_list:
            sample_board[one_move[0]][one_move[1]] = color
            dead_stones = self.go.find_dead_stones(sample_board, 3 - color)
            for x in dead_stones:
                sample_board[x[0]][x[1]] = NO_STONE
            if one_move is not None and self.go.previous_state != sample_board:
                return one_move
        return 0

    def remove_bad_moves(self, move_set, color):
        """
        A function that finds the moves that are bad - cause us to lose stones - and removes them from the list of available moves
        :param move_set: A list of moves which we can play
        :param color: The color we are playing as
        :return: An updated list of moves that do not contain bad moves
        """
        bad_moves = []
        for i in move_set:
            self.go.current_state[i[0]][i[1]] = color
            ai_move = self.helper.find_empty_positions_on_board(3 - color)
            for j in ai_move:
                self.go.current_state[j[0]][j[1]] = 3 - color
                dead_stones = self.go.find_dead_stones(self.go.current_state, color)
                self.go.current_state[j[0]][j[1]] = NO_STONE
                if i in dead_stones:
                    bad_moves.append(i)
            self.go.current_state[i[0]][i[1]] = NO_STONE
        for x in bad_moves:
            if x in move_set:
                move_set.remove(x)
        return move_set

    def find_saving_moves(self, color):
        """
        A function that creates a dictionary of the moves that can be played to save our stones
        :param color: The color we are playing as
        :return: A sorted dictionary that holds the moves the number of stones they save
        """
        saving_moves = {}
        ai_moves = self.find_empty_positions_on_board()
        for i in ai_moves:
            self.go.current_state[i[0]][i[1]] = 3 - color
            our_dead_stones = self.go.find_dead_stones(self.go.current_state, color)
            self.go.current_state[i[0]][i[1]] = NO_STONE
            if len(our_dead_stones) >= 1:
                saving_moves[i] = len(our_dead_stones)
        saving_moves = sorted(saving_moves, key=saving_moves.get, reverse=True)
        return saving_moves

    @staticmethod
    def check_saving_move_result(saving_moves, move_list):
        """
        A function that checks if the saving move is also a good move to play
        :param saving_moves: The dictionary that holds the saving moves
        :param move_list: The list of good moves to play
        :return: The best saving move
        """
        for one_move in saving_moves:
            if one_move is not None and one_move in move_list:
                return one_move
        return 0

    def play(self, piece_type):
        """
        A function that combines everything together and acts as the agent's main code
        :param piece_type: The color we are playing as
        :return: The move to make
        """
        # Find the empty positions on the board
        empty_positions = self.find_empty_positions_on_board()
        # Find the moves that kill the most stones and play those moves = GREEDY STEP
        killable_moves = self.find_kill_moves(empty_positions, piece_type)
        output = self.check_kill_move_result(killable_moves, piece_type)
        if output != 0:
            return output
        # Find possible moves that can be played and then remove the moves that cause our stones to die
        move_set = self.helper.find_empty_positions_on_board(piece_type)
        good_moves = self.remove_bad_moves(move_set, piece_type)
        if len(good_moves) == 0:
            return "PASS"
        # Find moves that can save our stones the most and play those moves
        saving_moves = self.find_saving_moves(piece_type)
        output = self.check_saving_move_result(saving_moves, good_moves)
        if output != 0:
            return output
        # Check if special moves are available that can be played to capture most area
        if len(good_moves) >= 15:
            special_move = self.check_special_moves(good_moves)
            return special_move
        # Start the Alpha Beta Pruning step
        # Find the opponent's best move and play it
        ai_current_state = copy.deepcopy(self.go.current_state)
        x, y, score = self.minimax.choose_best_move(ai_current_state, AI_SEARCH_DEPTH, 3 - piece_type)
        self.go.current_state[x][y] = 3 - piece_type
        # Find the best move that can kill the most stones and play those move = GREEDY STEP
        empty_spaces = self.find_empty_positions_on_board()
        kill_moves = self.find_kill_moves(empty_spaces, piece_type)
        self.go.current_state[x][y] = NO_STONE
        if len(kill_moves) != 0:
            sorted_moves = self.remove_bad_moves(kill_moves, piece_type)
            for i in sorted_moves:
                if i in good_moves:
                    return i
        # If not, find the best move using Alpha Beta Pruning and play it
        x, y, score = self.minimax.choose_best_move(go.current_state, SEARCH_DEPTH, piece_type)
        return x, y


if __name__ == "__main__":
    try:
        input_data = IOFunctions().read_input_file()
        stone_type, last_state, present_state = IOFunctions().parse_input(input_data)
        go = GOBoardFunctions()
        go.setup(stone_type, last_state, present_state)
        go.captured_pieces(stone_type, last_state, present_state)
        agent = MyAgent(go)
        move = agent.play(stone_type)
        if move is None:
            move = "PASS"
        IOFunctions().write_output_file(move)
    except:
        move = "PASS"
        IOFunctions().write_output_file(move)
