import random
import numpy as np
import pickle
import multiprocessing
from tqdm import tqdm
import time


# -----------------------------------------CODE FOR MCTS------------------------------------------------#


def update_board(board_temp, color, column):
    # this is a function that takes the current board status, a color, and a column and outputs the new board status
    # columns 0 - 6 are for putting a checker on the board: if column is full just return the current board...this should be forbidden by the player

    # the color input should be either 'plus' or 'minus'

    board = board_temp.copy()
    ncol = board.shape[1]
    nrow = board.shape[0]

    colsum = board[:, column, 0].sum() + board[:, column, 1].sum()
    row = int(5 - colsum)
    if row > -0.5:
        # if color == "plus":
        #     board[row, column] = 1
        # else:
        #     board[row, column] = 1
        if color == "plus":
            board[row, column, 0] = 1  # Set 'plus' layer
            board[row, column, 1] = 0  # Clear 'minus' layer
        elif color == "minus":
            board[row, column, 1] = 1  # Set 'minus' layer
            board[row, column, 0] = 0  # Clear 'plus' layer
    else:
        print(f"Column {column} is full. No changes made.")

    return board


def check_for_win(board, col):
    """
    Check if the last move resulted in a win on a 6x7x2 board.

    Parameters:
        board: np.array (6x7x2)
            The current board state.
        col: int
            The column where the last move was made.

    Returns:
        str: The type of win ('v-plus', 'h-plus', 'd-plus', etc.) or 'nobody'.
    """
    nrow, ncol, _ = board.shape

    # Find the row of the last move
    colsum = board[:, col, 0].sum() + board[:, col, 1].sum()
    row = int(nrow - colsum)

    for layer, player in zip([0, 1], ["plus", "minus"]):
        # Check vertical win
        if row + 3 < nrow:
            if all(board[row + i, col, layer] == 1 for i in range(4)):
                return f"v-{player}"

        # Check horizontal win
        for c_start in range(max(0, col - 3), min(col + 1, ncol - 3)):
            if all(board[row, c_start + i, layer] == 1 for i in range(4)):
                return f"h-{player}"

        # Check diagonal win (bottom-left to top-right)
        for i in range(-3, 1):
            if all(
                0 <= row + i + j < nrow
                and 0 <= col + i + j < ncol
                and board[row + i + j, col + i + j, layer] == 1
                for j in range(4)
            ):
                return f"d-{player}"

        # Check diagonal win (top-left to bottom-right)
        for i in range(-3, 1):
            if all(
                0 <= row - i - j < nrow
                and 0 <= col + i + j < ncol
                and board[row - i - j, col + i + j, layer] == 1
                for j in range(4)
            ):
                return f"d-{player}"

    return "nobody"


def find_legal(board):
    """
    Find all legal moves (columns) on a 6x7x2 board.

    Parameters:
        board: np.array (6x7x2)
            The current board state.

    Returns:
        list: A list of integers representing legal columns.
    """
    legal = [
        i for i in range(board.shape[1]) if board[0, i, 0] == 0 and board[0, i, 1] == 0
    ]
    return legal


def look_for_win(board_, color):
    """
    Check if the current player has a winning move.

    Parameters:
        board_: np.array (6x7x2)
            The current board state.
        color: str ('plus' or 'minus')
            The player's color to check for a winning move.

    Returns:
        int: The column index of a winning move, or -1 if no such move exists.
    """
    board_ = board_.copy()
    legal = find_legal(board_)
    winner = -1

    for m in legal:
        # Simulate the move
        bt = update_board(board_.copy(), color, m)

        # Check if the move results in a win
        wi = check_for_win(bt, m)
        if wi[2:] == color:
            winner = m
            break

    return winner


def find_all_nonlosers(board, color):
    """
    Find all moves that do not immediately lead to a loss for the current player.

    Parameters:
        board: np.array (6x7x2)
            The current board state.
        color: str ('plus' or 'minus')
            The player's color.

    Returns:
        list: A list of column indices representing non-losing moves.
    """
    # Determine opponent's color
    opp = "minus" if color == "plus" else "plus"

    # Get all legal moves
    legal = find_legal(board)

    # Simulate each legal move for the current player
    poss_boards = [update_board(board, color, l) for l in legal]

    # Simulate opponent's response for each resulting board
    poss_legal = [find_legal(b) for b in poss_boards]
    allowed = []

    for i in range(len(legal)):
        # Check if any of the opponent's moves result in a win
        opponent_wins = [
            j
            for j in poss_legal[i]
            if check_for_win(update_board(poss_boards[i], opp, j), j) != "nobody"
        ]

        # If no winning move for the opponent, this move is allowed
        if len(opponent_wins) == 0:
            allowed.append(legal[i])

    return allowed


def back_prop(winner, path, color0, md):
    """
    Perform backpropagation in the MCTS process.

    Parameters:
        winner: str
            The winner of the rollout ('v-plus', 'h-minus', etc., or 'nobody').
        path: list
            A list of board states (flattened tuples) visited during the simulation.
        color0: str ('plus' or 'minus')
            The color of the player initiating the simulation.
        md: dict
            The MCTS dictionary storing visit counts and win scores for each state.
    """
    for i, board_temp in enumerate(path):
        # Increment the visit count for the current board state
        md[board_temp][0] += 1

        # Determine if the current player was the winner
        if winner[2:] == color0:
            # If the winner is the same as the current player
            if i % 2 == 1:
                md[board_temp][1] += 1  # Increment win score
            else:
                md[board_temp][1] -= 1  # Penalize for opponent's win
        elif winner[2:] == "e":  # Handle ties
            # Do nothing for ties, or optionally modify this for tie scoring
            pass
        else:
            # If the opponent was the winner
            if i % 2 == 1:
                md[board_temp][1] -= 1  # Penalize for the player's loss
            else:
                md[board_temp][1] += 1  # Reward for delaying the loss


def rollout(board, next_player):
    """
    Perform a simulation (rollout) from the current board state.

    Parameters:
        board: np.array (6x7x2)
            The current board state.
        next_player: str ('plus' or 'minus')
            The player to start the rollout.

    Returns:
        str: The result of the rollout ('v-plus', 'h-minus', 'tie', or 'nobody').
    """
    winner = "nobody"
    player = next_player

    while winner == "nobody":
        # Find all legal moves
        legal = find_legal(board)
        if len(legal) == 0:
            # No legal moves, it's a tie
            winner = "tie"
            return winner

        # Choose a random legal move
        move = random.choice(legal)

        # Update the board with the chosen move
        board = update_board(board, player, move)

        # Check if this move results in a win
        winner = check_for_win(board, move)

        # Switch player
        player = "minus" if player == "plus" else "plus"

    return winner


def mcts(board_temp, color0, nsteps):
    """
    Perform Monte Carlo Tree Search (MCTS) to determine the best move.

    Parameters:
        board_temp: np.array (6x7x2)
            The current board state.
        color0: str ('plus' or 'minus')
            The player making the move.
        nsteps: int
            The number of iterations for MCTS. Higher values improve accuracy.

    Returns:
        int: The column index of the best move.
    """
    # Make a copy of the board
    board = board_temp.copy()

    ##############################################
    # Immediate win or blocking logic
    winColumn = look_for_win(board, color0)  # Check for a winning column
    if winColumn > -0.5:
        return winColumn  # Play the winning move immediately

    legal0 = find_all_nonlosers(
        board, color0
    )  # Avoid moves that allow the opponent to win
    if len(legal0) == 0:  # If no safe moves, play any legal move
        legal0 = find_legal(board)
    ##############################################

    # Initialize MCTS dictionary to store visit counts and scores
    mcts_dict = {tuple(board.ravel()): [0, 0]}

    # Run MCTS iterations
    for _ in range(nsteps):
        color = color0
        winner = "nobody"
        board_mcts = board.copy()
        path = [tuple(board_mcts.ravel())]  # Track the path of states

        while winner == "nobody":
            legal = find_legal(board_mcts)
            if len(legal) == 0:  # If no legal moves, it's a tie
                winner = "tie"
                back_prop(winner, path, color0, mcts_dict)
                break

            # Generate possible boards for legal moves
            board_list = [
                tuple(update_board(board_mcts, color, col).ravel()) for col in legal
            ]
            for bl in board_list:
                if bl not in mcts_dict:
                    mcts_dict[bl] = [0, 0]

            # Compute UCB1 scores for each move
            ucb1 = np.zeros(len(legal))
            for i, state in enumerate(board_list):
                num, score = mcts_dict[state]
                if num == 0:
                    ucb1[i] = 10 * nsteps  # Prioritize unexplored states
                else:
                    ucb1[i] = score / num + 2 * np.sqrt(
                        np.log(mcts_dict[path[-1]][0]) / num
                    )

            # Choose the move with the highest UCB1 score
            chosen = np.argmax(ucb1)
            board_mcts = update_board(board_mcts, color, legal[chosen])
            path.append(tuple(board_mcts.ravel()))
            winner = check_for_win(board_mcts, legal[chosen])

            # Backpropagate if a winner is found
            if winner[2:] == color:
                back_prop(winner, path, color0, mcts_dict)
                break

            # Switch player
            color = "minus" if color == "plus" else "plus"

            # Perform a rollout if the state has not been visited
            if mcts_dict[tuple(board_mcts.ravel())][0] == 0:
                winner = rollout(board_mcts, color)
                back_prop(winner, path, color0, mcts_dict)
                break

    # Evaluate the best move based on the MCTS dictionary
    maxval = -np.inf
    best_col = -1
    for col in legal0:
        board_temp = tuple(update_board(board, color0, col).ravel())
        num, score = mcts_dict[board_temp]
        if num == 0:
            compare = -np.inf
        else:
            compare = score / num
        if compare > maxval:
            maxval = compare
            best_col = col

    return best_col


# --------------------------------------PARALLEL PROCESSING FUNCTIONS--------------------------------------#
def save_dataset_parallel(dataset, filename):
    """Save the dataset to a file."""
    with open(filename, "wb") as file:
        pickle.dump(dataset, file)


############## FOR PROGRESSIVE SKILL LEVELING ###########
# def generate_dataset_progressive_opponent_parallel(
#     game_ids, player2_skill_levels, progress_queue
# ):
#     """
#     Generate a dataset where Player 1's (Plus) skill is random, and Player 2's skill follows a precomputed progression.
#     Sends updates to `progress_queue` whenever a game finishes.
#     """
#     dataset = []  # Store game states and moves

#     for game_index, game_id in enumerate(game_ids):
#         player2_skill = player2_skill_levels[game_index]
#         fixed_skill = random.randint(150, 2500)  # Randomize Player 1's skill

#         board = np.zeros((6, 7, 2), dtype=int)  # Initialize empty board
#         game_over = False
#         player = "plus"

#         while not game_over:
#             legal_moves = find_legal(board)
#             if not legal_moves:
#                 print(f"Game {game_id}: Tie!")
#                 progress_queue.put(1)  # Send update to progress bar
#                 break

#             if np.all(board == 0) and random.random() < 0.25:
#                 move = random.choice(legal_moves)
#             else:
#                 nsteps = fixed_skill if player == "plus" else player2_skill
#                 move = mcts(board, player, nsteps)

#             dataset.append(
#                 (game_id, board.copy(), move, fixed_skill, player2_skill, player)
#             )
#             board = update_board(board, player, move)

#             result = check_for_win(board, move)
#             if result != "nobody":
#                 print(f"Game {game_id}: Winner = {result}")
#                 progress_queue.put(1)  # Send update to progress bar
#                 game_over = True

#             player = "minus" if player == "plus" else "plus"

#     return dataset


################## FOR RANDOM SKILL LEVELING + RANDOM INITIAL MOVES ##################
def generate_dataset_randomopponents_randomfirstmoves_parallel(
    game_ids, progress_queue
):
    """
    Generate a dataset where Player 1's (Plus) skill is random, and Player 2's skill is randomized.
    Adds randomness to the first few moves based on a probability distribution.
    Sends updates to `progress_queue` whenever a game finishes.
    """
    dataset = []  # Store game states and moves

    for game_id in game_ids:  # Fixed: Remove `enumerate()`
        # player2_skill = random.randint(150, 5000)  # Randomize Player 2's skill
        # fixed_skill = random.randint(150, 5000)  # Randomize Player 1's skill
        player2_skill = random.randint(1000, 3000)
        fixed_skill = random.randint(1000, 3000)

        board = np.zeros((6, 7, 2), dtype=int)  # Initialize empty board
        game_over = False
        player = random.choice(["plus", "minus"])

        randomness = random.random() > 0.5

        # # Determine the randomness mode
        # rand_mode = random.choices(
        #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #     weights=[0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        # )[0]

        move_count = 0

        while not game_over:
            legal_moves = find_legal(board)
            if not legal_moves:
                print(
                    f"Game {game_id}: Tie! (P1 Skill: {fixed_skill}, P2 Skill: {player2_skill})"
                )
                progress_queue.put(1)  # Send update to progress bar
                break

            # # Apply randomness based on selected mode
            # if move_count < rand_mode:
            #     move = random.choice(legal_moves)  # Random move
            # else:
            #     nsteps = fixed_skill if player == "plus" else player2_skill
            #     move = mcts(board, player, nsteps)  # MCTS-based move

            if player == "plus":
                nsteps = fixed_skill
                move = mcts(board, player, nsteps)
            else:
                if (
                    randomness
                    and random.random() < 0.7 ** (move_count + 1)
                    and move_count < 10
                ):
                    move = random.choice(legal_moves)  # Random move
                else:
                    nsteps = player2_skill
                    move = mcts(board, player, nsteps)  # MCTS-based move

                move_count += 1  # Increment move counter

            dataset.append(
                (game_id, board.copy(), move, fixed_skill, player2_skill, player)
            )
            board = update_board(board, player, move)

            result = check_for_win(board, move)
            if result != "nobody":
                print(
                    f"Game {game_id}: Winner = {result} (P1 Skill: {fixed_skill}, P2 Skill: {player2_skill})"
                )
                progress_queue.put(1)  # Send update to progress bar
                game_over = True

            player = "minus" if player == "plus" else "plus"

    return dataset


############## FOR PROGRESSIVE SKILL LEVELING ###########
# def generate_partial_dataset(game_ids, player2_skill_levels): # Needed for parallel processing
#     """Generate a subset of the dataset using precomputed skill levels."""
#     try:
#         return generate_dataset_progressive_opponent_parallel(
#             game_ids, player2_skill_levels
#         )
#     except Exception as e:
#         print(f"Worker failed with error: {e}")
#         return []


################## FOR RANDOM SKILL LEVELING + RANDOM INITIAL MOVES ##################
def generate_partial_dataset(
    game_ids, progress_queue
):  # Needed for parallel processing
    """Generate a subset of the dataset with random opponents and randomized first moves."""
    try:
        return generate_dataset_randomopponents_randomfirstmoves_parallel(
            game_ids, progress_queue
        )
    except Exception as e:
        print(f"Worker failed with error: {e}")
        return []


############## FOR PROGRESSIVE SKILL LEVELING ###########
# def precompute_player2_skills(
#     num_games, initial_skill, phase1_end, phase1_inc, phase2_inc
# ):
#     """Precompute Player 2's skill progression based on the original logic."""
#     player2_skills = []
#     player2_skill = initial_skill

#     phase1_games = (phase1_end - initial_skill) // phase1_inc * 100
#     phase2_games = num_games - phase1_games

#     for game in range(num_games):
#         if game < phase1_games:
#             if game % 100 == 0 and player2_skill < phase1_end:
#                 player2_skill += phase1_inc
#         else:
#             if game % 100 == 0 and player2_skill < 2000:
#                 player2_skill += phase2_inc
#         player2_skills.append(player2_skill)

#     return player2_skills


def progress_updater(progress_queue, num_games):
    """Continuously updates tqdm progress bar based on worker updates."""
    with tqdm(total=num_games, desc="Games Completed", unit="game") as pbar:
        completed_games = 0
        while completed_games < num_games:
            progress_queue.get()  # Wait for an update
            completed_games += 1
            pbar.update(1)  # Update tqdm bar


# ------------------------------------------------GENERATE DATASET-------------------------------------#

############## FOR PROGRESSIVE SKILL LEVELING ###########
# def main():
#     num_games = 5400
#     num_workers = 4
#     initial_opponent_skill = 300
#     phase1_end_skill = 1000
#     phase1_increment = 50
#     phase2_increment = 25

#     # Precompute Player 2's skill progression
#     player2_skills = precompute_player2_skills(
#         num_games,
#         initial_opponent_skill,
#         phase1_end_skill,
#         phase1_increment,
#         phase2_increment,
#     )

#     # Split data for workers
#     games_per_worker = num_games // num_workers
#     game_splits = [
#         list(range(i * games_per_worker, (i + 1) * games_per_worker))
#         for i in range(num_workers)
#     ]
#     skill_splits = [
#         player2_skills[i * games_per_worker : (i + 1) * games_per_worker]
#         for i in range(num_workers)
#     ]

#     start_time = time.perf_counter()

#     # Use multiprocessing Manager to create a shared progress queue
#     manager = multiprocessing.Manager()
#     progress_queue = manager.Queue()

#     # Start the progress updater thread
#     progress_process = multiprocessing.Process(
#         target=progress_updater, args=(progress_queue, num_games)
#     )
#     progress_process.start()

#     # Use multiprocessing
#     ctx = multiprocessing.get_context("spawn")
#     executor = ctx.Pool(processes=num_workers)

#     futures = [
#         executor.apply_async(
#             generate_dataset_progressive_opponent_parallel,
#             args=(game_splits[i], skill_splits[i], progress_queue),
#         )
#         for i in range(num_workers)
#     ]

#     all_datasets = []
#     for future in futures:
#         result = future.get()
#         if result:
#             all_datasets.extend(result)

#     # Close the executor
#     executor.close()
#     executor.join()

#     # Tell the progress updater to stop once all games are processed
#     progress_process.join()

#     total_time = time.perf_counter() - start_time
#     print(f"Dataset generated in {total_time:.2f} seconds.")

#     # Save dataset
#     with open("Connect4Dataset_Progressive_and_Random_Skill.pkl", "wb") as file:
#         pickle.dump(all_datasets, file)

#     print(f"Dataset generated with {len(all_datasets)} entries.")


################## FOR RANDOM SKILL LEVELING + RANDOM INITIAL MOVES ##################
def main():
    num_games = 10
    num_workers = 10

    # Split game IDs for workers
    games_per_worker = num_games // num_workers
    game_splits = [
        list(range(i * games_per_worker, (i + 1) * games_per_worker))
        for i in range(num_workers)
    ]

    start_time = time.perf_counter()

    # Use multiprocessing Manager to create a shared progress queue
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    # Start the progress updater process
    progress_process = multiprocessing.Process(
        target=progress_updater, args=(progress_queue, num_games)
    )
    progress_process.start()

    # Use multiprocessing
    ctx = multiprocessing.get_context("spawn")
    executor = ctx.Pool(processes=num_workers)

    # Update dataset generation function
    futures = [
        executor.apply_async(
            generate_dataset_randomopponents_randomfirstmoves_parallel,
            args=(game_splits[i], progress_queue),
        )
        for i in range(num_workers)
    ]

    all_datasets = []
    for future in futures:
        result = future.get()
        if result:
            all_datasets.extend(result)

    # Close the executor
    executor.close()
    executor.join()

    # Tell the progress updater to stop once all games are processed
    progress_process.join()

    total_time = time.perf_counter() - start_time
    print(f"Dataset generated in {total_time:.2f} seconds.")

    # Save dataset
    with open("Connect4Dataset_SmartRandom.pkl", "wb") as file:
        pickle.dump(all_datasets, file)

    print(f"Dataset generated with {len(all_datasets)} entries.")


if __name__ == "__main__":
    main()
