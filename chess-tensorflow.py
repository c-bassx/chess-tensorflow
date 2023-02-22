import tensorflow as tf
import numpy as np
import chess.pgn
import chess.engine # Adds stockfish to help train the model
from io import StringIO

print("Program initialized.")

def fen_to_board(fen):
    """
    Convert a FEN string to a 2D list representing a chess board.

    Parameters:
        fen (str): A FEN string representing the current state of the board.

    Returns:
        A 2D list representing the board.
    """
    if isinstance(fen, list):
        fen = fen[0]
    rows = fen.split("/")
    board = []
    for row in rows:
        board_row = []
        for square in row:
            if square.isdigit():
                board_row.extend([None] * int(square))
            else:
                board_row.append(square)
        board.append(board_row)
    return board

# Define a function to convert FEN strings to 8x8 numerical matrices
def fen_to_matrix(fen):
    """
    Converts a FEN string into an 8x8 numerical matrix that can be fed into the model

    Parameters:
    A FEN string representing the current state of the board.

    Returns:
    8x8 Numerical Matrix
    """
    board = np.zeros((8, 8), dtype=np.int8)
    rows = fen.split()[0].split('/')
    for i, row in enumerate(rows):
        j = 0
        for char in row:
            if char.isnumeric():
                j += int(char)
            else:
                if char.islower():
                    piece = ord(char) - ord('a') + 1
                else:
                    piece = ord(char) - ord('A') + 1
                board[i, j] = piece
                j += 1
    return board.flatten()

# Load the chess games database
with open("SicilianNajdorf6g3.pgn", "rb") as f:
    games = f.read().decode('iso-8859-1')
print("Database loaded.")

# Collect a dataset of labeled chess board positions and their corresponding moves
positions = []
moves = []
pgn = StringIO(games)
while True:
    game = chess.pgn.read_game(pgn)
    if not game:
        break
    board = game.board()
    for move in game.mainline_moves():
        positions.append(board.fen())
        moves.append(str(move))
        board.push(move)

if not positions or not moves:
    print("No valid positions or moves found!")
    exit()

# Convert the positions and moves into numerical matrices
X = np.array([fen_to_matrix(pos) for pos in positions])
y = list(map(lambda m: chess.Move.from_uci(m).uci(), moves))  # convert moves to list of uci strings
class_labels = sorted(set(y))
num_classes = len(class_labels)
label_map = {label: i for i, label in enumerate(class_labels)}
y = np.array([label_map[label] for label in y])
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Define a custom loss function that penalizes illegal and dangerous moves
def custom_loss(y_true, y_pred):
    illegal_moves = tf.math.reduce_any(tf.math.equal(y_true, 0), axis=1)  # find illegal moves
    dangerous_moves = tf.math.reduce_any(tf.math.equal(y_true, 1), axis=1)  # find dangerous moves
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)  # cross-entropy loss
    illegal_penalty = tf.where(illegal_moves, 10.0, 0.0)  # penalty for illegal moves
    dangerous_penalty = tf.where(dangerous_moves, 1.0, 0.0)  # penalty for dangerous moves
    return loss + illegal_penalty + dangerous_penalty

# Build a TensorFlow model that takes in the preprocessed chess board and outputs the best move
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
print("Tensorflow model built.")

# Define early stopping callback to monitor validation loss and stop training if no improvement is seen for 1 epoch
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # monitor validation loss
    patience=1,  # stop training if no improvement is seen for 1 epoch
    restore_best_weights=True  # restore the weights for the epoch with the best validation loss
)

# Create an instance of the Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")

# Define a function to evaluate the model's moves 
def evaluate_move(move, board):
    # Make the move on a copy of the board
    test_board = board.copy()
    test_board.push(move)

    # Calculate the value of the move based on captures, attacks, threats, checks, defending, pins, and forks
    value = 0
    if test_board.is_capture(move):
        value += 1
    elif board.is_capture(move):
        value -= 1
    if test_board.is_check():
        value += 2
    if test_board.gives_check(move):
        value += 1
    if board.gives_check(move):
        value -= 1
    if test_board.is_attacked_by(board.turn, move.to_square):
        value -= 1
    if board.is_attacked_by(test_board.turn, move.to_square):
        value += 1
    if test_board.is_pinned(test_board.turn, move.to_square):
        value += 1
    if test_board.is_pinned(board.turn, move.to_square):
        value -= 1
    if test_board.is_en_passant(move):
        value += 1
    if board.is_en_passant(move):
        value -= 1

    # Return the value of the move
    return value
    '''
    # Evaluate the move based on piece values
    piece_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    captured_piece = board.piece_at(move.to_square)
    material_gain = 0
    if captured_piece is not None:
        material_gain = piece_value[captured_piece.piece_type]

    # Return the material gain from the move
    return material_gain '''

def select_move(board):
    # Get a list of all legal moves on the board
    legal_moves = list(board.legal_moves)

    # Calculate the value of each move
    move_values = [evaluate_move(move, board) for move in legal_moves]

    # Select the move with the highest value
    best_move = legal_moves[np.argmax(move_values)]

    return best_move

def predict_move(board, top_k=3):
    """
    Use the model to predict the best move for a given board.

    Parameters:
        board (chess.Board): A chess board.
        top_k (int): The number of top moves to consider.

    Returns:
        The best move predicted by the model.
    """
    # Generate all legal moves
    legal_moves = list(board.legal_moves)

    # Create a matrix of all possible boards resulting from each legal move
    X_test = np.zeros((len(legal_moves), 64))
    for i, move in enumerate(legal_moves):
        test_board = board.copy()
        test_board.push(move)
        X_test[i] = fen_to_matrix(test_board.fen())

    # Use the model to predict the value of each possible move
    y_pred = model.predict(X_test)

    # Evaluate the top-k legal moves based on captures, attacks, threats, checks, defending, pins, and forks
    move_values = []
    for i, move in enumerate(legal_moves):
        value = evaluate_move(move, board)
        move_values.append(value + y_pred[i])

    # Choose the move with the highest evaluation
    best_move_idx = np.argmax(move_values)
    return legal_moves[best_move_idx]

# Define a function to play a game between the model and itself
def self_play(model, X, y, label_map, engine, batch_size=128, epochs=10, verbose=1):
    """
    Train the model by playing against itself.

    Parameters:
        model (tf.keras.Model): The TensorFlow model to train.
        X (ndarray): An array of preprocessed chess board positions.
        y (ndarray): An array of the corresponding moves in UCI notation.
        label_map (dict): A dictionary mapping moves in UCI notation to integers.
        engine (chess.engine.Engine): The Stockfish engine to use for move evaluation.
        batch_size (int): The number of training examples to use in each batch.
        epochs (int): The number of epochs to train for.
        verbose (int): The level of verbosity during training.

    Returns:
        The trained TensorFlow model.
    """

    # Set up the training parameters
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=custom_loss, optimizer=optimizer)

    # Iterate over the specified number of epochs
    for epoch in range(epochs):
        print("Epoch", epoch+1)

        # Shuffle the training data
        permutation = np.random.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]

        # Set up the training batches
        num_batches = int(np.ceil(X.shape[0] / batch_size))
        X_batches = np.array_split(X, num_batches)
        y_batches = np.array_split(y, num_batches)

        # Iterate over the batches
        for i in range(num_batches):
            X_batch = X_batches[i]
            y_batch = y_batches[i]

            # Create a copy of the model for the current side
            side_model = tf.keras.models.clone_model(model)
            side_model.set_weights(model.get_weights())

            # Randomly perturb the model's parameters for this side
            perturb_amount = np.random.normal(0, 0.1, size=side_model.count_params())
            side_model.set_weights([w + p for w, p in zip(side_model.get_weights(), perturb_amount)])

            # Iterate over the training examples in the current batch
            for j in range(X_batch.shape[0]):
                # Generate the move prediction for the current position using the model for this side
                X_board = X_batch[j].reshape(1, -1)
                y_true = y_batch[j].reshape(1, -1)
                y_pred = side_model.predict(X_board)

                # Choose the best legal move using the Stockfish engine
                board = chess.Board(fen_to_board(str(X_batch[j])))
                best_move = None
                best_score = float("-inf")
                for move in board.legal_moves:
                    score = evaluate_move(move, board)
                    if score > best_score:
                        best_move = move
                        best_score = score

                # Update the model's parameters based on the outcome of the game
                if board.turn == chess.WHITE:
                    if best_score == float("inf"):
                        side_model.fit(X_board, y_true, batch_size=1, epochs=1, verbose=0)
                    elif best_score == float("-inf"):
                        side_model.fit(X_board, y_pred, batch_size=1, epochs=1, verbose=0)
                    else:
                        y_best = label_map[best_move.uci()]
                        y_best = tf.keras.utils.to_categorical(y_best, num_classes=len(label_map))
                        side_model.fit(X_board, y_best, batch_size=1, epochs=1, verbose=0)

                else:
                    if best_score == float("inf"):
                        side_model.fit(X_board, y_pred, batch_size=1, epochs=1, verbose=0)
                    elif best_score == float("-inf"):
                        side_model.fit(X_board, y_true, batch_size=1, epochs=1, verbose=0)
                    else:
                        y_best = label_map[best_move.uci()]
                        y_best = tf.keras.utils.to_categorical(y_best, num_classes=len(label_map))
                        side_model.fit(X_board, y_best, batch_size=1, epochs=1, verbose=0)

    return model
    

# Train the model using the labeled dataset of chess board positions and moves
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, validation_split=0.2, callbacks=[early_stopping])

# Play games of chess against itself
# self_play(model, X, y, label_map, engine, batch_size=128, epochs=10, verbose=1)

# Save the model weights and optimizer state to a file
model.save_weights("chess_model_weights.h5")

# Load the saved model weights and optimizer state
model.load_weights("chess_model_weights.h5")

# Use the trained model to predict the best move given a FEN position
while True:
    fen = input("Enter a FEN position (or 'end' to quit): ")
    if fen == "end":
        break
    
    board = chess.Board(fen)
    side = "white" if board.turn == chess.WHITE else "black"
    
    X_new = np.array([fen_to_matrix(board.fen())])
    y_pred = model.predict(X_new)
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    y_pred_filtered = np.zeros_like(y_pred)
    for i, label in enumerate(class_labels):
        if label in legal_moves_uci:
            y_pred_filtered[:, i] = y_pred[:, i]
    best_move_idx = np.argmax(y_pred_filtered)
    best_move_uci = class_labels[best_move_idx]  # convert index back to uci string
    best_move = chess.Move.from_uci(best_move_uci)
    if best_move_uci not in legal_moves_uci:
        print("Illegal move predicted!")
        model.train_on_batch(X_new, y_pred_filtered)  # penalize the model for making an illegal move
        continue
    
    board.push(best_move)
    print("Best move for", side, "side:", best_move.uci())
    if board.is_checkmate():
        print(side, "is checkmated!")
        model.train_on_batch(X_new, y_pred_filtered)  # penalize the model for being checkmated
    elif board.is_check():
        print(side, "is in check!")
        model.train_on_batch(X_new, y_pred_filtered)  # penalize the model for being checked
    elif board.is_stalemate():
        print("Stalemate!")
        model.train_on_batch(X_new, y_pred_filtered)  # penalize the model for causing a stalemate
