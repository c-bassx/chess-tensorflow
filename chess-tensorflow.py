import tensorflow as tf
import pandas as pd
import numpy as np
import chess.pgn
from io import StringIO

# Define a function to convert FEN strings to 8x8 numerical matrices
def fen_to_matrix(fen):
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
with open("Ivanchuk.pgn", "rb") as f:
    games = f.read().decode('iso-8859-1')

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

# Build a TensorFlow model that takes in the preprocessed chess board and outputs the best move
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Train the model using the labeled dataset of chess board positions and moves
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=1)

# Use the trained model to predict the best move given a FEN position
while True:
    fen = input("Enter a FEN position (or 'end' to quit): ")
    if fen == "end":
        break
    
    board = chess.Board(fen)
    side = "white" if board.turn == chess.WHITE else "black"
    
    X_new = np.array([fen_to_matrix(board.fen())])
    y_pred = model.predict(X_new)
    best_move_idx = np.argmax(y_pred)
    best_move_uci = class_labels[best_move_idx]  # convert index back to uci string
    best_move = chess.Move.from_uci(best_move_uci)

    print("Best move for", side, "side:", best_move.uci())

'''
import tensorflow as tf
import pandas as pd
import numpy as np
import chess.pgn
from io import StringIO

# Define a function to convert FEN strings to 8x8 numerical matrices
def fen_to_matrix(fen, side):
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
    if side == "black":
        board = np.rot90(board, 2)
    return board.flatten()

# Load the chess games database
with open("Kasparov.pgn", "rb") as f:
    games = f.read().decode('iso-8859-1')

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
        if board.is_legal(move):
            positions.append(board.fen())
            moves.append(str(move))
        board.push(move)

if not positions or not moves:
    print("No valid positions or moves found!")
    exit()

# Convert the positions and moves into numerical matrices
X = np.array([fen_to_matrix(pos, "white") for pos in positions] + [fen_to_matrix(pos, "black") for pos in positions])
y = list(map(lambda m: chess.Move.from_uci(m).uci(), moves))  # convert moves to list of uci strings
class_labels = sorted(set(y))
num_classes = len(class_labels)
label_map = {label: i for i, label in enumerate(class_labels)}
y = np.array([label_map[label] for label in y])
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Build a TensorFlow model that takes in the preprocessed chess board and outputs the best move
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Train the model using the labeled dataset of chess board positions and moves
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10)

# Use the trained model to predict the best move given a FEN position
while True:
    fen = input("Enter a FEN position (or 'end' to quit): ")
    if fen == "end":
        break

    board = chess.Board(fen)
    side = "white" if board.turn == chess.WHITE else "black"

    while True:
        # Get the legal moves for the current position
        legal_moves = [move.uci() for move in board.legal_moves]

        X_new = np.array([fen_to_matrix(board.fen(), side)])  # Fix here: add `side` argument
        y_pred = model.predict(X_new)
        best_move_idx = np.argmax(y_pred)
        best_move_uci = class_labels[best_move_idx]  # convert index back to uci string

        # Check if the predicted move is legal
        if best_move_uci in legal_moves:
            break

    # Check if it's the correct side's turn to move
    if (side == "white" and best_move_uci[0].isupper()) or (side == "black" and best_move_uci[0].islower()):
        print(f"Predicted move: {best_move_uci}")
        board.push_uci(best_move_uci)
        print(f"New FEN position: {board.fen()}")
    else:
        print("Incorrect side predicted!")
        '''