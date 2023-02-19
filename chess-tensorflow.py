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
with open("chess-games.pgn", "rb") as f:
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

# Train the model using the labeled dataset of chess board positions and moves
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100)

# Save the model weights and optimizer state to a filee
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