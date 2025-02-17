import tensorflow as tf
import numpy as np
import anvil.server
import os
from Customer_trans_layers import PositionalIndex, ClassTokenIndex  # Import custom layers

# ✅ Ensure layers are registered
from keras.saving import register_keras_serializable

@register_keras_serializable()
class PositionalIndex(tf.keras.layers.Layer):
    def call(self, x):
        bs = tf.shape(x)[0]
        number_of_vectors = tf.shape(x)[1]
        indices = tf.range(number_of_vectors)
        indices = tf.expand_dims(indices, 0)
        return tf.tile(indices, [bs, 1])

    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class ClassTokenIndex(tf.keras.layers.Layer):
    def call(self, x):
        bs = tf.shape(x)[0]
        number_of_vectors = 1
        indices = tf.range(number_of_vectors)
        indices = tf.expand_dims(indices, 0)
        return tf.tile(indices, [bs, 1])

    def get_config(self):
        return super().get_config()

# ✅ **Anvil Uplink Key**
ANVIL_KEY = "server_FQYMV77OZUB7T4LCWBYFM3CA-H7QYHHJFDSKCNP6B"
anvil.server.connect(ANVIL_KEY)

# ✅ **Lazy Load Transformer Model**
MODEL_PATH = "/app/connect4_transformer_4x4.keras"
model = None

def load_model():
    """ Load the Transformer model, ensuring custom layers are recognized. """
    global model
    if model is None:
        print(f"📌 Attempting to load Transformer model from: {MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH, 
                custom_objects={"PositionalIndex": PositionalIndex, "ClassTokenIndex": ClassTokenIndex}
            )
            print("✅ Transformer Model loaded successfully.")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")

def convert_board_for_transformer(board):
    """
    Convert a (6,7) Connect 4 board into the expected (12,32) shape for the Transformer model.
    """
    board = np.array(board)

    # ✅ Validate board shape
    if board.shape != (6, 7):  
        print(f"❌ Invalid board shape received: {board.shape}")
        return {"error": f"Invalid board shape: {board.shape}, expected (6,7)"}

    # Flatten the board and pad it to (12,32)
    padded_board = np.zeros((12, 32), dtype=np.float32)
    padded_board[:6, :7] = board  # Place the board in the first 6 rows, 7 columns

    transformed_board = padded_board.reshape(1, 12, 32)  # Ensure batch dimension (1, 12, 32)

    print(f"🚀 Transformed Board Shape: {transformed_board.shape}")
    return transformed_board

@anvil.server.callable
def get_best_move_transformer(board):
    """Receives a board state, reshapes it, runs Transformer model, and returns the best move."""
    try:
        load_model()  # Ensure model is loaded
        if model is None:
            return {"error": "Model could not be loaded"}

        transformed_board = convert_board_for_transformer(board)

        if isinstance(transformed_board, dict):  # If an error dict was returned
            return transformed_board

        prediction = model.predict(transformed_board)
        print(f"🚀 Model Predictions: {prediction}")

        sorted_moves = np.argsort(prediction[0])[::-1]  # Get moves sorted by highest probability

        # ✅ Step 1: Check for a valid move
        for move in sorted_moves:
            if board[0][move] == 0:  # Check if top row of column is empty
                print(f"🤖 Transformer AI chose valid move: {move}")
                return {"best_move": int(move)}

        # 🚨 If all predicted columns are full, fallback to a random move
        print("⚠️ All predicted columns are full! Choosing random valid column.")
        valid_moves = [col for col in range(7) if board[0][col] == 0]
        if valid_moves:
            fallback_move = np.random.choice(valid_moves)
            print(f"🤖 Fallback move: {fallback_move}")
            return {"best_move": int(fallback_move)}

        return {"error": "No valid moves left!"}

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return {"error": str(e)}

anvil.server.wait_forever()
