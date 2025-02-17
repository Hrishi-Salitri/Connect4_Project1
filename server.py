import tensorflow as tf
import numpy as np
import anvil.server
import os

# ✅ **Anvil Uplink Key (Replace with Your Own)**
ANVIL_KEY = "server_FQYMV77OZUB7T4LCWBYFM3CA-H7QYHHJFDSKCNP6B"
anvil.server.connect(ANVIL_KEY)

# ✅ **Lazy Load CNN Model**
MODEL_PATH = "/app/best_model_combined.h5"
model = None  # Model will load only when needed

def load_model():
    """ Load the CNN model when first needed. """
    global model
    if model is None:
        print(f"📌 Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")

# ✅ **Anvil Callable Function for AI Move Prediction**
@anvil.server.callable
def get_best_move(board):
    """Receives a board state, runs AI model, and returns the best move."""
    try:
        load_model()  # Ensure the model is loaded

        # ✅ Convert to numpy array
        board = np.array(board)

        # ✅ Validate board shape
        if board.shape != (6, 7):
            print(f"❌ Invalid board shape received: {board.shape}")
            return {"error": f"Invalid board shape: {board.shape}, expected (6,7)"}

        # ✅ Convert board to (6,7,2) format (split into Player 1 and Player 2)
        player1_board = (board == 1).astype(np.float32)  # Player 1 pieces
        player2_board = (board == 2).astype(np.float32)  # Player 2 pieces
        board = np.stack([player1_board, player2_board], axis=-1)  # Shape (6,7,2)

        # ✅ Reshape for model input
        board = board.reshape(1, 6, 7, 2)
        print(f"📊 Board shape after processing: {board.shape}")  

        # ✅ Run AI model prediction
        prediction = model.predict(board)
        best_move = int(np.argmax(prediction))

        # ✅ Validate AI move (must be between 0-6)
        if not (0 <= best_move <= 6):
            print(f"❌ Invalid AI move generated: {best_move}")
            return {"error": f"Invalid AI move: {best_move}"}

        print(f"🤖 AI chose move: {best_move}")
        return {"best_move": best_move}

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return {"error": str(e)}

# ✅ **Keep Anvil Uplink Running**
anvil.server.wait_forever()
