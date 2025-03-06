# AI-Powered Connect 4 Bot

## Overview
This project is an AI-driven Connect 4 bot that utilizes **deep learning** and **Monte Carlo Tree Search (MCTS)** to play the game strategically. The bot is trained using **convolutional neural networks (CNNs)** and **transformers** to predict the best moves. The model is deployed on **AWS** using **Docker**, and an interactive web application built with **Anvil** allows users to challenge the AI.

## Features
- **AI Strategy Training:** Uses Monte Carlo Tree Search (MCTS) to generate optimal moves for training.
- **Deep Learning Models:** Implements **CNNs** and **transformers** to classify board positions and determine the best moves.
- **Cloud Deployment:** AI models are hosted on **AWS** with a Dockerized backend for scalability.
- **Web Interface:** Built with **Anvil**, allowing users to play against the AI in real-time.
- **Optimized Performance:** Fine-tuned hyperparameters to improve model accuracy and decision-making.

## Tech Stack
- **Python** (TensorFlow, PyTorch, NumPy, Pandas)
- **Monte Carlo Tree Search (MCTS)** for data generation
- **AWS (S3)** for cloud hosting
- **Docker** for model deployment
- **Anvil** for front-end web development


## How It Works
1. **Data Collection**: MCTS self-plays thousands of games to generate a dataset of board positions and optimal moves.
2. **Model Training**: A CNN and a Transformer model are trained using supervised learning on the generated dataset.
3. **Web Interface**: The trained models are deployed on AWS and made accessible via a web app using Anvil.
4. **Gameplay**: Users interact with the bot, selecting either the CNN or Transformer-based AI to compete against.

## Future Enhancements
- Implement reinforcement learning for improved AI strategy... TBC
- Enhance UI/UX of the web application.
- Optimize model performance with further hyperparameter tuning.

## Contributors
- **Hrishi Salitri** – **Carson Mullen** - **Eshaan Arora** - **Sam Song**



