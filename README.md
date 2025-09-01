# DeepRL_Gothic

### How would you teach an AI to play an RPG, when the goal isn't just to win, but to *experience* a story?

This is an R&D project aimed at tackling the problem of modeling complex, goal-oriented behavior for an AI agent in a non-linear, open-world environment.

## The Core Problem

In traditional games like Chess or Go, the objective is clear: win the game. This makes defining a success metric for an AI straightforward.

Open-world RPGs, like Gothic, present a far more complex challenge. What does it mean to "play well"?
* Is it completing quests efficiently?
* Is it exploring the world?
* Is it making "good" narrative choices?
* Is it simply surviving?

This project explores these questions by attempting to define a viable reward function for a Reinforcement Learning (RL) agent in a world where the goals are ambiguous and multi-faceted.

## Project Goals

1.  **Define a Reward Function:** The primary conceptual challenge is to translate the abstract idea of "progress" in an RPG into a mathematical reward function that can effectively guide an RL agent.
2.  **Live Data Extraction:** To provide the agent with "senses" within the game world, a key technical goal is to extract real-time world-state data (e.g., player coordinates, direction vector, world events).
3.  **Agent Training:** To train an agent capable of performing non-trivial, multi-step tasks that go beyond simple navigation, such as completing an early-game quest.

## Technical Approach

This project is a blend of a conceptual AI challenge and a practical reverse-engineering task.

* **AI & Reinforcement Learning:** The core of the project is written in Python, utilizing modern Deep Reinforcement Learning frameworks to model the agent's decision-making process.
* **Game Engine Hooking:** To feed the model with live data, the project leverages the **Union Framework** to hook into the legacy game engine of Gothic. This allows for the extraction of memory addresses corresponding to the player's state and the surrounding environment.
* **Data Pipeline:** The extracted data serves as the state representation for the RL agent, allowing it to perceive the world and make decisions.

## Current Status

This project is currently in the research and development phase. The primary focus is on establishing a stable data pipeline between the game engine and the Python-based RL agent.
