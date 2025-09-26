"""
Move the simulated unclean data to the raw CSV file in the data folder
and provide a function to load it.
"""

import os
from simulate_data import PitchSimulator

ps = PitchSimulator()
df, stats = ps.simulate_gameflow(num_events=2000)

raw_csv_path = "data/raw/simulated_game_data.csv"
df.to_csv(raw_csv_path, index=False)
print(f"Simulatef Data saved to {raw_csv_path}")