"""
simulate.py
Simulates football-like event data as a stochastic chaotic system.
General enough to adapt to other domains later for prediction of chaotic systems.
"""

import numpy as np
import random
import pandas as pd

# Columns for simulated events
columns = ["time", "team", "event", "start_x", "start_y", "end_x", "end_y", "possession_duration"]

# Base event probabilities (goal is conditional on shot)
base_event_probabilities = {
    "pass": 0.75,
    "shot": 0.10,
    "turnover": 0.15
}


def build_event(time, team, event, start, end, possession_duration=None):
    """Helper to build a consistent event dictionary."""
    return {
        "time": time,
        "team": team,
        "event": event,
        "start_x": start[0],
        "start_y": start[1],
        "end_x": end[0],
        "end_y": end[1],
        "possession_duration": possession_duration if possession_duration is not None else random.randint(1, 10)
    }


class PitchSimulator:
    def __init__(self, length=100, width=50):
        # centered at origin
        self.length = length
        self.width = width
        self.x_min = -length / 2
        self.x_max = length / 2
        self.y_min = -width / 2
        self.y_max = width / 2

    def clamp_position(self, x, y):
        """Keep x,y inside pitch bounds."""
        x = max(self.x_min, min(self.x_max, x))
        y = max(self.y_min, min(self.y_max, y))
        return x, y

    def generate_position(self, team, event_type, start_pos=None):
        """Return an (x,y) plausible for the given team/event. Center origin logic."""
        if start_pos is None:
            start_x, start_y = 0.0, 0.0
        else:
            start_x, start_y = start_pos

        if event_type == "shot":
            # Aim at opponent's goal
            if team == "team0":
                x = self.x_max
                y = start_y + np.random.uniform(-5, 5)
            else:
                x = self.x_min
                y = start_y + np.random.uniform(-5, 5)

        elif event_type == "pass":
            # Bias forward passes
            if team == "team0":
                x = start_x + np.random.uniform(-5, 15)
            else:
                x = start_x + np.random.uniform(-15, 5)
            y = start_y + np.random.uniform(-8, 8)

        elif event_type == "turnover":
            x, y = start_x, start_y

        else:  # fallback random
            if team == "team0":
                x = np.random.uniform(0, self.x_max)
            else:
                x = np.random.uniform(self.x_min, 0)
            y = np.random.uniform(self.y_min, self.y_max)

        return self.clamp_position(x, y)

    def adjust_event_probabilities(self, last_event, position, team):
        """
        Adjust probabilities dynamically based on last event and field position.
        - Momentum: passes beget passes, turnovers increase pass likelihood
        - Field position: closer to opponent's goal increases shot chance
        """
        probs = base_event_probabilities.copy()

        # Momentum bias
        if last_event == "pass":
            probs["pass"] += 0.05
        elif last_event == "turnover":
            probs["pass"] += 0.10
            probs["turnover"] -= 0.05
        elif last_event == "shot":
            probs["pass"] += 0.05

        # Position-based bias
        x, _ = position
        if team == "team0":
            dist_to_goal = self.x_max - x
        else:
            dist_to_goal = x - self.x_min

        if dist_to_goal < self.length / 4:  # final 25% of pitch
            probs["shot"] += 0.10
            probs["pass"] -= 0.05
        elif dist_to_goal > self.length / 2:  # deep in own half
            probs["turnover"] += 0.05

        # Normalize to ensure probabilities sum to 1
        total = sum(probs.values())
        for k in probs:
            probs[k] = max(0, probs[k]) / total
        return probs

    def simulate_gameflow(self, num_events=100, goal_base_prob=0.2):
        """
        Simulate a chaotic football gameflow with dynamic probabilities.
        Returns a pandas DataFrame of logged events.
        """
        match_log = []
        current_team = "team0"
        current_position = (0.0, 0.0)
        time = 0

        # kickoff event
        kickoff_end = (-10.0, 0.0) if current_team == "team0" else (10.0, 0.0)
        kickoff_event = build_event(time, current_team, "kickoff_pass", (0.0, 0.0), kickoff_end, possession_duration=1)
        match_log.append(kickoff_event)
        current_position = kickoff_end
        last_event = "kickoff_pass"
        time += 1

        for i in range(num_events):
            # Adjusted probabilities
            probs = self.adjust_event_probabilities(last_event, current_position, current_team)
            event = np.random.choice(list(probs.keys()), p=list(probs.values()))

            start_x, start_y = current_position
            new_x, new_y = self.generate_position(current_team, event, start_pos=current_position)

            if event == "shot":
                # Distance to goal affects chance of scoring
                if current_team == "team0":
                    dist = abs(self.x_max - start_x)
                else:
                    dist = abs(self.x_min - start_x)
                adjusted_prob = goal_base_prob * (1 / (1 + dist / 20.0))

                if np.random.rand() < adjusted_prob:
                    step = build_event(time, current_team, "goal", current_position, (new_x, new_y))
                    match_log.append(step)
                    # Reset play
                    current_team = "team1" if current_team == "team0" else "team0"
                    current_position = (0.0, 0.0)
                    time += 1
                    kickoff_end = (-10.0, 0.0) if current_team == "team0" else (10.0, 0.0)
                    kickoff_event = build_event(time, current_team, "kickoff_pass", (0.0, 0.0), kickoff_end, possession_duration=1)
                    match_log.append(kickoff_event)
                    current_position = kickoff_end
                    last_event = "kickoff_pass"
                    time += 1
                    continue
                else:
                    step_event = "shot"
            elif event == "turnover":
                step_event = "turnover"
            else:
                step_event = "pass"

            # Append event
            step = build_event(time, current_team, step_event, current_position, (new_x, new_y))
            match_log.append(step)

            # Update state
            if step_event == "turnover":
                current_team = "team1" if current_team == "team0" else "team0"
                current_position = (new_x, new_y)
            else:
                current_position = (new_x, new_y)

            last_event = step_event
            time += 1

        return pd.DataFrame(match_log, columns=columns)


if __name__ == "__main__":
    ps = PitchSimulator()
    df_flow = ps.simulate_gameflow(num_events=200)
    print(df_flow.head(20))





    


