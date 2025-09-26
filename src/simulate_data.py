"""
simulate_data.py
Simulates football-like event data as a stochastic chaotic system.
Optimized randomness with priority factors (zone, distance, possession streak, pass distance).
"""

import numpy as np
import random
import pandas as pd

# Extended columns for richer factors
columns = [
    "time", "team", "event", "start_x", "start_y", "end_x", "end_y",
    "possession_duration", "zone", "dist_to_goal", "possession_streak", "pass_distance"
]

# Base event probabilities (goal is conditional on shot)
base_event_probabilities = {
    "pass": 0.70,
    "shot": 0.10,
    "turnover": 0.20
}


def build_event(time, team, event, start, end,
                possession_duration=None, zone=None,
                dist_to_goal=None, possession_streak=None, pass_distance=None):
    """Helper to build a consistent event dictionary."""
    return {
        "time": time,
        "team": team,
        "event": event,
        "start_x": start[0],
        "start_y": start[1],
        "end_x": end[0],
        "end_y": end[1],
        "possession_duration": possession_duration if possession_duration is not None else random.randint(1, 10),
        "zone": zone,
        "dist_to_goal": dist_to_goal,
        "possession_streak": possession_streak,
        "pass_distance": pass_distance
    }


class PitchSimulator:
    def __init__(self, length=100, width=50):
        self.length = length
        self.width = width
        self.x_min = -length / 2
        self.x_max = length / 2
        self.y_min = -width / 2
        self.y_max = width / 2

        self.stats = {
            "team0_goals": 0,
            "team1_goals": 0,
            "team0_shots": 0,
            "team1_shots": 0,
            "team0_passes": 0,
            "team1_passes": 0,
            "team0_turnovers": 0,
            "team1_turnovers": 0,
            "team0_possession": 0,
            "team1_possession": 0,
        }

        self.possession_streak = 0  # counts consecutive events for current team

    def clamp_position(self, x, y):
        """Keep x,y inside pitch bounds."""
        x = max(self.x_min, min(self.x_max, x))
        y = max(self.y_min, min(self.y_max, y))
        return x, y

    def compute_zone_and_distance(self, team, pos):
        """Return (zone, dist_to_goal) for given position and team."""
        x, y = pos
        # Zone: thirds of the pitch
        if abs(x) < self.length / 6:
            zone = "middle"
        elif (team == "team0" and x > 0) or (team == "team1" and x < 0):
            zone = "attacking"
        else:
            zone = "defensive"

        if team == "team0":
            dist_to_goal = self.x_max - x
        else:
            dist_to_goal = x - self.x_min
        return zone, dist_to_goal

    def generate_pass_position(self, team, start_pos):
        """Generate plausible pass endpoint based on distance bucket."""
        start_x, start_y = start_pos

        # Distance bucket: short, medium, long
        bucket = np.random.choice(["short", "medium", "long"], p=[0.6, 0.3, 0.1])
        if bucket == "short":
            dist = np.random.uniform(5, 10)
        elif bucket == "medium":
            dist = np.random.uniform(10, 20)
        else:  # long risky
            dist = np.random.uniform(20, 35)

        # Angle biased forward
        if team == "team0":
            angle = np.random.uniform(-np.pi / 6, np.pi / 6)  # forward cone
        else:
            angle = np.random.uniform(np.pi - np.pi / 6, np.pi + np.pi / 6)

        dx = dist * np.cos(angle)
        dy = dist * np.sin(angle)
        x, y = self.clamp_position(start_x + dx, start_y + dy)

        return (x, y), dist

    def adjust_event_probabilities(self, last_event, position, team):
        """Dynamic adjustment based on zone, distance, momentum."""
        probs = base_event_probabilities.copy()
        zone, dist_to_goal = self.compute_zone_and_distance(team, position)

        # Momentum
        if last_event == "pass" and self.possession_streak >= 3:
            probs["pass"] += 0.05
        elif last_event == "turnover":
            probs["pass"] += 0.10
            probs["turnover"] -= 0.05

        # Zone effect
        if zone == "attacking":
            probs["shot"] += 0.15
            probs["pass"] -= 0.05
        elif zone == "defensive":
            probs["turnover"] += 0.05

        # Distance effect: closer = more likely to shoot
        if dist_to_goal < 20:
            probs["shot"] += 0.20
        elif dist_to_goal > 60:
            probs["shot"] -= 0.05

        # Normalize
        total = sum(max(0, v) for v in probs.values())
        for k in probs:
            probs[k] = max(0, probs[k]) / total
        return probs

    def simulate_gameflow(self, num_events=100, goal_base_prob=0.2):
        match_log = []
        current_team = "team0"
        current_position = (0.0, 0.0)
        time = 0

        # kickoff
        kickoff_end = (-10.0, 0.0) if current_team == "team0" else (10.0, 0.0)
        zone, dist_to_goal = self.compute_zone_and_distance(current_team, kickoff_end)
        kickoff_event = build_event(time, current_team, "kickoff_pass", (0.0, 0.0), kickoff_end,
                                    possession_duration=1, zone=zone, dist_to_goal=dist_to_goal,
                                    possession_streak=1, pass_distance=10)
        match_log.append(kickoff_event)
        current_position = kickoff_end
        last_event = "kickoff_pass"
        self.possession_streak = 1
        time += 1

        for i in range(num_events):
            # possession stat (crude: based on x side)
            if current_position[0] < 0:
                self.stats["team0_possession"] += 1
            else:
                self.stats["team1_possession"] += 1

            # choose event
            probs = self.adjust_event_probabilities(last_event, current_position, current_team)
            event = np.random.choice(list(probs.keys()), p=list(probs.values()))

            start_x, start_y = current_position
            pass_distance = None

            if event == "pass":
                (new_x, new_y), pass_distance = self.generate_pass_position(current_team, current_position)
                self.stats[f"{current_team}_passes"] += 1
                step_event = "pass"

                # chance of failed pass -> turnover
                if pass_distance > 20 and np.random.rand() < 0.4:
                    step_event = "turnover"
                    self.stats[f"{current_team}_turnovers"] += 1
                    new_x, new_y = new_x, new_y
                    current_team = "team1" if current_team == "team0" else "team0"
                    self.possession_streak = 0
                else:
                    self.possession_streak += 1

            elif event == "shot":
                zone, dist = self.compute_zone_and_distance(current_team, current_position)
                self.stats[f"{current_team}_shots"] += 1
                adjusted_prob = goal_base_prob * (1 / (1 + dist / 20.0))

                if np.random.rand() < adjusted_prob:
                    self.stats[f"{current_team}_goals"] += 1
                    step_event = "goal"
                    new_x, new_y = self.x_max if current_team == "team0" else self.x_min, 0
                    self.possession_streak += 1
                    # reset after goal
                    match_log.append(build_event(time, current_team, step_event,
                                                 current_position, (new_x, new_y),
                                                 zone=zone, dist_to_goal=dist,
                                                 possession_streak=self.possession_streak,
                                                 pass_distance=pass_distance))
                    time += 1
                    current_team = "team1" if current_team == "team0" else "team0"
                    current_position = (0.0, 0.0)
                    kickoff_end = (-10.0, 0.0) if current_team == "team0" else (10.0, 0.0)
                    zone, dist_to_goal = self.compute_zone_and_distance(current_team, kickoff_end)
                    kickoff_event = build_event(time, current_team, "kickoff_pass",
                                                (0.0, 0.0), kickoff_end,
                                                possession_duration=1, zone=zone,
                                                dist_to_goal=dist_to_goal,
                                                possession_streak=1, pass_distance=10)
                    match_log.append(kickoff_event)
                    last_event = "kickoff_pass"
                    self.possession_streak = 1
                    time += 1
                    continue
                else:
                    step_event = "shot"
                    (new_x, new_y) = self.generate_pass_position(current_team, current_position)[0]
                    self.possession_streak += 1

            else:  # turnover
                new_x, new_y = current_position
                self.stats[f"{current_team}_turnovers"] += 1
                step_event = "turnover"
                current_team = "team1" if current_team == "team0" else "team0"
                self.possession_streak = 0

            zone, dist_to_goal = self.compute_zone_and_distance(current_team, (new_x, new_y))
            step = build_event(time, current_team, step_event,
                               current_position, (new_x, new_y),
                               zone=zone, dist_to_goal=dist_to_goal,
                               possession_streak=self.possession_streak,
                               pass_distance=pass_distance)
            match_log.append(step)

            current_position = (new_x, new_y)
            last_event = step_event
            time += 1

        return pd.DataFrame(match_log, columns=columns), self.stats


if __name__ == "__main__":
    ps = PitchSimulator()
    df_flow, stats = ps.simulate_gameflow(num_events=200)
    print(df_flow.head(20))
    print("\n=== FINAL SCOREBOARD ===")
    print(f"Team 0: {stats['team0_goals']} | Team 1: {stats['team1_goals']}")
    print("\n=== MATCH STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")





    


