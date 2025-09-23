''' python script to simulate data
Will be using football data as a lens throughout the project
but the code should be general enough to be used for other purposes for prediction of chaotic systems
'''

import numpy as np
import random
import pandas as pd

# Columns for simple simulated events (consistent spelling)
columns = ["time", "team", "event", "start_x", "start_y", "end_x", "end_y", "possession_duration"]

# Base event probabilities (goal is handled as conditional on shot)
event_probabilities = {
    "pass": 0.75,
    "shot": 0.10,
    "turnover": 0.15
}

# Simple data-only simulator (keeps parity with earlier function)
def simulate_data(num_events=100):
    data = []
    for i in range(num_events):
        team = np.random.choice(["team0", "team1"])
        event = np.random.choice(list(event_probabilities.keys()), p=list(event_probabilities.values()))
        # If it's a shot, we may upgrade to a goal later in gameflow; here we keep as shot
        x = np.random.uniform(-50, 50)  # center-origin pitch (length normalized to 100)
        y = np.random.uniform(-25, 25)  # width normalized to 50
        possession_duration = random.randint(0, 10)

        row = (i, team, event, x, y, x, y, possession_duration)  # start==end for this simple generator
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df


class PitchSimulator:
    def __init__(self, length=100, width=50):
        # length and width are in the same units you used before; origin at center
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
        # If a start_pos is provided, bias new position relative to it; otherwise use team bias
        if start_pos is None:
            start_x, start_y = 0.0, 0.0
        else:
            start_x, start_y = start_pos

        # Basic biasing rules:
        # - team0 attacks to +x, team1 attacks to -x
        # - shot should be near opponent goal
        if event_type == "shot":
            if team == "team0":
                x = self.x_max  # aim at right goal
                y = start_y + np.random.uniform(-5, 5)
            else:
                x = self.x_min  # aim at left goal
                y = start_y + np.random.uniform(-5, 5)
        elif event_type == "pass":
            # short-to-medium pass biased forward for attacking team
            if team == "team0":
                x = start_x + np.random.uniform(-5, 15)  # can go slightly backward or forward
            else:
                x = start_x + np.random.uniform(-15, 5)
            y = start_y + np.random.uniform(-8, 8)
        elif event_type == "turnover":
            # stay roughly where we are (possession lost)
            x, y = start_x, start_y
        else:
            # generic random position on the appropriate half
            if team == "team0":
                x = np.random.uniform(0, self.x_max)
            else:
                x = np.random.uniform(self.x_min, 0)
            y = np.random.uniform(self.y_min, self.y_max)

        # clamp the position so we don't drift off-pitch
        x, y = self.clamp_position(x, y)
        return x, y

    def simulate_gameflow(self, num_events=100, goal_base_prob=0.2):
        """
        Simulate a short gameflow:
        - kickoff at center is forced first event
        - then loop num_events times sampling pass/shot/turnover
        - shot may become goal (probability reduced by distance)
        Returns a pandas DataFrame of the logged events.
        """
        match_log = []
        current_team = "team0"
        current_position = (0.0, 0.0)  # kickoff
        time = 0

        # forced kickoff_pass
        kickoff_end = (-10.0, 0.0) if current_team == "team0" else (10.0, 0.0)
        kickoff_entry = {
            "time": time,
            "team": current_team,
            "event": "kickoff_pass",
            "start_x": 0.0,
            "start_y": 0.0,
            "end_x": kickoff_end[0],
            "end_y": kickoff_end[1],
            "possession_duration": 0
        }
        match_log.append(kickoff_entry)
        current_position = kickoff_end
        time += 1

        for i in range(num_events):
            # sample event
            event = np.random.choice(list(event_probabilities.keys()), p=list(event_probabilities.values()))

            start_x, start_y = current_position

            # generate end position based on event and team
            new_x, new_y = self.generate_position(current_team, event, start_pos=current_position)

            # if shot: calculate goal chance (closer => larger chance)
            if event == "shot":
                # distance roughly by x-distance to target goal (abs to keep consistent)
                if current_team == "team0":
                    dist = abs(self.x_max - start_x)
                else:
                    dist = abs(self.x_min - start_x)

                # simple decay: closer shots have higher chance
                adjusted_prob = goal_base_prob * (1 / (1 + dist / 20.0))

                if np.random.rand() < adjusted_prob:
                    # goal happened
                    step = {
                        "time": time,
                        "team": current_team,
                        "event": "goal",
                        "start_x": start_x,
                        "start_y": start_y,
                        "end_x": new_x,
                        "end_y": new_y,
                        "possession_duration": random.randint(0, 10)
                    }
                    match_log.append(step)

                    # reset to kickoff with opposite team starting
                    current_team = "team1" if current_team == "team0" else "team0"
                    current_position = (0.0, 0.0)
                    time += 1
                    # forced kickoff after goal
                    kickoff_end = (-10.0, 0.0) if current_team == "team0" else (10.0, 0.0)
                    kickoff_entry = {
                        "time": time,
                        "team": current_team,
                        "event": "kickoff_pass",
                        "start_x": 0.0,
                        "start_y": 0.0,
                        "end_x": kickoff_end[0],
                        "end_y": kickoff_end[1],
                        "possession_duration": 0
                    }
                    match_log.append(kickoff_entry)
                    current_position = kickoff_end
                    time += 1
                    # continue outer loop (we've consumed this iteration)
                    continue
                else:
                    # shot but no goal: treat as shot event (possession may remain or change)
                    step_event = "shot"
            elif event == "turnover":
                # possession switches - turnover is by current team and then opponent gets the ball at same position
                step_event = "turnover"
                # the team recorded for the event is the team that *lost* possession in many logs;
                # here we log the team that was in possession (current_team), then flip possession
            else:
                step_event = "pass"

            # Build log step (for pass/shot/turnover non-goal)
            step = {
                "time": time,
                "team": current_team,
                "event": step_event,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": new_x,
                "end_y": new_y,
                "possession_duration": random.randint(0, 10)
            }
            match_log.append(step)

            # Update state: if turnover, switch possession to opponent
            if step_event == "turnover":
                current_team = "team1" if current_team == "team0" else "team0"
                # ball stays where turnover occurred
                current_position = (new_x, new_y)
            else:
                # possession remains with current team
                current_position = (new_x, new_y)

            # keep time moving
            time += 1

        # return a DataFrame for downstream ease-of-use
        df = pd.DataFrame(match_log, columns=columns)
        return df


# Example quick usage (remove or wrap in if __name__ == '__main__' when adding to repo)
# df_simple = simulate_data(200)
# ps = PitchSimulator()
# df_flow = ps.simulate_gameflow(num_events=200)
# print(df_flow.head())
if __name__ == "__main__":
    ps = PitchSimulator()
    df_flow = ps.simulate_gameflow(num_events=200)
    print(df_flow)






    


