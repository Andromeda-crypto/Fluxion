''' python script to simulate data
Will be uding football data as a lens throughout the project
but the code should be general enough to be used for other purposes for prediction of chaotic systems
'''


import numpy as np
import random
import pandas as pd


# event describes whether teh action that follwed was a pass, shot, turnover, etc
# x and y are the coordinates on the pitch for the postion of the player performing the event
# possestion duration is the seconds since last possession change
columns = ["team", "event", "x", "y", "possesion_duration"]

# now we assign some probabilities to the events
# goal has a conditonal on shot

event_probabilities = {
    "pass": 0.7,
    "shot": 0.1,
    "turnover": 0.15,
    "goal" : 0.05} # will add m0re events such as foul, corner, freekic, later

# now we simulate some data using numpy

def simulate_data(num_events=100):
    data = []
    for i in range(num_events):
        team = np.random.choice(["team0", "team1"])
        event = np.random.choice(list(event_probabilities.keys()), p=list(event_probabilities.values()))
        if event == "shot":
            if np.random.rand() < event_probabilities["goal"]:
                event = "goal"
        x = np.random.uniform(0,100)
        y = np.random.uniform(0,100)
        possesion_duration = random.randint(0,10)
    
        table = (team, event,x,y,possesion_duration)

        data.append(table)
        

    df = pd.DataFrame(data, columns=columns)
    return df

simulate_data()


''' now we can simulate a football pitch as an x-y plane 
it will help determine the position of the player, and the ball 
this will help with more accurate and realistic predictions '''

class PitchSimulator:
    def __init__(self, length = 100, width=50):
        self.length = length
        self.width = width

    def generate_postion(team, event_type):
        start_position = (0,0) # let's take origin of plane to be the center of the pitch
        


    


