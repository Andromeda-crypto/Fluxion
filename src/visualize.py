''' 
takes the dataframe from simulate.py and plots it to make 
it easier to visualize and tweak as we go on forward
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulate_data import PitchSimulator


def draw_pitch(ax, length=100, width= 50):
    ax.set_xlim(-length/2,length/2)
    ax.set_ylim(-width/2, width/2)
    ax.axhline(0,color='k', linewidth=0.5)
    ax.axvline(0,color='k', linewidth= 0.5)

    ax.plot([length/2, length/2], [-7.32/2, 7.32/2], color='red', linewidth= 2)
    ax.plot([-length/2, -length/2], [-7.32/2, 7/32/2], color='red', linewidth= 2)


def animate_simulation(df):
    """
    Step by step simualtion of df
    """
    fig, ax = plt.subplots(figsize=(10,6))
    draw_pitch(ax)


    # color coding specific events 
    event_colors = {
        "pass": "blue",
        "shot": "orange",
        "goal": "gold",
        "turnover": "black",
        "kickoff_pass": "green"
    }

    ball, = ax.plot([],[],'o', color='blue', markersize=8)  # moving ball
    title = ax.set_title("")


    def update(frame):
        row = df.iloc[frame]
        color = event_colors.get(row["event"], "gray")
        ball.set_data([row("end_x")], [row("end_y")])
        ball.set_color(color)
        title.set_text(f"t={row['time']} | {row['team']} | {row['event']}")

        return ball, title
    
    ani = animation.FuncAnimation(fig, update, frames = len(df), interval=500, blit=False, repeat=False)
    plt.show()


def plot_event_heatmap(df, length=100, width=50):
    """
    Static heatmap of event end positions
    """
    plt.figure(figsize=(10, 6))
    plt.hexbin(df["end_x"], df["end_y"], gridsize=20, cmap="viridis", alpha=0.8)
    plt.colorbar(label="Event density")
    plt.title("Event End Position Heatmap")
    plt.xlim(-length/2, length/2)
    plt.ylim(-width/2, width/2)
    plt.show()


if __name__ == "__main__":
    ps = PitchSimulator()
    df = ps.simulate_gameflow(num_events=200)

    animate_simulation(df)

    plot_event_heatmap(df)
    
