"""
visualize.py
Visualizes simulated football-like event data from simulate_data.py
and shows a scoreboard + statsheet.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulate_data import PitchSimulator


def draw_pitch(ax, length=100, width=50):
    """Draws a simple pitch centered at origin."""
    ax.set_xlim(-length / 2, length / 2)
    ax.set_ylim(-width / 2, width / 2)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)

    # Goals
    ax.plot([length / 2, length / 2], [-7.32 / 2, 7.32 / 2], color="red", linewidth=2)
    ax.plot([-length / 2, -length / 2], [-7.32 / 2, 7.32 / 2], color="red", linewidth=2)


def animate_simulation(df):
    """Step-by-step animation of events in DataFrame."""
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_pitch(ax)

    # event-specific colors
    event_colors = {
        "pass": "blue",
        "shot": "orange",
        "goal": "gold",
        "turnover": "black",
        "kickoff_pass": "green",
    }

    ball, = ax.plot([], [], "o", color="blue", markersize=8)  # moving ball
    arrows = []  # to hold event arrows
    title = ax.set_title("")

    def update(frame):
        row = df.iloc[frame]
        color = event_colors.get(row["event"], "gray")

        # Update ball position
        ball.set_data([row["end_x"]], [row["end_y"]])
        ball.set_color(color)

        # Draw arrow for this event
        arrow = ax.arrow(
            row["start_x"],
            row["start_y"],
            row["end_x"] - row["start_x"],
            row["end_y"] - row["start_y"],
            head_width=1.5,
            head_length=2.5,
            fc=color,
            ec=color,
            alpha=0.6,
        )
        arrows.append(arrow)

        title.set_text(f"t={row['time']} | {row['team']} | {row['event']}")
        return ball, arrow, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(df), interval=200, blit=False, repeat=False
    )
    plt.show()


def plot_event_heatmap(df, length=100, width=50):
    """Static heatmap of event end positions."""
    plt.figure(figsize=(10, 6))
    plt.hexbin(df["end_x"], df["end_y"], gridsize=30, cmap="viridis", alpha=0.8)
    plt.colorbar(label="Event density")
    plt.title("Event End Position Heatmap")
    plt.xlim(-length / 2, length / 2)
    plt.ylim(-width / 2, width / 2)
    plt.show()


def display_stats(stats):
    """Prints scoreboard + statsheet nicely (flat dict from simulate_data.py)."""
    print("\n=== SCOREBOARD ===")
    print(f"Team0: {stats['team0_goals']}  |  Team1: {stats['team1_goals']}")

    print("\n=== MATCH STATS ===")
    for team in ["team0", "team1"]:
        print(f"\n{team.upper()}:")
        print(f"  Passes      : {stats[f'{team}_passes']}")
        print(f"  Shots       : {stats[f'{team}_shots']}")
        print(f"  Turnovers   : {stats[f'{team}_turnovers']}")
        print(f"  Goals       : {stats[f'{team}_goals']}")
        print(f"  Possession  : {stats[f'{team}_possession']}")


if __name__ == "__main__":
    ps = PitchSimulator()
    df, stats = ps.simulate_gameflow(num_events=200)

    # Step-by-step replay
    animate_simulation(df)

    # Aggregate view
    plot_event_heatmap(df)

    # Scoreboard + statsheet
    display_stats(stats)

