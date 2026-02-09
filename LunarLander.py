"""
Lunar Lander
Made with Gymnasium
January 2025 - Machine Learning Classes
University Carlos III of Madrid

This template uses the Gymnasium LunarLander-v3 environment.
Students will implement a rule-based agent to land the spacecraft.
"""

import os
import time

import gymnasium as gym
import pygame

# GRAVITY setting
# Moon gravity   -> -1.62
# Mars gravity   -> -3.72
# Earth gravity  -> -9.81 (default, very hard!)
GRAVITY = -9.81

# Agent mode: Set to True to use the Tutorial 1 agent, False for keyboard control
USE_AGENT = True

# Environment configuration
ENV_NAME = "LunarLander-v3"

# Action definitions
ACTION_NOTHING = 0  # Do nothing
ACTION_LEFT_ENGINE = 1  # Fire left orientation engine
ACTION_MAIN_ENGINE = 2  # Fire main engine (downward thrust)
ACTION_RIGHT_ENGINE = 3  # Fire right orientation engine


# GAME STATE CLASS
class GameState:
    def __init__(self, observation):
        """
        Initialize game state from Gymnasium observation.

        The LunarLander-v3 observation space consists of 8 values:
        - obs[0]: x position (horizontal position of the lander)
        - obs[1]: y position (vertical position of the lander)
        - obs[2]: x velocity (horizontal velocity)
        - obs[3]: y velocity (vertical velocity)
        - obs[4]: angle (lander angle)
        - obs[5]: angular velocity (rotation speed)
        - obs[6]: left leg contact (1.0 if touching ground, 0.0 otherwise)
        - obs[7]: right leg contact (1.0 if touching ground, 0.0 otherwise)
        """
        self.x_position = observation[0]
        self.y_position = observation[1]
        self.x_velocity = observation[2]
        self.y_velocity = observation[3]
        self.angle = observation[4]
        self.angular_velocity = observation[5]
        self.left_leg_contact = observation[6]
        self.right_leg_contact = observation[7]

        # Store raw observation for convenience
        self.observation = observation

        # Score tracking
        self.score = 0.0
        self.episode_reward = 0.0

        # Current action
        self.action = ACTION_NOTHING

    def update(self, observation, reward):
        """Update state with new observation and reward."""
        self.x_position = observation[0]
        self.y_position = observation[1]
        self.x_velocity = observation[2]
        self.y_velocity = observation[3]
        self.angle = observation[4]
        self.angular_velocity = observation[5]
        self.left_leg_contact = observation[6]
        self.right_leg_contact = observation[7]
        self.observation = observation
        self.episode_reward += reward
        self.score = self.episode_reward

    def reset(self, observation):
        """Reset state for a new episode."""
        self.__init__(observation)


def print_state(game):
    """
    Print the current game state to the terminal.
    This function shows all available information about the lander.
    """
    print("--------GAME STATE--------")
    print(f"Position: X={game.x_position:.3f}, Y={game.y_position:.3f}")
    print(f"Velocity: X={game.x_velocity:.3f}, Y={game.y_velocity:.3f}")
    print(f"Angle: {game.angle:.3f} rad ({game.angle * 180 / 3.14159:.1f} deg)")
    print(f"Angular Velocity: {game.angular_velocity:.3f}")
    print(f"Left Leg Contact: {game.left_leg_contact:.1f}")
    print(f"Right Leg Contact: {game.right_leg_contact:.1f}")
    print(f"Score: {game.score:.2f}")
    print(f"Last Action: {game.action}")
    print("--------------------------")


# TODO: IMPLEMENT HERE THE METHOD TO SAVE DATA TO FILE
def print_line_data(game):
    """
    Return a string with the game state information to be saved to a file.

    This method should return a string with the relevant information from
    the game state, with values separated by commas.

    The student should decide which features are relevant for the task.

    YOUR CODE HERE
    """
    return f"{game.x_position},{game.y_position},{game.x_velocity},{game.y_velocity},{game.angle},{game.angular_velocity},{game.action},{game.episode_reward},{game.score}\n"


# TODO: IMPLEMENT HERE THE INTELLIGENT AGENT METHOD
def move_tutorial_1(game):
    """
    Implement your own rule-based agent to land the spacecraft.

    This method receives the current game state and must return an action:
    - ACTION_NOTHING (0): Do nothing
    - ACTION_LEFT_ENGINE (1): Fire left orientation engine (rotate clockwise)
    - ACTION_MAIN_ENGINE (2): Fire main engine (slow down descent)
    - ACTION_RIGHT_ENGINE (3): Fire right orientation engine (rotate counter-clockwise)

    Goal: Land safely between the two flags on the landing pad.
    - Landing pad is always at coordinates (0, 0)
    - Landing outside the pad is possible but gives less reward
    - Crash (too fast or wrong angle) ends the episode with negative reward
    - Successful landing gives +100 to +140 points
    - Each leg contact gives +10 points
    - Firing main engine costs -0.3 points per frame
    - Firing side engines costs -0.03 points per frame

    Tips:
    - Use y_velocity to control descent speed (should be slow when landing)
    - Use angle to keep the lander upright (close to 0)
    - Use x_position and x_velocity to center over the landing pad

    YOUR CODE HERE
    """

    # constants
    # these constants control how aggressively the agent tries to correct its position and angle
    # they were tuned through trial and error, visual debugging, they could probably be improved further with better tuning

    GAME_X_POSITION_MULTIPLIER = 0.5  # how aggressively to correct horizontal position
    MAX_TILT = 0.2  # force the lander to stay relatively upright
    ANGLE_ERROR_MULTIPLIER = 0.5  # how aggressively to correct angle
    ANGULAR_VELOCITY_DAMPENING = (
        0.3  # how much to dampen corrections based on current angular velocity
    )
    ANGLE_TOLERANCE = (
        0.05  # if angle error is within this range, consider it acceptable
    )
    MAX_FALLING_VELOCITY = (
        -0.4
    )  # if falling faster than this, fire main engine to slow down
    STOP_MAIN_ENGINE_HEIGHT = (
        0.05  # stop firing main engine close to the ground to improve score slightly
    )
    v_x_error = (
        game.x_velocity - (-game.x_position * GAME_X_POSITION_MULTIPLIER)
    )  # move towards center, proportional to distance and velocity, we want to be moving slowly towards the center as a general rule

    # clamp target angle to max tilt limits to prevent over-rotation
    if v_x_error > MAX_TILT:
        target_angle = MAX_TILT
    elif v_x_error < -MAX_TILT:
        target_angle = -MAX_TILT
    else:
        target_angle = v_x_error

    # correct angle based on current angular velocity to prevent overshooting
    angle_error = target_angle - game.angle
    angle_todo = (ANGLE_ERROR_MULTIPLIER * angle_error) - (
        ANGULAR_VELOCITY_DAMPENING * game.angular_velocity
    )  # dampening constants to prevent overrotation

    # if either leg on the ground, shut down robot
    if game.right_leg_contact or game.left_leg_contact:
        return ACTION_NOTHING

    # fix angle first, if we are tilted
    if angle_todo > ANGLE_TOLERANCE:
        return ACTION_LEFT_ENGINE
    elif angle_todo < -ANGLE_TOLERANCE:
        return ACTION_RIGHT_ENGINE

    # if tilt is ok but we are falling too fast, fire main engine to slow down
    if game.y_velocity < MAX_FALLING_VELOCITY and (
        game.y_position > STOP_MAIN_ENGINE_HEIGHT
    ):  # only fire main engine if we are above the ground
        return ACTION_MAIN_ENGINE

    return ACTION_NOTHING


def move_keyboard(keys_pressed):
    """
    Convert keyboard input to action.

    Controls:
    - UP arrow or W: Fire main engine
    - LEFT arrow or A: Fire left engine
    - RIGHT arrow or D: Fire right engine
    - No key: Do nothing

    Args:
        keys_pressed: pygame key state from pygame.key.get_pressed()

    Returns:
        Action integer (0-3)
    """
    if keys_pressed[pygame.K_UP] or keys_pressed[pygame.K_w]:
        return ACTION_MAIN_ENGINE
    elif keys_pressed[pygame.K_LEFT] or keys_pressed[pygame.K_a]:
        return ACTION_LEFT_ENGINE
    elif keys_pressed[pygame.K_RIGHT] or keys_pressed[pygame.K_d]:
        return ACTION_RIGHT_ENGINE
    else:
        return ACTION_NOTHING


def main():
    """Main game loop."""
    print("=" * 50)
    print("LUNAR LANDER - Machine Learning (UC3M)")
    print("=" * 50)
    print("\nInitializing environment...")

    # Initialize pygame for keyboard input
    pygame.init()

    # Create the environment with human rendering and configured gravity
    env = gym.make(ENV_NAME, gravity=GRAVITY, render_mode="human")

    print(f"Environment: {ENV_NAME}")
    print(f"Gravity: {GRAVITY}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

    if USE_AGENT:
        print("\nRunning in AGENT mode (move_tutorial_1)")
    else:
        print("\nRunning in KEYBOARD mode")
        print("Controls (focus on the game window!):")
        print("  W or UP arrow    -> Fire main engine (slow descent)")
        print("  A or LEFT arrow  -> Fire left engine (rotate clockwise)")
        print("  D or RIGHT arrow -> Fire right engine (rotate counter-clockwise)")
        print("  Q or ESC         -> Quit game")

    print("\nGoal: Land safely on the pad between the two flags!")
    print("-" * 50)

    # Initialize the environment
    observation, info = env.reset()
    game = GameState(observation)

    # FPS controller
    clock = pygame.time.Clock()

    episode_count = 0
    running = True

    # write headers to data file and create the file if it doesn't exist
    # runs prior to the game loop for efficiency
    if not os.path.isfile("lunar_lander_data.csv"):
        with open("lunar_lander_data.csv", "w") as f:
            f.write(
                "x_position,y_position,x_velocity,y_velocity,angle,angular_velocity,action,episode_reward,score\n"
            )
    try:
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False

            if not running:
                break

            # Determine action based on USE_AGENT variable
            if USE_AGENT:
                action = move_tutorial_1(game)
            else:
                keys_pressed = pygame.key.get_pressed()
                action = move_keyboard(keys_pressed)

            # Store action in game state
            game.action = action

            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)

            # Update game state
            game.update(observation, reward)

            # Print state
            print_state(game)
            # calling print_line_data gives us the string to write to the file, we can then write it to the file in one step
            curr_tick_line_data = print_line_data(game)
            with open(
                "lunar_lander_data.csv", "a"
            ) as f:  # open in append mode to add data without overwriting
                f.write(curr_tick_line_data)

            # Check if episode ended
            if terminated or truncated:
                episode_count += 1
                if terminated:
                    if game.score > 0:
                        print(
                            f"\n*** EPISODE {episode_count} COMPLETE! Final Score: {game.score:.2f} ***"
                        )
                        if game.left_leg_contact and game.right_leg_contact:
                            print("*** SUCCESSFUL LANDING! ***\n")
                        else:
                            print("*** Landed but not on both legs ***\n")
                    else:
                        print(
                            f"\n*** CRASH! Episode {episode_count} Final Score: {game.score:.2f} ***\n"
                        )
                else:
                    print(
                        f"\n*** Episode {episode_count} truncated. Final Score: {game.score:.2f} ***\n"
                    )

                # Reset environment
                time.sleep(1)
                observation, info = env.reset()
                game.reset(observation)
                print("New episode started!\n")

            # Control frame rate
            clock.tick(30)

    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    finally:
        env.close()
        pygame.quit()
        print(f"\nGame ended. Total episodes: {episode_count}")
        print("Thank you for playing!")


if __name__ == "__main__":
    main()
