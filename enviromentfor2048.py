import gym
from gym import spaces
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import numpy as np
import time
from gym.envs.registration import register
from gym.utils import seeding

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(Game2048Env, self).__init__()
        # Your initialization code here

        # Action space, observation space, etc.
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = spaces.Box(low=0, high=2**16, shape=(4, 4), dtype=int)
        self.driver = None
        # Initialize seed
        self.seed()
        
    def init_game(self):
        options = Options()
        options.add_argument('--log-level=3')  # This sets the log level to errors only.

        service = Service(executable_path="chromedriver.exe", log_path='NUL')  # 'NUL' discards logs, use 'NULL' on Unix-based systems
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.get("https://play2048.co/")
    
    def calculate_reward(self, state):
        # Flatten the state to a single list of values
        flattened_state = state.flatten()
        # Find the three largest numbers in the flattened state
        three_largest = np.sort(flattened_state)[-1:]
        # Sum the three largest numbers
        reward = np.sum(three_largest)
        return reward

    def step(self, action):
        keys = [Keys.UP, Keys.RIGHT, Keys.DOWN, Keys.LEFT]
        game = self.driver.find_element(By.TAG_NAME, "body")
        game.send_keys(keys[action])
        
        # Add a brief pause for the game to update
        time.sleep(0.1)
        
        # Extract the new state
        new_state = self.get_state()
        done = self.is_game_over()
        
        # Check for the largest numbers and their positions
        max_val = np.max(new_state)
        max_positions = np.argwhere(new_state == max_val)
        
        if len(max_positions) > 1:
            # If there are two or more largest numbers, calculate the minimum distance between them
            min_distance = self.calculate_min_distance(max_positions)
            # Adjust the reward based on the distance
            reward = 10 / (1 + min_distance)  # Example adjustment, you can refine this formula
        else:
            # If there's only one largest number, fall back to the original score-based reward
            reward = 10 * self.calculate_reward(new_state)
        return new_state, reward, done, {}
    
    def calculate_min_distance(self, positions):
        # Calculate the minimum distance between any two positions
        min_distance = float('inf')
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sum(np.abs(positions[i] - positions[j]))
                if dist < min_distance:
                    min_distance = dist
        return min_distance
    
    def reset(self):
        if self.driver is not None:
            try:
                # Attempt to click the "Try again" or "New Game" button to start a new game
                # Note: The exact class name or selector for this button might vary,
                # so you may need to adjust this selector based on the game's current design.
                retry_button = self.driver.find_element(By.CLASS_NAME, "retry-button")
                retry_button.click()
            except Exception as e:
                # If there was an error finding or clicking the button, 
                # (e.g., the button doesn't exist because it's the first game),
                # then initialize the game as normal.
                print(f"Could not find the retry button, initializing a new game. Error: {e}")
                self.driver.get("https://play2048.co/")
        else:
            # If the driver is not initialized, initialize the game.
            self.init_game()
        
        # After starting a new game, wait briefly for the game to reset.
        time.sleep(1)  # Adjust the sleep time as necessary.
        
        return self.get_state()  # Return the initial state for the new game.
    
    def render(self, mode='human', close=False):
        # For simplicity, we don't implement rendering as the game is visible in the browser
        pass
    
    def close(self):
        if self.driver:
            self.driver.quit()
    
    def get_state(self):
        grid = np.zeros((4, 4), dtype=int)
        tiles = self.driver.find_elements(By.CLASS_NAME, "tile")
        for tile in tiles:
            # Extract the tile's value and position from its class names
            classes = tile.get_attribute('class').split()
            value = None
            row = None
            col = None
            for cls in classes:
                # Filter classes to find those that indicate a numeric tile value
                if cls.startswith('tile-') and not any(non_numeric in cls for non_numeric in ['new', 'merged', 'position']):
                    try:
                        # Assuming class name format is 'tile-{value}' where {value} is the numeric value of the tile
                        value = int(cls.split('-')[1])
                    except ValueError:
                        print(f"Error parsing value from class: {cls}")
                        continue
                elif 'position-' in cls:
                    try:
                        # Assuming class name format is 'tile-position-{row}-{col}'
                        positions = cls.split('-')[2:]
                        row, col = int(positions[0]) - 1, int(positions[1]) - 1
                    except ValueError:
                        print(f"Error parsing position from class: {cls}")
                        continue
            if value is not None and row is not None and col is not None:
                grid[row, col] = value
        return grid



    def get_score(self):
        try:
            score_container = self.driver.find_element(By.CLASS_NAME, "score-container")
            score = int(score_container.text.split()[0])
            return score
        except:
            # Fallback if score cannot be found, consider logging or handling differently
            return 0

    def is_game_over(self):
        overlay = self.driver.find_elements(By.CLASS_NAME, "game-over")
        return len(overlay) > 0
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

