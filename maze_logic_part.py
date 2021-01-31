import numpy as np
import random
from mini_object import Point, Slope, VISION_8, VISION_4
from typing import Tuple, Union, List, Optional, Dict, Any
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name
import os
import json
import sys

class Vision():
    __slots__ = ('dist_to_wall', 'dist_to_treasure')
    def __init__(self,
                dist_to_wall: Union[float, int],
                dist_to_treasure: Union[float, int],
                ):
        self.dist_to_wall = float(dist_to_wall)
        self.dist_to_treasure = float(dist_to_treasure)

class DrawableVision():
    __slots__ = ('wall_location', 'treasure_location')
    def __init__(self,
                wall_location: Point,
                treasure_location: Point = None,
                ):
        self.wall_location = wall_location
        self.treasure_location = treasure_location

class maze():
    """maze logic part
    """
    def __init__(self, board_size: tuple, maps: List[List[Point]], 
                player: str = 'player', 
                chromosome: Optional[Dict[str, List[np.ndarray]]] = None,
                adv_seed: Optional[int] = None, 
                treas_seed: Optional[int] = None,
                map_seed: Optional[int] = None,
                starting_direction: Optional[str] = None,
                hidden_layer_architecture: Optional[List[int]] = [20, 12],
                hidden_activation: Optional[ActivationFunction] = 'relu',
                output_activation: Optional[ActivationFunction] = 'sigmoid',
                lifespan: Optional[Union[int, float]] = np.inf
                ):
        self.lifespan = lifespan
        self.score = 0
        self._fitness = 0
        self._frames = 0
        self._frames_since_last_treasure = 0
        self.possible_directions = ('l', 'd', 'r', 'u')
        self.board_size = board_size

        self.maps = maps

        self._vision_type = VISION_8
        self._vision: List[Vision] = [None] * len(self._vision_type)
        self._drawable_vision: List[DrawableVision] = [None] * len(self._vision_type)

        num_inputs = len(self._vision_type) * 2 + 4 + 2 + 2#@TODO: Add one-hot back in 
        self.vision_as_array: np.ndarray = np.zeros((num_inputs, 1))

        # Select map
        if not map_seed:
            map_seed = np.random.randint(-1000000000,1000000000)
        self.map_seed = map_seed
        self.rand_map = random.Random(self.map_seed)
        self.generate_walls()
        
        # 'human' or 'computer'
        self.player = player
                
        # Initial adventurer
        if not adv_seed:
            adv_seed = np.random.randint(-1000000000,1000000000)
        self.adv_seed = adv_seed
        self.rand_adv = random.Random(self.adv_seed)
        self.adv_pos = self.generate_valid_pos(self.rand_adv, self.walls)
        self.wall_and_adv_pos = [self.adv_pos] + self.walls

        # Initial treasure
        if treas_seed is None:
            treas_seed = np.random.randint(-1000000000,1000000000)
        self.treas_seed = treas_seed
        self.rand_treas = random.Random(self.treas_seed)
        self.generate_treasure()

        # Initial direction
        # Give human player react time
        if not starting_direction:
            distance = []
            for slope in VISION_4:
                vision, _ = self.look_in_direction(slope)
                distance.append(vision.dist_to_wall)
            index = np.argmin(distance)
            self.direction = self.possible_directions[index]
        else:
            self.direction = starting_direction
        if player.lower() == 'computer':
            self.hidden_layer_architecture = hidden_layer_architecture        
            self.hidden_activation = hidden_activation
            self.output_activation = output_activation
            self.vision_as_array: np.ndarray = np.zeros((num_inputs, 1))
            self.network_architecture = [num_inputs]                            # Inputs
            self.network_architecture.extend(self.hidden_layer_architecture)    # Hidden layers
            self.network_architecture.append(4 + 2)                                 # 4 outputs, ['u', 'd', 'l', 'r']
                                                                                # 2 continual passing output
            self.network = FeedForwardNetwork(self.network_architecture,
                                            get_activation_by_name(self.hidden_activation),
                                            get_activation_by_name(self.output_activation)
            )
            self.pass_output = np.array([[0], [0]])
            if chromosome:
                self.network.params = chromosome
            else:
                pass

        self.starting_direction = self.direction

        self.is_alive = True

    def level_up(self):

        self.generate_walls()

        self.rand_adv = random.Random(self.adv_seed)
        self.adv_pos = self.generate_valid_pos(self.rand_adv, self.walls)
        self.wall_and_adv_pos = [self.adv_pos] + self.walls

        self.generate_treasure()

        distance = []
        for slope in VISION_4:
            vision, _ = self.look_in_direction(slope)
            distance.append(vision.dist_to_wall)
        index = np.argmin(distance)
        self.direction = self.possible_directions[index]
        # reset passing output
        if self.player == 'computer':
            self.pass_output = np.array([[0], [0]])

    @property
    def fitness(self):
        return self._fitness
    
    def calculate_fitness(self):
        # Same as in snake-AI
        # self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self._frames)**1.3) * (self.score**1.2))
        self._fitness = (self._frames) + (self.score**4.1)*500 - (((.25 * self._frames)**1.3) * self.score)
        self._fitness = max(self._fitness, .1)
                    
    def generate_valid_pos(self, r_obj: random.Random, avoid: List[Point]) -> Point:
        width = self.board_size[0]
        height = self.board_size[1]
        # Find all posible point could generate treasure
        possibilities = [divmod(i, height)  for i in range(width * height) 
                                            if divmod(i, height) not in avoid]
        loc = r_obj.choice(possibilities)
        return Point(loc[0], loc[1])

    def generate_treasure(self):
        self.treas_pos = self.generate_valid_pos(self.rand_treas, self.wall_and_adv_pos)

    def generate_walls(self):
        walls = self.rand_map.choice(self.maps)
        self.walls = [w.copy() for w in walls]

    def update(self, direction=None) -> bool:
        if self.is_alive:
            self._frames += 1
            if self.player == 'human':
                if direction:
                    self.direction = direction
            elif self.player == 'computer':
                self.look()
                self.network.feed_forward(self.vision_as_array)
                self.direction = self.possible_directions[np.argmax(self.network.out[:4])]
                self.pass_output = self.network.out[4:6]
            else:
                raise ValueError('Error input for player.')
            return True
        else:
            return False
    
    def move(self) -> bool:
        if not self.is_alive:
            return False
        
        direction = self.direction.lower()
        # Find next position
        adv_pos = self.adv_pos
        if direction == 'u':
            next_pos = Point(adv_pos.x, adv_pos.y - 1)
        elif direction == 'd':
            next_pos = Point(adv_pos.x, adv_pos.y + 1)
        elif direction == 'r':
            next_pos = Point(adv_pos.x + 1, adv_pos.y)
        elif direction == 'l':
            next_pos = Point(adv_pos.x - 1, adv_pos.y)

        # Is the next position we want to move valid?
        if self._is_valid(next_pos):
            # get the treasure
            if next_pos == self.treas_pos:
                self.score += 1
                self._frames_since_last_treasure = -1
                self.generate_treasure()
                if self.score >= 120:
                    print('Mission Complete')
                    self.is_alive = False
                if self.score % 3 == 0:
                    self.level_up()
            # move to next pos
            self.adv_pos = next_pos
            self.wall_and_adv_pos = [next_pos] + self.walls

            self._frames_since_last_treasure += 1
            #@NOTE: If you have different sized grids you may want to change this
            if self._frames_since_last_treasure > 130:
                self.is_alive = False
                return False

            return True
        else:
            self.is_alive = False
            return False

    def _is_valid(self, pos: Point) -> bool:
        """Determine where position is valide for moving
        """
        if (pos.x < 0) or (pos.x > self.board_size[0] - 1):
            return False
        if (pos.y < 0) or (pos.y > self.board_size[1] - 1):
            return False
        
        if pos in self.wall_and_adv_pos:
            return False
        # Otherwise you good
        else:
            return True

    def look(self):
        # collect adventurer vision information
        for i, slope in enumerate(self._vision_type):
            vision, drawable_vision = self.look_in_direction(slope)
            self._vision[i] = vision
            self._drawable_vision[i] = drawable_vision
        
        # Update the input array
        self._vision_as_input_array()
    
    def look_in_direction(self, slope: Slope) -> Tuple['vision','drawable_vision']:
        dist_to_wall = None
        dist_to_treasure = np.inf

        wall_location = None
        treasure_location = None

        position = self.adv_pos.copy()
        distance = 1.0
        total_distance = 0.0

        # Move step by step with slope direction
        # Can't start by looking at yourself
        position.x += slope.dx
        position.y += slope.dy
        total_distance += distance
        wall_found = False
        treasure_found = False

        while self._within_wall(position):
            if not wall_found and position in self.walls:
                dist_to_wall = total_distance
                wall_location = position.copy()
                wall_found = True
            if not treasure_found and position == self.treas_pos:
                dist_to_treasure = total_distance
                treasure_location = position.copy()
                treasure_found = True
            
            position.x += slope.dx
            position.y += slope.dy
            total_distance += distance
        
        if not wall_found:
            dist_to_wall = total_distance
            wall_location = position.copy()

        # for VISION_8
        dist_to_wall = 1.0 / dist_to_wall
        dist_to_treasure = 1.0 / dist_to_treasure        

        vision = Vision(dist_to_wall, dist_to_treasure)
        drawable_vision = DrawableVision(wall_location, treasure_location)
        return (vision, drawable_vision)

    def _vision_as_input_array(self) -> None:
        # Split _vision into np array where rows [0-1] are _vision[0].dist_to_wall, _vision[0].dist_to_treasure,
        # rows [2-4] are _vision[1].dist_to_wall, _vision[1].dist_to_treasure, etc. etc. etc.
        for va_index, v_index in zip(range(0, len(self._vision) * 2, 2), range(len(self._vision))):
            vision = self._vision[v_index]
            self.vision_as_array[va_index, 0]     = vision.dist_to_wall
            self.vision_as_array[va_index + 1, 0] = vision.dist_to_treasure

        i = len(self._vision) * 2  # Start at the end

        direction = self.direction[0].lower()
        # One-hot encode direction
        direction_one_hot = np.zeros((len(self.possible_directions), 1))
        direction_one_hot[self.possible_directions.index(direction), 0] = 1
        self.vision_as_array[i: i + len(self.possible_directions)] = direction_one_hot

        i += len(self.possible_directions)

        treas_dir = np.zeros((2, 1))
        diff = self.adv_pos - self.treas_pos
        treas_dir[0,0] = (diff.x > 0) - (diff.x < 0)
        treas_dir[1,0] = (diff.y > 0) - (diff.y < 0)
        self.vision_as_array[i: i + 2] = treas_dir

        i += 2
        self.vision_as_array[i: i + 2] = self.pass_output        

    def _within_wall(self, position: Point) -> bool:
        return  position.x >= 0 and position.y >= 0 and \
                position.x < self.board_size[0] and \
                position.y < self.board_size[1]

    def check_pos_type(self, pos: Union[tuple, Point]) -> int:
        """Check what type is filled into the position
        
        Possible type:
            none, adventurer, treasure, wall.

        """
        if isinstance(pos, tuple) and len(pos) == 2:
            pos = Point(pos[0], pos[1])
        
        if pos == self.adv_pos:
            return 'adventurer'
        elif pos == self.treas_pos:
            return 'treasure'
        elif pos in self.walls:
            return 'wall'
        else:
            return 'none'

    def add_wall(self, pos: Union[tuple, Point]):
        """Add a wall to now map.
        """
        if isinstance(pos, tuple) and len(pos) == 2:
            pos = Point(pos[0], pos[1])

        self.walls.append(pos)
    
    def delete_wall(self, pos: Union[tuple, Point]):
        """Delete a wall exist in now map.
        """
        if isinstance(pos, tuple) and len(pos) == 2:
            pos = Point(pos[0], pos[1])
        
        if pos in self.walls:
            self.walls.remove(pos)
        else:
            raise ValueError('Input "Pos" must exist in the map walls.')


def save_maze(population_folder: str, individual_name: str, Maze: maze, settings: Dict[str, Any]) -> None:
    # Make population folder if it doesn't exist
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # Save off settings
    if 'settings.json' not in os.listdir(population_folder):
        f = os.path.join(population_folder, 'settings.json')
        with open(f, 'w', encoding='utf-8') as out:
            json.dump(settings, out, sort_keys=True, indent=4)

    # Make directory for the individual
    individual_dir = os.path.join(population_folder, individual_name)
    os.makedirs(individual_dir)

    # Save some constructor information for replay
    # @NOTE: No need to save chromosome since that is saved as .npy
    # @NOTE: No need to save board_size or hidden_layer_architecture
    #        since these are taken from settings
    constructor = {}
    constructor['adv_seed'] = Maze.adv_seed
    constructor['treas_seed'] = Maze.treas_seed
    constructor['map_seed'] = Maze.map_seed
    constructor['starting_direction'] = Maze.starting_direction
    maze_constructor_file = os.path.join(individual_dir, 'constructor_params.json')

    # Save
    with open(maze_constructor_file, 'w', encoding='utf-8') as out:
        json.dump(constructor, out, sort_keys=True, indent=4)

    L = len(Maze.network.layer_nodes)
    for l in range(1, L):
        w_name = 'W' + str(l)
        b_name = 'b' + str(l)

        weights = Maze.network.params[w_name]
        bias = Maze.network.params[b_name]

        np.save(os.path.join(individual_dir, w_name), weights)
        np.save(os.path.join(individual_dir, b_name), bias)

def load_maze(  population_folder: str, individual_name: str, maps: List[List[Point]], 
                settings: Optional[Union[Dict[str, Any], str]] = None) -> maze:
    if not settings:
        f = os.path.join(population_folder, 'settings.json')
        if not os.path.exists(f):
            raise Exception("settings needs to be passed as an argument if 'settings.json' does not exist under population folder")
        
        with open(f, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    elif isinstance(settings, dict):
        settings = settings

    elif isinstance(settings, str):
        filepath = settings
        with open(filepath, 'r', encoding='utf-8') as fp:
            settings = json.load(fp)

    params = {}
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            params[param] = np.load(os.path.join(population_folder, individual_name, fname))
        else:
            continue

    # Load constructor params for the specific maze
    constructor_params = {}
    maze_constructor_file = os.path.join(population_folder, individual_name, 'constructor_params.json')
    with open(maze_constructor_file, 'r', encoding='utf-8') as fp:
        constructor_params = json.load(fp)

    Maze = maze(settings['board_size'], maps=maps, player='computer', 
                chromosome=params, 
                adv_seed=constructor_params['adv_seed'],
                treas_seed=constructor_params['treas_seed'],
                map_seed=constructor_params['map_seed'],
                starting_direction=constructor_params['starting_direction'],
                hidden_layer_architecture=settings['hidden_network_architecture'],
                hidden_activation=settings['hidden_layer_activation'],
                output_activation=settings['output_layer_activation'],
                lifespan=settings['lifespan']
                )
    # Maze = maze(settings['board_size'], maps=maps, player='computer', 
    #             chromosome=params, 
    #             adv_seed=None,
    #             treas_seed=None,
    #             map_seed=None,
    #             starting_direction=constructor_params['starting_direction'],
    #             hidden_layer_architecture=settings['hidden_network_architecture'],
    #             hidden_activation=settings['hidden_layer_activation'],
    #             output_activation=settings['output_layer_activation'],
    #             lifespan=settings['lifespan']
    #             )
    return Maze