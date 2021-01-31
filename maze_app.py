from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from typing import List, Tuple
import sys
import maze_logic_part as mlp
import numpy as np
from math import sqrt
from decimal import Decimal
import random
import csv
import glob
import os
import pickle
import re
from neural_network import FeedForwardNetwork, sigmoid, linear, relu
from settings import settings
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, roulette_wheel_selection, tournament_selection
from genetic_algorithm.mutation import gaussian_mutation, random_uniform_mutation
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.crossover import uniform_binary_crossover, single_point_binary_crossover
from genetic_algorithm.individual import Individual
from mini_object import Point
from help_func import load_points, save_points, get_jump_index, next_file_index

SQUARE_SIZE = (35, 35)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings, file_path, show=True, fps=20):
        super().__init__()

        self.setAutoFillBackground(True)
        # palette = self.palette()
        # palette.setColor(self.backgroundRole(), QtGui.QColor(240, 240, 240))
        self.settings = settings
        self.player = self.settings['player']
        self._SBX_eta = self.settings['SBX_eta']
        self._mutation_bins = np.cumsum([self.settings['probability_gaussian'],
                                        self.settings['probability_random_uniform']
        ])
        self._crossover_bins = np.cumsum([self.settings['probability_SBX'],
                                         self.settings['probability_SPBX']
        ])
        self._SPBX_type = self.settings['SPBX_type'].lower()
        self._mutation_rate = self.settings['mutation_rate']

        # Determine size of next gen based off selection type
        self._next_gen_size = None
        if self.settings['selection_type'].lower() == 'plus':
            self._next_gen_size = self.settings['num_parents'] + self.settings['num_offspring']
        elif self.settings['selection_type'].lower() == 'comma':
            self._next_gen_size = self.settings['num_offspring']
        else:
            raise Exception('Selection type "{}" is invalid'.format(self.settings['selection_type']))

        self.board_size = settings['board_size']
        self.maze_widget_width = SQUARE_SIZE[0] * self.board_size[0]
        self.maze_widget_height = SQUARE_SIZE[1] * self.board_size[1]

        self.top = 150
        self.left = 150
        self.width = self.maze_widget_width
        # Menu take 20 height(while menu.height=30)
        # StatusBar take 30 height
        self.height = self.maze_widget_height + 20 + 30

        # Initial maps
        self.file_path = file_path
        self.map_fpath = os.path.join(file_path, f'maps{self.board_size[0]}x{self.board_size[1]}')
        maps_path = glob.glob(os.path.join(self.map_fpath, '*'))
        self.maps = []
        for m in maps_path:
            self.maps.append(load_points(m))
        file_index_list = self.get_file_index_list(maps_path)
        self.maps_jump_id, self.maps_max_id = get_jump_index(file_index_list)

        # computer mode
        if self.player == 'computer':
            individuals: List[Individual] = []
            PGP = self.settings['PGP']
            for _ in range(int(self.settings['num_parents'] * PGP)):
                individual = mlp.maze(  self.board_size, 
                                        maps=self.maps, 
                                        player=self.player, 
                                        hidden_layer_architecture=self.settings['hidden_network_architecture'],
                                        hidden_activation=self.settings['hidden_layer_activation'],
                                        output_activation=self.settings['output_layer_activation'],
                                        lifespan=self.settings['lifespan'])
                individuals.append(individual)
            for _ in range(int(self.settings['num_parents'] * PGP), self.settings['num_parents']):
                individual = mlp.load_maze(self.settings['lmp'], self.settings['lmn'], self.maps, self.settings)
                individuals.append(individual)

            self.best_fitness = 0
            self.best_score = 0

            self._current_individual = 0
            self.population = Population(individuals)

            self.maze = self.population.individuals[self._current_individual]
            self.current_generation = 0

            self.save_condiction = 0
            
            if not os.path.exists(self.settings['smfp']):
                os.makedirs(self.settings['smfp'])
            
        elif self.player == 'human':
            self.maze = mlp.maze(self.board_size, self.maps, 'human')
        else:
            raise ValueError('Undefine player.')

        self.init_window()    
        self.timer = QtCore.QTimer(self)
        if self.player == 'human':
            self.timer.timeout.connect(self.Hplayer_update)
        else:
            self.timer.timeout.connect(self.Cplayer_update)
        self.timer.start(1000./fps)

        self.pause = 0
        self.fps = fps

        if show:
            self.show()

    def init_window(self):
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle('Maze AI')
        self.setGeometry(self.top, self.left, self.width, self.height)

        self.maze_widget_window = MazeWidget(self.centralWidget, self.board_size, self.maze)
        self.maze_widget_window.setGeometry(QtCore.QRect(0, 0, self.maze_widget_width, self.maze_widget_height))
        self.maze_widget_window.setObjectName('maze_widget_window')

        # menu
        menubar = self.menuBar()        
        ActionMenu = menubar.addMenu('Action')

        startAct = QtWidgets.QAction('Start', self)
        startAct.setShortcut('Ctrl+S')
        pauseAct = QtWidgets.QAction('Pause', self)
        pauseAct.setShortcut('Ctrl+P')
        setFPSAct = QtWidgets.QAction('Set fps', self)
        setFPSAct.setShortcut('Ctrl+H')
        newMapAct = QtWidgets.QAction('New Map', self)
        newMapAct.setShortcut('Ctrl+M')

        startAct.triggered.connect(self.startEvt)
        pauseAct.triggered.connect(self.pauseEvt)
        setFPSAct.triggered.connect(self.setFPSEvt)
        newMapAct.triggered.connect(self.newMapEvt)

        ActionMenu.addAction(startAct)
        ActionMenu.addAction(pauseAct)
        ActionMenu.addAction(setFPSAct)
        ActionMenu.addAction(newMapAct)

        # statusBar
        self.statusbar = self.statusBar()    
        self.maze_widget_window.msg2Statusbar[str].connect(self.statusbar.showMessage)

    def Hplayer_update(self) -> None:
        # check if pause
        if self.pause == 1:
            self.timer.stop()
            return

        self.maze_widget_window.update()
        # Current individual is alive
        if self.maze.is_alive:
            self.maze.move()
        # Current individual is dead         
        else:
            self.maze = self.maze_widget_window.new_game()

    def Cplayer_update(self) -> None:
        # check if pause
        if self.pause == 1:
            self.timer.stop()
            return

        self.maze_widget_window.update()
        # Current individual is alive
        if self.maze.is_alive:
            self.maze.move()
        # Current individual is dead         
        else:
            # Calculate fitness of current individual
            self.maze.calculate_fitness()
            fitness = self.maze.fitness
            print(self._current_individual, fitness, ' ', self.maze.score)

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.save_condiction = 1
                self.best_indiviaul = self._current_individual
                print('new best')

            self._current_individual += 1
            # Next generation
            if (self.current_generation > 0 and self._current_individual == self._next_gen_size) or\
                (self.current_generation == 0 and self._current_individual == settings['num_parents']):
                print(self.settings)
                print('======================= Gneration {} ======================='.format(self.current_generation))
                print('----Max fitness:', self.population.fittest_individual.fitness)
                print('----Best Score:', self.population.fittest_individual.score)
                print('----Average fitness:', self.population.average_fitness)
                if self.save_condiction == 1:
                    self.save_condiction = 0
                    mlp.save_maze(self.settings['smfp'], 
                            'best_maze_at{}'.format(self.current_generation), 
                            self.population.individuals[self.best_indiviaul], self.settings)
                self.next_generation()
                print('check end')

            self.maze = self.population.individuals[self._current_individual]
            self.maze_widget_window.maze = self.maze

    def next_generation(self):
        self.current_generation += 1
        self._current_individual = 0

        # Calculate fitness of individuals
        for individual in self.population.individuals:
            individual.calculate_fitness()
        
        self.population.individuals = elitism_selection(self.population, self.settings['num_parents'])
        
        random.shuffle(self.population.individuals)
        next_pop: List[mlp.maze] = []

        # parents + offspring selection type ('plus')
        if self.settings['selection_type'].lower() == 'plus':
            # Decrement lifespan
            for individual in self.population.individuals:
                individual.lifespan -= 1

            for individual in self.population.individuals:
                params = individual.network.params
                board_size = individual.board_size
                hidden_layer_architecture = individual.hidden_layer_architecture
                hidden_activation = individual.hidden_activation
                output_activation = individual.output_activation
                lifespan = individual.lifespan

                # If the individual is still alive, they survive
                if lifespan > 0:
                    s = mlp.maze(board_size, self.maps, player=self.player, 
                            chromosome=params, hidden_layer_architecture=hidden_layer_architecture,
                            hidden_activation=hidden_activation, output_activation=output_activation,
                            lifespan=lifespan)#,
                    next_pop.append(s)

        while len(next_pop) < self._next_gen_size:
            p1, p2 = roulette_wheel_selection(self.population, 2)

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            # Each W_l and b_l are treated as their own chromosome.
            # Because of this I need to perform crossover/mutation on each chromosome between parents
            for l in range(1, L):
                p1_W_l = p1.network.params['W' + str(l)]
                p2_W_l = p2.network.params['W' + str(l)]  
                p1_b_l = p1.network.params['b' + str(l)]
                p2_b_l = p2.network.params['b' + str(l)]

                # Crossover
                # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # Mutation
                # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
                self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                # Assign children from crossover/mutation
                c1_params['W' + str(l)] = c1_W_l
                c2_params['W' + str(l)] = c2_W_l
                c1_params['b' + str(l)] = c1_b_l
                c2_params['b' + str(l)] = c2_b_l

                # Clip to [-1, 1]
                np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
                np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
                np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
                np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])

            # Create children from chromosomes generated above
            c1 = mlp.maze(p1.board_size, maps=self.maps, player=self.player,
                        chromosome=c1_params, hidden_layer_architecture=p1.hidden_layer_architecture,
                        hidden_activation=p1.hidden_activation, output_activation=p1.output_activation,
                        lifespan=self.settings['lifespan'])
            c2 = mlp.maze(p2.board_size, maps=self.maps, player=self.player, 
                        chromosome=c2_params, hidden_layer_architecture=p2.hidden_layer_architecture,
                        hidden_activation=p2.hidden_activation, output_activation=p2.output_activation,
                        lifespan=self.settings['lifespan'])

            # Add children to the next generation
            next_pop.extend([c1, c2])
        
        # Set the next generation
        random.shuffle(next_pop)
        self.population.individuals = next_pop

    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rand_crossover = random.random()
        crossover_bucket = np.digitize(rand_crossover, self._crossover_bins)
        child1_weights, child2_weights = None, None
        child1_bias, child2_bias = None, None

        # SBX
        if crossover_bucket == 0:
            child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, self._SBX_eta)
            child1_bias, child2_bias =  SBX(parent1_bias, parent2_bias, self._SBX_eta)

        # Single point binary crossover (SPBX)
        elif crossover_bucket == 1:
            child1_weights, child2_weights = single_point_binary_crossover(parent1_weights, parent2_weights, major=self._SPBX_type)
            child1_bias, child2_bias =  single_point_binary_crossover(parent1_bias, parent2_bias, major=self._SPBX_type)
        
        else:
            raise Exception('Unable to determine valid crossover based off probabilities')

        return child1_weights, child2_weights, child1_bias, child2_bias

    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
        scale = .2
        rand_mutation = random.random()
        mutation_bucket = np.digitize(rand_mutation, self._mutation_bins)

        mutation_rate = self._mutation_rate
        if self.settings['mutation_rate_type'].lower() == 'decaying':
            mutation_rate = mutation_rate / sqrt(self.current_generation + 1)

        # Gaussian
        if mutation_bucket == 0:
            # Mutate weights
            gaussian_mutation(child1_weights, mutation_rate, scale=scale)
            gaussian_mutation(child2_weights, mutation_rate, scale=scale)

            # Mutate bias
            gaussian_mutation(child1_bias, mutation_rate, scale=scale)
            gaussian_mutation(child2_bias, mutation_rate, scale=scale)
        
        # Uniform random
        elif mutation_bucket == 1:
            # Mutate weights
            random_uniform_mutation(child1_weights, mutation_rate, -1, 1)
            random_uniform_mutation(child2_weights, mutation_rate, -1, 1)

            # Mutate bias
            random_uniform_mutation(child1_bias, mutation_rate, -1, 1)
            random_uniform_mutation(child2_bias, mutation_rate, -1, 1)

        else:
            raise Exception('Unable to determine valid mutation based off probabilities.')
    
    def startEvt(self):
        if self.pause == 1:
            self.pause = 0
            self.timer.start(1000./self.fps)
            if self.maze_widget_window.change_map == 1:
                self.maze_widget_window.change_map = 0
                next_id = next_file_index(self.maps_jump_id, self.maps_max_id)
                save_points(self.maze.walls, self.map_fpath, f'map{next_id}')
    
    def pauseEvt(self):
        self.pause = 1
        self.statusbar.showMessage('Pause')
    
    def setFPSEvt(self):
        self.pauseEvt()
        self.statusbar.showMessage('setFPSEvt')
        fps, ok = QtWidgets.QInputDialog.getText(self, 'setting', 'fps:')
        if ok:
            self.fps = float(fps)
    
    def newMapEvt(self):
        self.pauseEvt()
        self.statusbar.showMessage('newMap')
        self.maze_widget_window.change_map = 1

    def get_file_index_list(self, file_names):
        regrex = os.path.join(self.map_fpath, 'map([0-9]+)')
        # avoid bad escape in re
        regrex = regrex.replace('\\', '\\\\')
        regrex = regrex.replace('.', '\\.')
        index_list = []
        for fn in file_names:
            z = re.search(regrex, fn)
            try:
                index_list.append(int(z.group(1)))
            except:
                print("Error file name:")
                print(fn)
        return index_list

class MazeWidget(QtWidgets.QWidget):
    msg2Statusbar = QtCore.pyqtSignal(str)
    def __init__(self, parent, board_size=(11, 11), maze=None, player='human'):
        super().__init__(parent)

        self.board_size = board_size
        self.player = player
        self.change_map = 0             # For creating new map
        if maze:
            self.maze = maze
        self.setFocus()

        self.draw_vision = True
        self.show()

    def new_game(self) -> mlp.maze:
        self.maze = mlp.maze(self.board_size, self.maze.maps, self.player)
        return self.maze

    def update(self):
        if self.maze.is_alive:
            self.maze.update()
            self.repaint()
        else:
            # dead
            pass
        self.show_score()

    def draw_border(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(Qt.black))
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        painter.setPen(QtCore.Qt.black)
        painter.drawLine(0, 0, width, 0)
        painter.drawLine(width, 0, width, height)
        painter.drawLine(0, height, width, height)
        painter.drawLine(0, 0, 0, height)

    def draw_maze(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(0, 0, 0))
        # painter.setPen(QtGui.QPen(Qt.black))
        painter.setPen(pen)
        brush = QtGui.QBrush()
        brush.setColor(Qt.red)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(198, 5, 20)))
        # painter.setBrush(brush)

        # draw adventure
        adv_image_path = r"./warrior.png"
        adv_pixmap = QtGui.QPixmap(adv_image_path)
        adv_pixmap = adv_pixmap.scaled(QtCore.QSize(SQUARE_SIZE[0], SQUARE_SIZE[1]))

        r = QtCore.QRect(self.maze.adv_pos.x * SQUARE_SIZE[0],      # Upper left x-coord
                         self.maze.adv_pos.y * SQUARE_SIZE[1],      # Upper left y-coord
                         SQUARE_SIZE[0],                            # Width
                         SQUARE_SIZE[1])                            # Height
        painter.drawPixmap(r, adv_pixmap)

        # draw wall
        wall_image_path = r"./wall.png"
        wall_pixmap = QtGui.QPixmap(wall_image_path)
        wall_pixmap = wall_pixmap.scaled(QtCore.QSize(SQUARE_SIZE[0], SQUARE_SIZE[1]))
        for wall in self.maze.walls:
            r = QtCore.QRect(wall.x * SQUARE_SIZE[0],      # Upper left x-coord
                             wall.y * SQUARE_SIZE[1],      # Upper left y-coord
                             SQUARE_SIZE[0],                            # Width
                             SQUARE_SIZE[1])                            # Height
            painter.drawPixmap(r, wall_pixmap)


    def draw_treasure(self, painter: QtGui.QPainter) -> None:
        treas_pos = self.maze.treas_pos
        if treas_pos:
            painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
            painter.setPen(QtGui.QPen(Qt.black))
            painter.setBrush(QtGui.QBrush(Qt.green))

            r = QtCore.QRect(treas_pos.x * SQUARE_SIZE[0],
                             treas_pos.y * SQUARE_SIZE[1],
                             SQUARE_SIZE[0],
                             SQUARE_SIZE[1])

            image_path = r"./treasure.png"
            pixmap = QtGui.QPixmap(image_path)
            pixmap = pixmap.scaled(r.size())
            painter.drawPixmap(r, pixmap)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.draw_border(painter)
        self.draw_treasure(painter)
        self.draw_maze(painter)
        
        painter.end()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key_press = event.key()
        if self.player == 'human':
            if key_press == Qt.Key_Up:
                self.maze.direction = 'u'
            elif key_press == Qt.Key_Down:
                self.maze.direction = 'd'
            elif key_press == Qt.Key_Right:
                self.maze.direction = 'r'
            elif key_press == Qt.Key_Left:
                self.maze.direction = 'l'

    def _calc_distance(self, x1, x2, y1, y2) -> float:
        diff_x = float(abs(x2-x1))
        diff_y = float(abs(y2-y1))
        dist = ((diff_x * diff_x) + (diff_y * diff_y)) ** 0.5
        return dist
    
    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.change_map == 1:
            click_pos = event.pos()
            x = click_pos.x() // SQUARE_SIZE[0]
            y = click_pos.y() // SQUARE_SIZE[1]
            
            pos_type = self.maze.check_pos_type((x,y)).lower()
            if pos_type == 'none':
                self.maze.add_wall((x,y))
                self.repaint()
            elif pos_type == 'wall':
                self.maze.delete_wall((x,y))
                self.repaint()                

    def show_score(self):
        if self.maze.is_alive:
            self.msg2Statusbar.emit(str(self.maze.score))
        else:
            self.msg2Statusbar.emit('GameOver')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(settings, '.', show=settings['show'], fps=settings['fps'])
    sys.exit(app.exec_())