# **Maze-AI**
## Game rules
1. Adventurer move along four direction \(↑, ↓, ←, →\). Adventurer move **1** step per frame. You could only change the direction adventurer move.
2. Whenever adventurer met a treasure, you get **1** point.
3. Every time you get more **3** points, the map changes randomly.
4. If adventurer rush into wall or the bound, **game over**.
5. Collect the points as much as you can. Once you get **120** points, you complete the game.
## Main purpose
Training Machine to automatically complete this game by using **Forward Network** with **Genetic Algorithm**.
## Demo
![](https://i.imgur.com/Pt9rYIO.gif)
## Simple Run
### Human playing mode
1. Open *setting.py* and set **'player'** to **'human'**.
2. Set your prefer **'fps'** and check the term **'show'** is setting with **True**.
3. Run *maze_app.py*
4. Use ↑, ↓, ←, → to control the adventurer.
```python=
settings = {
    'board_size':                  (11, 11),
    #### Simple run setting part ####
    
    # Player
    'player':                       'human', 
    # Fps
    'fps':                          20,
    # Show or not
    'show':                         True,
```
### Computer training mode
* ==*If you just want to reproduce the demo, just run the default code*==
1. Open *setting.py*.
2. First, change **'player'** to **'computer'**.
```python=
settings = {
    ...
    'player':                       'computer', 
    ...
```
3. Set your prefer **'fps'** and **'show'**.
==*Recommand setting: \(fps:200, show:True\) or \(fps:1000, show:False\)*==
```python=
settings = {
    ...
    # Fps
    'fps':                          20,
    # Show or not
    'show':                         True,

    ...
```
4. In Genetic Algorithm, we need to give the machine first generation. **'PGP'**(Parent Generate Percentage) determines the composition of the first generation. **'lmp'**(load maze file path) and **'lmn'**(load maze name) are the path and the file name of the maze which we want to add to the first generation.     
    - Set **PGP** to 1 means we totally generate the first generation by random.
    - Set **PGP** to 0 means we don't generate new maze, we just totally use the copies of the old one to be our first generation. 
    - Set **PGP** between 0 to 1 means we generate **PGP** portion at random in first generation and use the copyies of old one for others.
 
```python=
settings = {
    ...
    # Parent Generate Percentage
    'PGP':                          0, 
    # Load maze file path
    'lmp':                          r'./save_c3addDifaddPassD3_3', 
    # Load maze name
    'lmn':                          'best_maze_at592',
    ...
```
5. Last, set the save file path for maze. Whenever we find a better parameter than before, we save it to the **'smfp'**(Save maze file path).
```python=
settings = {
    ...
    # Save maze file path
    'smfp':                         r'./save_c3addDifaddPassD3_3', 
    ...
```
6. Finally, we could run *maze_app.py*.
## Additional function
### Pause
Click menu **Action** -> **Pause** or **Ctrl+P**.
### Restart
Click menu **Action** -> **Start** or **Ctrl+S**.
### Set Fps during playing
Click menu **Action** -> **Set Fps** or **Ctrl+H**.
After setting new fps, restart(**Ctrl+S**) the game.
### Create new map
==*---Could only create during playing---*==
1. Click menu **Action** -> **New Map** or **Ctrl+M**
2. Click somewhere on the map. If it is empty, it will generate a wall. If there already exist a wall, the wall would be clear.
3. At the moment you restart\(**Ctrl+S**\) the game, this new map add to the map file.
![](https://i.imgur.com/Ds8U0PE.gif)
## Network Input
### Vision part
Same as the [snakeAI](https://github.com/Chrispresso/SnakeAI), this algorithm collect 8-direction vision to network input.
In each direction, it records **how many steps the adventurer move along the direction before he rush into a wall/treasure**.
![](https://i.imgur.com/nU9WSny.jpg)
### now direction part
Same as the [snakeAI](https://github.com/Chrispresso/SnakeAI), the direction now adventurer moving is being given to the network input.
### treaure position part
Since vision part only search 8-direction, there is a situation that some time the treaure may not be found along these 8-direction.
In this Algorithm, it record treasure position by which quadrant the treasure in. For example, now the treasure is in the \(1, 1\) quadrant\(see the picture below\).
![](https://i.imgur.com/lopMvB5.jpg)
### accumulated part
For each case in game Snake, snake could just walk the longest path \(pass every node on map like hamilton cycle\) and then complete the mission. In other words, there exists a general solution to the snake game.Unfortunately, this maze game has random map. Unless each map overlap with others, there doesn't exist general solution. We need to give the network some information from past.
Therefore, I decided to add this part to the network input. It comes from **two of the last time network ouput**. Just like RNN does, I expect this two output could repeatedly pass the information from past. 
## Network output
4 output for One-hot encode direction.
2 output for accumulated part.
Total 6 output.
## Something else
1. At first time, I didn't add accumulated part and do not change the map during each play. It seems train well, but the result is just for certain map only.
2. Then I decide to change the map per **30** points, but it jump into the same situation. I guess it was cause by the mechanism of calculate fitness since the fitness growth exponentially by score.
3. After the guess, I adjust the setting. Now the map changes per **3** points. Although the network could do well in every map not just certain, the maximum score could only reach **30** points.
4. After observed the moving action of the machine, I added the accumulated part to Input. Also, after traing for a while, I adjusted the calculate fitness function and use the best parameter in last training period to continual train. Finally, I got this result. 
5. Still, if we change the random seed to the best maze, the score wouldn't reach 120. However, after observe the moving action deciding by machine, I think the model is more general than the original one.
## Reference
https://github.com/Chrispresso/SnakeAI
## Icon
<div>Wall icons made by <a href="https://www.flaticon.com/authors/skyclick" title="Skyclick">Skyclick</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
<div>Warrior icons made by <a href="https://www.flaticon.com/authors/flat-icons" title="Flat Icons">Flat Icons</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
<div>Treasure icons made by <a href="https://www.flaticon.com/authors/smashicons" title="Smashicons">Smashicons</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
