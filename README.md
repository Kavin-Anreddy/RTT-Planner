# RRT / RRT* Path Planning Simulator

This project is a 2D path planning simulator built with Python and Pygame. It implements both RRT and RRT* algorithms and visualizes how a path is generated in real time while avoiding moving obstacles.

The simulator supports dynamic environments, meaning the path is continuously checked and recomputed if obstacles interfere.

## Features

- RRT and RRT* (with rewiring)
- Moving obstacles with collision detection
- Real-time replanning
- KD-tree for efficient nearest-neighbor search
- Optional path smoothing (Catmull-Rom splines)
- Interactive controls and live visualization
- Better optimization will be included in the future!

## Controls

- Left click: Move start position  
- Right click: Move goal position  
- Space: Pause / resume  
- R: Reset simulation  
- S: Toggle RRT / RRT*  
- G: Toggle goal bias  
- P: Toggle path smoothing  
- F: Toggle fast mode  
- + / -: Adjust obstacle speed  
- Q / Esc: Quit  

## Installation

1. Install dependencies:

```bash
pip install pygame numpy
