# Continous Control Project- DRL NanoDegree

This project is part of Udacity's Deep Reinforcement Learning NanoDegree. Its main objective is to maintain the position of a double-jointed arm at the target location for as many time steps as possible using Unity's ML-agents reacher environment.

![env](reacher.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Installation

1. Clone the repository.
```
git clone https://github.com/JawharAfli/Continous-Control-DRLND.git

cd Continous-Control-DRLND
```

2. Prepare the environment.
```
conda env create -f environment.yml

conda activate drlnd

pip install torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Download the Unity ML-Agents Reacher Environment.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

4. Include the path of the reacher environment into the configuration file `config.py`.

## Train your reacher agent

```
python reacher.py --train
```
