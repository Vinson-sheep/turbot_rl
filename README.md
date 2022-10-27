# DRL-based collision avoidance for turtlebot3

## Introduction
Deep reinforcement learning implementation for collision avoidance of mobile robot. In this project, DDPG, TD3 and SAC are adopted to realize short-distance navigation for turtlebot3. The user can easily train a practical path planner for real mobile robot without any prior knowledge.

## **feature**
| interface | version |
| --------- | ------- |
| ubuntu    | 18.04   |
| ros       | melodic |
| pytorch   | 1.4.0   |


## **installation**
**ROS melodic**

refer to [here](http://wiki.ros.org/Installation/Ubuntu)

**Gazebo 9**

```
sudo apt install ros-melodic-gazebo-*
```
download [models.zip](https://lark-assets-prod-aliyun.oss-cn-hangzhou.aliyuncs.com/yuque/0/2021/zip/985678/1614564736994-3ac600e4-75fc-44ef-9cf5-3f1c3f0c8679.zip?OSSAccessKeyId=LTAI4GGhPJmQ4HWCmhDAn4F5&Expires=1642594102&Signature=Gh7XDBHBCv6GiHHxj1R6Oisrxfc%3D&response-content-disposition=attachment%3Bfilename*%3DUTF-8%27%27models.zip) and unpack under `~/.gazebo/`.

**dependancy**
```
sudo apt install ros-melodic-xacro
```

**Anaconda**

refer to [here](https://www.anaconda.com/)

**virtual environment**

create env

```
conda create -n my_env python=2.7
conda activate my_env
```

update pip
```
pip install --upgrade pip
```

install dependence

```
pip install requirements.txt
```

**Building pkg**

create workspace

```
mkdir catkin_ws
cd catkin_ws
mkdir src
cd src
catkin_init_workspace
```

Download this project and put it into `src`.

```
cd ..
catkin_make
source devel/setup.bash
```

**Run Samples**

```
# open gazebo
roslaunch turbot_rl setup.launch

# start training 
rosrun turbot_rl training_node.py
```
