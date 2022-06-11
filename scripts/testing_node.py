#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import rospy
from common.game import Game
from turbot_rl.msg import State
import numpy as np
import os
import pickle
# from tensorboardX import SummaryWriter

import DDPG
import TD3
import SAC


load_progress = False

action_dim = 2

policy = "TD3" # DDPG or TD3 or SAC
filter = "NONE" # NONE / MAF / FOLF
game_name = "TEST1" # TEST1 / TEST2 / TEST3 / TEST4

# Median Average Filter
window_size = 4
# First-Order Lag Filter
action_discount = 0.5

max_episode = 200
max_step_size = 300

# variable
episode_begin = 0
success_num = 0
crash_num = 0
agent = None

# file url
url = os.path.dirname(os.path.realpath(__file__)) + '/data/'
# writer = SummaryWriter(url + '../../log')

step_time = 0.5
episode_times = np.array([])

# initialize agent
kwargs = {
    'load_buffer_flag': False,
    'load_actor_flag': True,
    'load_critic_flag': False,
    'load_log_alpha_flag': False,
    'load_optim_flag': False,
    'fix_actor_flag': True,
    'use_priority': False
}

if (policy == "TD3"):
    agent = TD3.TD3(**kwargs)
if (policy == "DDPG"):
    agent = DDPG.DDPG(**kwargs)
if (policy == "SAC"):
    agent = SAC.SAC(**kwargs)

def save(episode, success_num, crash_num):
    save_file = open(url + 'temp_test.bin',"wb")
    pickle.dump(episode,save_file)
    pickle.dump(success_num,save_file)
    pickle.dump(crash_num, save_file)
    save_file.close()
    np.save(url + "test_episode_times.npy", episode_times)

def load():
    episode_times = np.load(url + "test_episode_times.npy")
    load_file = open(url + 'temp_test.bin',"rb")
    episode=pickle.load(load_file)
    success_num=pickle.load(load_file)
    crash_num = pickle.load(load_file)
    print("Restore episode = %d, success num = %d, crash num = %d." % (episode+1, success_num, crash_num))
    return episode, success_num, crash_num

if __name__ == '__main__':

    # initialize ros
    rospy.init_node("testing_node")

    # raw data
    # rawCmdPub = rospy.Publisher("raw_cmd", PositionTarget, queue_size=1)
    # modCmdPub = rospy.Publisher("mod_cmd", PositionTarget, queue_size=1)
    statePub = rospy.Publisher("state", State, queue_size=1)

    # wait for world building
    rospy.sleep(rospy.Duration(3))
    
    # initialize environment
    env = Game("turtlebot3_burger", game_name)

    # load data if true
    if load_progress: 
        episode_begin, success_num, crash_num = load()
        episode_begin += 1

    # start to test
    for episode in range(episode_begin, max_episode):

        print("=====================================")
        print("=========== Episode %d ===============" % (episode))
        print("=====================================")

        s0 = env.reset(episode)

        # filter initialize
        if (filter == "NONE"):
            pass
        if (filter == "MAF"):
            window_vx = [0.0]*window_size
            window_vyaw = [0.0]*window_size
        if (filter == "FOLF"):
            momentum = [0.0]*action_dim
        episode_begin_time = rospy.Time.now()
        for step in range(max_step_size):

            a0 = agent.act(s0)

            # DEBUG
            pt = [0, 0]
            pt[0] = (a0[0]+1)/10.0
            pt[1] = a0[1]

            if (filter == "NONE"):
                # DEBUG
                pass
                
            if (filter == "MAF"):
                # update queue
                window_vx.pop(0)
                window_vx.append(a0[0])
                window_vyaw.pop(0)
                window_vyaw.append(a0[1])
                # copy data and sort
                window_vyaw_copy = window_vyaw[:]
                window_vyaw_copy.sort()
                window_vx_copy = window_vx[:]
                window_vx_copy.sort()
                # DEBUG
                pt[0] = (np.mean(window_vx_copy[1:-1])+1)/10.0
                pt[1] = np.mean(window_vyaw_copy[1:-1])

            if (filter == "FOLF"):
                momentum[0] = (1-action_discount)*momentum[0] + action_discount*a0[0]
                momentum[1] = (1-action_discount)*momentum[1] + action_discount*a0[1]
                # DEBUG
                pt[0] = (momentum[0]+1)/10.0
                pt[1] = momentum[1]

            # # DEBUG
            # modCmdPub.publish(pt)

            # step
            s1, _, done = env.step(step_time, pt[0], 0, pt[1])

            # DEBUG
            msg = State()
            msg.header.stamp = rospy.Time.now()
            msg.data = s0
            statePub.publish(msg)

            s0 = s1

            # check result
            crash_indicator, _, _ = env.is_crashed()
            arrive_indicator = env.is_arrived()
           
            if done == True:
                if crash_indicator == True:
                    crash_num += 1
                    print("Crashed!")
                elif arrive_indicator == True:
                    success_num += 1
                    print("Arrived")
                break

            if rospy.is_shutdown(): break

        episode_time = (rospy.Time.now() - episode_begin_time).to_sec()
           
        episode_times = np.append(episode_times, episode_time)

        print('[' + str(episode+1) + '] success_num = %d, crash_num = %d' %(success_num, crash_num))

        if rospy.is_shutdown(): break

        save(episode, success_num, crash_num)

    rospy.spin()