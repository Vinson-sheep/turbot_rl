#! /usr/bin/env python
#-*- coding: UTF-8 -*- 
from math import atan2

import rospy
import tf
# ros include
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose, Point, PoseStamped, TwistStamped, TwistWithCovarianceStamped
from gazebo_msgs.srv import SetModelState, GetModelState, GetModelStateRequest, GetModelStateResponse
from gazebo_msgs.msg import ModelState 
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from turbot_rl.msg import Reward, Acc
from nav_msgs.msg import Odometry
# from common.world import World
from world import World
import random
import numpy as np
import math
import angles

# from pykalman import KalmanFilter

class Game:

    def __init__(self, model_name, game_name):
        """
            initialize
        """

        # self.talker_node = rospy.init_node('talker', anonymous=True)
        self.model_name = model_name
        self.game_name = game_name

        self.odom = Odometry()
        self.pose = Pose()
        self.scan = LaserScan()

        self.crash_limit = 0.2

        self.start_flag = False

        self.target_x = 10
        self.target_y = 10

        self.height = 0.0

        self.step_count = 0

        self.rate = rospy.Rate(10)

        ## env.space
        self.state_num = 40
        self.action_num = 2
        self.observation_space = np.empty(self.state_num)
        self.action_space = np.empty(self.action_num)
        ##

        # initialize world
        if (game_name == "EMPTY"):
            self.safe_space = [[0, 0], [8, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 8
            self.wall_rate = 0
            self.cylinder_num = [0, 0]
        
        if (game_name == "TRAIN"):
            self.safe_space = [[0, 0], [8, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 8
            self.wall_rate = 0.8
            self.cylinder_num = [70, 100]

        if (game_name == "TEST1"):
            self.safe_space = [[0, 0], [8, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 8
            self.wall_rate = 0.5
            self.cylinder_num = [5, 50]

        if (game_name == "TEST2"):
            self.safe_space = [[0, 0], [8, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 8
            self.wall_rate = 0.8
            self.cylinder_num = [50, 100]

        if (game_name == "TEST3"):
            self.safe_space = [[0, 0], [8, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 8
            self.wall_rate = 0.8
            self.cylinder_num = [100, 150]

        if (game_name == "TEST4"):
            self.safe_space = [[0, 0], [8, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 8
            self.wall_rate = 0.8
            self.cylinder_num = [150, 200]

        if (game_name == "CUSTOMIZED"):
            self.safe_space = [[0, 0], [8, 0]]
            self.safe_radius = [1.0, 0.8]
            self.target_distance = 8
            self.wall_rate = 0.8
            self.cylinder_num = [150, 200]            

        self.world = World(self.safe_space, self.safe_radius, wall_rate=self.wall_rate, cylinder_num=self.cylinder_num)

        # subscriber
        self.scanSub = rospy.Subscriber("/scan_downsampled", LaserScan, self._scanCB)
        self.odomSub = rospy.Subscriber("/odom", Odometry, self._odomCB)
        
        # publisher
        self.cmdPub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.rewardPub = rospy.Publisher("reward", Reward, queue_size=1)
        # self.accPub = rospy.Publisher("accel_body", Acc, queue_size=1)
        self.setModelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)

        # service client
        self.modelStateClient = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        self.tf_listener = tf.TransformListener()
        self.odom_frame = "odom"
        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Can not find transform")
                rospy.signal_shutdown("tf_Exception")
        (self.position, self.rotation) = self.get_odom()

    def reset(self, trial):
        """
            reset uav and target
        """
        
        if trial % 1 == 0:            
            if self.start_flag:
                self.world.clear()
                rospy.loginfo("clear world.")
            else:
                self.start_flag = True

            self.world.reset()
            rospy.loginfo("reset_world.")

        # stop
        self.target_x = self.target_distance
        self.target_y = 0

        self.world.set_target(self.target_x, self.target_y)

        rospy.loginfo("initialize target position.")

        begin_time = rospy.Time.now()
        while (rospy.Time.now() - begin_time) < rospy.Duration(1):
            msg = Twist()
            self.cmdPub.publish(msg)
            self.rate.sleep()

        # reset turbot position
        msg = ModelState()
        msg.model_name = self.model_name

        home_yaw = random.uniform(-math.pi, math.pi)
        qtr =  quaternion_from_euler(0, 0, home_yaw)

        msg.pose.orientation.x = qtr[0]
        msg.pose.orientation.y = qtr[1]
        msg.pose.orientation.z = qtr[2]
        msg.pose.orientation.w = qtr[3]

        self.setModelStatePub.publish(msg)

        rospy.loginfo("initialize turbot position.")

        # clear world
        # if self.start_flag:
        #     self.world.clear()
        #     self.start_flag = False
        #     rospy.loginfo("clear world.")
        

        # initialize target point
       
        # reset world
       
        # print("step_count ", self.step_count)
        # if self.step_count % 10 == 0: 
        self.step_count = 0
        # self.start_flag = True

        return self.cur_state()


    def is_crashed(self):
        """
            determine if the uav is crashed.
            return:
                - True if it was crashed.
                - crash reward
                - relative laser index (to check the direction)
        """
        self.laser_crashed_flag = False
        self.laser_crashed_reward = 0
        self.crash_index = -1

        for i in range(len(self.scan.ranges)):
            if self.scan.ranges[i] < 3*self.crash_limit:
                self.laser_crashed_reward = min(-10, self.laser_crashed_reward)
            if self.scan.ranges[i] < 2*self.crash_limit:
                self.laser_crashed_reward = min(-25.0, self.laser_crashed_reward)
            if self.scan.ranges[i] < self.crash_limit:
                self.laser_crashed_reward = -100.0
                self.laser_crashed_flag = True
                self.crash_index = i
                break

        return self.laser_crashed_flag, self.laser_crashed_reward, self.crash_index

    def step(self, time_step=0.1, vx=0.1, vy=0.1, yaw_rate=0.1, step = 0):
        """
            game step
        """
        self.step_count += 1
        time_begin = rospy.Time.now()

        self.hold_able = False
        (self.position, self.rotation) = self.get_odom()
        # record last x and y
        # last_pos_x_uav = self.pose.position.x
        # last_pos_y_uav = self.pose.position.y
        last_pos_x_uav = self.position.x
        last_pos_y_uav = self.position.y
        # print("last_x", last_pos_x_uav ,"last_y" ,last_pos_y_uav )
        last_distance = math.sqrt((self.target_x - last_pos_x_uav)**2 + (self.target_y - last_pos_y_uav)**2)

        # send control command
        msg = Twist()
        msg.linear.x = vx
        msg.angular.z = yaw_rate
        self.cmdPub.publish(msg)
        rospy.sleep(rospy.Duration(time_step))
        # record current x and y
        (self.position, self.rotation) = self.get_odom()

        # cur_pos_x_uav = self.pose.position.x
        # cur_pos_y_uav = self.pose.position.y
        cur_pos_x_uav = self.position.x
        cur_pos_y_uav = self.position.y
        
        # print("now_x", cur_pos_x_uav ,"now_y" ,cur_pos_y_uav )
        cur_distance = math.sqrt((self.target_x - cur_pos_x_uav)**2 + (self.target_y - cur_pos_y_uav)**2)
        
        # distance reward
        distance_reward = (last_distance - cur_distance)*(5/time_step)*3.0*7
        
        self.done = False

        # arrive reward
        self.arrive_reward = 0
        if cur_distance < 0.3:
            self.arrive_reward = 100
            self.done = True

        # crash reward
        crash_indicator, crash_reward, _ = self.is_crashed()
        if crash_indicator == True:
            self.done = True

        # laser reward
        state = np.array(self.scan.ranges) / float(self.scan.range_max)
        # print("state is ", state)
        for j in range(36):
                if state[j] > 1:
                    state[j] = 1.
        laser_reward = -1.0*np.abs(np.sum((state - 1)**4))

        # linear punish reward (abandan)
        self.linear_punish_reward_x = 0
        # if self.body_v.twist.linear.x < 0.1:
        #     self.linear_punish_reward_x = -1

        # angular punish reward
        self.angular_punish_reward = 0
        if abs(self.odom.twist.twist.angular.z) > 0.4:
            self.angular_punish_reward = -1
        elif abs(self.odom.twist.twist.angular.z) > 0.7:
            self.angular_punish_reward = -2

        # step punish reward
        self.step_punish_reward = -self.step_count * 0.08
        # self.step_punish_reward = 0


        # acc punish
        self.acc_x_punish_reward = 0
        self.acc_yaw_punish_reward = 0
        # self.acc_x_punish_reward = -4.0*abs(self.acc_x)
        # self.acc_yaw_punish_reward = -2.0*abs(self.acc_yaw)

        # right turing reward
        right_turning_reward = 0    # to break balance of turning
        # if self.odom.twist.twist.angular.z < 0:
            # right_turning_reward = 0.3*abs(self.odom.twist.twist.angular.z)
        # print("distance_reward ", distance_reward)
        # print("arrive_reward ", self.arrive_reward)
        # print("crash_reward ", crash_reward )
        # print("laser_reward", laser_reward)
        # print("angular_punish_reward", self.angular_punish_reward)
        # print("step_punish_reward", self.step_punish_reward)
        total_reward = distance_reward \
                        + self.arrive_reward \
                        + crash_reward \
                        + laser_reward \
                        + self.angular_punish_reward \
                        + self.step_punish_reward \
                        + self.acc_x_punish_reward \
                        + self.acc_yaw_punish_reward \
                        + right_turning_reward \
 
        msg = Reward()
        msg.header.stamp = rospy.Time.now()
        msg.distance_reward = distance_reward
        msg.arrive_reward = self.arrive_reward
        msg.crash_reward = crash_reward
        msg.laser_reward = laser_reward
        msg.linear_punish_reward = self.linear_punish_reward_x
        msg.angular_punish_reward = self.angular_punish_reward
        msg.step_punish_reward = self.step_punish_reward
        msg.acc_x_punish_reward = self.acc_x_punish_reward
        msg.acc_yaw_punish_reward = self.acc_yaw_punish_reward
        msg.right_turning_reward = right_turning_reward
        msg.total_reward = total_reward        
        self.rewardPub.publish(msg)

        # out of limit
        if self.position.x < -10 or self.position.x > 12:
            self.done = True
        if self.position.y < -10 or self.position.y > 10:
            self.done = True

        # sleep
        

        
        return self.cur_state(), total_reward/20.0, self.done


    def cur_state(self):
        """
            36 laser
            1  vx
            1  yaw_rate
            1  distance
            1  angle_diff
        """
        # ranges msg
        state = [ (i - self.scan.range_max/2)/(self.scan.range_max/2) for i in self.scan.ranges]
        for j in range(36):
                if state[j] > 1:
                    state[j] = 1.
        # pose msg
        (self.position, self.rotation) = self.get_odom()
        state.append(self.odom.twist.twist.linear.x/0.2)
        state.append(self.odom.twist.twist.angular.z)

        # relative distance and normalize
        # distance_uav_target =  math.sqrt((self.target_x - self.pose.position.x)**2 + (self.target_y - self.pose.position.y)**2)/10
        distance_uav_target =  math.sqrt((self.target_x - self.position.x)**2 + (self.target_y - self.position.y)**2)/3.0

        angle_uav_targer = atan2(self.target_y - self.position.y, self.target_x - self.position.x)
        # (_, _, angle_uav) = euler_from_quaternion([
        #                                         self.pose.orientation.x,
        #                                         self.pose.orientation.y,
        #                                         self.pose.orientation.z,
        #                                         self.pose.orientation.w
        #                                         ])
        angle_uav = self.rotation
        angle_diff = angles.shortest_angular_distance(angle_uav, angle_uav_targer)/math.pi
        state.append(distance_uav_target)
        state.append(angle_diff)

        return np.array(state)

    # tool funciton

    def _scanCB(self, msg):
        self.scan = msg


    def _odomCB(self, msg):
        self.odom = msg


    def _is_arrived(self, x, y, z, limit=0.1):
        # return (self.pose.position.x - x)**2 < limit and \
        #         (self.pose.position.y - y)**2 < limit and \
        #             (self.pose.position.z - z)**2 < limit
        (self.position, self.rotation) = self.get_odom()

        return (self.position.x - x)**2 < limit and \
                (self.position.y - y)**2 < limit and \
                    (self.position.z - z)**2 < limit


    def is_arrived(self):
        (self.position, self.rotation) = self.get_odom()
        cur_pos_x_uav = self.position.x
        cur_pos_y_uav = self.position.y
        cur_distance = math.sqrt((self.target_x - cur_pos_x_uav)**2 + (self.target_y - cur_pos_y_uav)**2)
        if cur_distance < 0.3:
            return True
        return False

    def get_odom(self):
        try:
            (trans, angular) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(angular)
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("tf Exception Error")
            return

        return (Point(*trans), rotation[2])

if __name__ == '__main__':

    rospy.init_node("test")

    game = Game("turtlebot3_burger", "empty_3m")

    game.reset()

    rospy.sleep(rospy.Duration(3.0))

    begin_time = rospy.Time.now()

    while (rospy.Time.now() - begin_time) < rospy.Duration(3):
        game.step(0.1, 0.1, 0, 0.2)

    game.reset()

    rospy.sleep(rospy.Duration(3.0))

    begin_time = rospy.Time.now()

    while (rospy.Time.now() - begin_time) < rospy.Duration(3):
        game.step(0.1, 0.1, 0, 0.2)
