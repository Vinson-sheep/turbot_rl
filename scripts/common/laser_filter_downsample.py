#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import rospy

from sensor_msgs.msg import LaserScan

input_num = 360
output_num = 36

class laser_filter_downsample:
    def __init__(self):

        self.laserSub = rospy.Subscriber("/scan", LaserScan, self.laserCB)
        self.laserPub = rospy.Publisher("/scan_downsampled", LaserScan, queue_size=1)


    def laserCB(self, msg):
        pub_msg = LaserScan()
        pub_msg.header.seq = msg.header.seq
        pub_msg.header.stamp = rospy.Time.now()
        pub_msg.header.frame_id = msg.header.frame_id
        pub_msg.angle_max = msg.angle_max
        pub_msg.angle_min = msg.angle_min
        pub_msg.angle_increment = msg.angle_max/float(output_num)
        pub_msg.time_increment = msg.time_increment
        pub_msg.scan_time = msg.scan_time
        pub_msg.range_max = 3.5
        pub_msg.range_min = 0.14

        for i in range(output_num):
            index = int(i*(input_num/output_num) + input_num/output_num/2)
            pub_msg.ranges.append(msg.ranges[index])

        self.laserPub.publish(pub_msg)




if __name__ == '__main__':
    rospy.init_node("laser_filter_downsample")
    
    instance = laser_filter_downsample()

    rospy.spin()