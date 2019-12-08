#!/usr/bin/env python2

import rospy
import rosbag
from datetime import datetime
import cv2
import numpy as np
import message_filters
from e2e_self_driving.msg import TrainInstance
from sensor_msgs.msg import CompressedImage, Joy


class Recorder(object):
    def __init__(self):
        
        self.frame = 1
        self.recording = False

        self.bag = None


        self.output_location = rospy.get_param("~output_dir")

        self.left_image_sub = message_filters.Subscriber('/camera/left_image', CompressedImage)
        self.right_image_sub = message_filters.Subscriber('/camera/right_image', CompressedImage)
        self.control_sub = message_filters.Subscriber('/joystick', Joy)

        self.ts = message_filters.TimeSynchronizer([self.left_image_sub, self.right_image_sub, self.control_sub], 10)
        self.ts.registerCallback(self.callback)


    def callback(self, left_image, right_image, controls):

        self.recording = self.recording and controls.buttons[6] == 0

        if self.recording:

            msg = TrainInstance()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = str(self.frame)
            msg.axes = controls.axes
            msg.buttons = controls.buttons
            msg.image_format = left_image.format
            msg.left_image_data = left_image.data
            msg.right_image_data = right_image.data

            self.bag.write('train_data',msg)

            self.frame += 1

        elif controls.buttons[7] == 1:
            
            time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            self.recording = True
            self.bag = rosbag.Bag(self.output_location + 'train_data_' + time + '.bag', 'w')   


        

if __name__ == '__main__':
    rospy.init_node('recorder', anonymous=True)
    new_recorder = Recorder()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    rospy.spin()

