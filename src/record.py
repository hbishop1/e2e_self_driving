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

        self.ts = message_filters.ApproximateTimeSynchronizer([self.left_image_sub, self.right_image_sub, self.control_sub],10,0.1)
        self.ts.registerCallback(self.callback)
        print("initialised correctly")

    def callback(self, left_image, right_image, controls):

        if controls.buttons[7] == 1 and not self.recording:

            print("Started recording rosbag")
            
            time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            self.recording = True
            self.bag = rosbag.Bag(self.output_location + 'train_data_' + time + '.bag', 'w')   

        elif controls.buttons[6] == 1 and self.recording:

            self.recording = False
            self.bag.close()

            print("Recorded rosbag of length: " + str(self.bag.get_end_time() - self.bag.get_start_time()))

        elif self.recording:

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



        

if __name__ == '__main__':
    rospy.init_node('recorder', anonymous=True)
    recorder = Recorder()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        recorder.bag.close()
        print("Shutting down")
    rospy.spin()

