#!/usr/bin/env python2

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

def zed_camera():

    rospy.init_node("camera_info_publisher", anonymous=True)

    fps = rospy.get_param("~fps")
    device = rospy.get_param("~device")

    ns = rospy.get_namespace()
    
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FPS, fps)

    publisher_left = rospy.Publisher(ns + "left", CompressedImage, queue_size=10)
    publisher_right = rospy.Publisher(ns + "right", CompressedImage, queue_size=10)
    
    while not rospy.is_shutdown():

        _, frame = cap.read()
        left_right_image = np.split(frame, 2, axis=1)

        for i in range(2):

            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', left_right_image[i])[1]).tostring()

            [publisher_left,publisher_right][i].publish(msg)

if __name__ == "__main__":

    zed_camera()