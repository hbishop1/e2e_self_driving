#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
from e2e_self_driving.msg import TrainInstance
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from std_msgs.msg import String
from cv_bridge import CvBridge
from model import My_PilotNet


class Visualiser(object):

    def __init__(self, model_path=None):

        if not model_path is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.model = My_PilotNet().to(self.device)
            self.model.eval()
            self.model.load_state_dict(torch.load(model_path))
            self.tensor = transforms.Compose([transforms.ToTensor()])
            print(self.model)


        self.left_image_pub = rospy.Publisher("left_image", Image, queue_size=10)
        self.right_image_pub = rospy.Publisher("right_image", Image, queue_size=10)
        self.gt_vector_pub = rospy.Publisher("gt_vector", Marker, queue_size=10)
        self.model_vector_pub = rospy.Publisher("model_vector", Marker, queue_size=10)
        self.subscribrer = rospy.Subscriber("/train_data", TrainInstance, self.callback)

        self.bridge = CvBridge()

    def callback(self,instance):

        left_image = cv2.imdecode(np.fromstring(instance.left_image_data,np.uint8), cv2.IMREAD_COLOR)
        right_image = cv2.imdecode(np.fromstring(instance.right_image_data,np.uint8), cv2.IMREAD_COLOR)

        msg = self.bridge.cv2_to_imgmsg(left_image,"passthrough")
        self.left_image_pub.publish(msg)

        msg = self.bridge.cv2_to_imgmsg(right_image,"passthrough")
        self.right_image_pub.publish(msg)

        marker = Marker()
        marker.header.frame_id = "my_frame"
        marker.id = 0
        marker.type = 0
        marker.action = 0
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = instance.axes[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = instance.axes[1]
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0

        self.gt_vector_pub.publish(marker)

        if not self.model is None:
            
            left_image = Variable(self.tensor(left_image).float(),requires_grad = True).unsqueeze(0).to(self.device)
            right_image = Variable(self.tensor(right_image).float(),requires_grad = True).unsqueeze(0).to(self.device)

            outputs = self.model(left_image, right_image)

            marker = Marker()
            marker.header.frame_id = "my_frame"
            marker.id = 0
            marker.type = 0
            marker.action = 0
            marker.pose.position.x = 0
            marker.pose.position.y = 0
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = max(min(outputs[0][0],1),-1)
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(min(outputs[0][1],1),0)
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.g = 1.0

            self.model_vector_pub.publish(marker)


if __name__ == '__main__':

    path = rospy.get_param("/visualiser/model_path")

    rospy.init_node('visualiser', anonymous=True)

    if path == "None":
        visualiser = Visualiser()
    else:
        visualiser = Visualiser(path)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    