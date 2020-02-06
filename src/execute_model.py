import rospy
import numpy as np
import cv2
from e2e_self_driving.msg import TrainInstance
from e2e_self_driving.src import model.My_PilotNet
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from std_msgs.msg import String

class Model_run(object):

    def __init__(self):

        self.publish = rospy.Publisher("left_image", Image, queue_size=10)
        self.subscriber = rospy.Subscriber("/train_data", TrainInstance, self.callback)


    def callback(self,instance):

        for i in range(2):

            image = cv2.imdecode(np.fromstring([instance.left_image_data,instance.right_image_data][i],np.uint8), cv2.IMREAD_COLOR)
            msg = self.bridge.cv2_to_imgmsg(image,"passthrough")
            [self.left_image_pub,self.right_image_pub][i].publish(msg)

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

        self.vector_pub.publish(marker)



if __name__ == '__main__':

    rospy.init_node('model', anonymous=True)
    visualiser = Visualiser()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

