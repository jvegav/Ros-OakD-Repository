#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoSubscriber(Node):
    
    def __init__(self):
        super().__init__("video_subscriber_node")
        self.get_logger().info("Nodo de suscripción de video iniciado")

        # Suscriptor al tópico de video
        self.subscription = self.create_subscription(
            Image,
            "stereo_video",
            self.video_callback,
            10)
        self.bridge = CvBridge()

    def video_callback(self, msg):
        try:
            # Convierte el mensaje de ROS Image a imagen de OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, "rgb8")

            # Muestra la imagen como video
            cv2.imshow("Video en Vivo", frame)
            cv2.waitKey(1)  # Permite una actualización de ventana continua para mostrar el flujo de video

        except Exception as e:
            self.get_logger().error(f"Error al procesar el frame del video: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VideoSubscriber()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
