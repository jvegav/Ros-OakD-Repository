#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import depthai as dai
import numpy as np
from cv_bridge import CvBridge

# Nodo ROS2 que publica el video en vivo
class MyNode(Node):

    def __init__(self):
        super().__init__("video_publisher_node")
        self.get_logger().info("Nodo de transmisión de video en vivo iniciado")

        # Publicador de imágenes
        self.publisher_ = self.create_publisher(Image, "stereo_video", 10)
        
        # Configuración de DepthAI y CvBridge
        self.pipeline = dai.Pipeline()
        self.bridge = CvBridge()

        # Configuración de las cámaras
        monoLeft = getMonoCamera(self.pipeline, isLeft=True)
        monoRight = getMonoCamera(self.pipeline, isLeft=False)
        self.stereo = getStereoPair(self.pipeline, monoLeft, monoRight)

        # Configuración de las salidas
        self.xoutDisp = self.pipeline.createXLinkOut()
        self.xoutDisp.setStreamName("disparity")
        self.xoutRectifiedLeft = self.pipeline.createXLinkOut()
        self.xoutRectifiedLeft.setStreamName("rectifiedLeft")
        self.xoutRectifiedRight = self.pipeline.createXLinkOut()
        self.xoutRectifiedRight.setStreamName("rectifiedRight")

        # Conectar salidas al pipeline
        self.stereo.disparity.link(self.xoutDisp.input)
        self.stereo.rectifiedLeft.link(self.xoutRectifiedLeft.input)
        self.stereo.rectifiedRight.link(self.xoutRectifiedRight.input)

        # Inicialización del dispositivo DepthAI
        try:
            self.device = dai.Device(self.pipeline)
            self.get_logger().info("Dispositivo DepthAI conectado correctamente")
            self.disparityQueue = self.device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
            self.rectifiedLeftQueue = self.device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
            self.rectifiedRightQueue = self.device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)
            self.disparityMultiplier = 255 / self.stereo.getMaxDisparity()

            # Timer para publicar imágenes continuamente a 30 Hz
            self.timer = self.create_timer(0.033, self.process_images)

        except Exception as e:
            self.get_logger().error(f"No se pudo conectar con el dispositivo DepthAI: {e}")
            self.destroy_node()

    def process_images(self):
        try:
            disparity = getFrame(self.disparityQueue)
            disparity = (disparity * self.disparityMultiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

            leftFrame = getFrame(self.rectifiedLeftQueue)
            rightFrame = getFrame(self.rectifiedRightQueue)

            # Convierte a RGB si es necesario
            if len(leftFrame.shape) == 2:
                leftFrame = cv2.cvtColor(leftFrame, cv2.COLOR_GRAY2RGB)
            if len(rightFrame.shape) == 2:
                rightFrame = cv2.cvtColor(rightFrame, cv2.COLOR_GRAY2RGB)
            if len(disparity.shape) == 2:
                disparity = cv2.cvtColor(disparity, cv2.COLOR_GRAY2RGB)

            # Combina las imágenes en una sola
            imOut = np.hstack((leftFrame, rightFrame, disparity))

            # Publica el video como una secuencia de imágenes
            ros_image = self.bridge.cv2_to_imgmsg(imOut, encoding="rgb8")
            self.publisher_.publish(ros_image)

        except Exception as e:
            self.get_logger().error(f"Error al procesar la imagen: {e}")
            self.destroy_node()

def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()

def getMonoCamera(pipeline, isLeft):
    mono = pipeline.createMonoCamera()
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono.setBoardSocket(dai.CameraBoardSocket.LEFT if isLeft else dai.CameraBoardSocket.RIGHT)
    return mono

def getStereoPair(pipeline, monoLeft, monoRight):
    stereo = pipeline.createStereoDepth()
    stereo.setLeftRightCheck(True)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    return stereo

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

