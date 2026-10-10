"""Driver node for a DJI Tello: publishes camera frames and applies cmd_vel."""

import threading

import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_srvs.srv import Empty
from cv_bridge import CvBridge

try:
    from djitellopy import Tello
except ImportError:
    Tello = None


class TelloControllerNode(Node):

    def __init__(self):
        super().__init__('tello_controller')

        self.declare_parameter('cmd_vel_topic', 'tello/cmd_vel')
        self.declare_parameter('image_topic', 'tello/camera/image_raw')
        self.declare_parameter('yaw_topic', 'tello/yaw')
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('video_rate_hz', 30.0)
        self.declare_parameter('speed_limit', 50)
        self.declare_parameter('cmd_timeout_sec', 0.5)
        self.declare_parameter('simulate', False)
        self.declare_parameter('enable_motors', True)

        self._cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self._image_topic = self.get_parameter('image_topic').value
        self._yaw_topic = self.get_parameter('yaw_topic').value
        control_rate_hz = self.get_parameter('control_rate_hz').value
        video_rate_hz = self.get_parameter('video_rate_hz').value
        self._speed_limit = int(self.get_parameter('speed_limit').value)
        self._cmd_timeout_sec = self.get_parameter('cmd_timeout_sec').value
        self._simulate = self.get_parameter('simulate').value
        self._enable_motors = self.get_parameter('enable_motors').value

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._last_cmd = Twist()
        self._last_cmd_time = self.get_clock().now()

        self._tello = None
        self._frame_reader = None

        if self._simulate:
            self.get_logger().warn('Running in simulate mode: no Tello connection will be made.')
        else:
            self._connect_to_tello()

        self._cmd_vel_sub = self.create_subscription(
            Twist, self._cmd_vel_topic, self._cmd_vel_callback, 10)
        self._image_pub = self.create_publisher(Image, self._image_topic, 10)
        self._yaw_pub = self.create_publisher(Float32, self._yaw_topic, 10)

        self._takeoff_srv = self.create_service(Empty, 'tello/takeoff', self._takeoff_callback)
        self._land_srv = self.create_service(Empty, 'tello/land', self._land_callback)

        self._control_timer = self.create_timer(1.0 / control_rate_hz, self._control_loop)
        if not self._simulate:
            self._video_timer = self.create_timer(1.0 / video_rate_hz, self._video_loop)

        self.get_logger().info(
            'tello_controller started (simulate=%s, enable_motors=%s)'
            % (self._simulate, self._enable_motors))
        if not self._simulate and not self._enable_motors:
            self.get_logger().warn(
                'enable_motors is false: camera streaming is active but takeoff and '
                'RC control commands will be ignored (camera-only mode).')

    def _connect_to_tello(self):
        if Tello is None:
            raise RuntimeError(
                "djitellopy is not installed. Run 'pip install djitellopy' or "
                "launch this node with the 'simulate' parameter set to true.")
        self._tello = Tello()
        self._tello.connect()
        self.get_logger().info('Connected to Tello. Battery: %d%%' % self._tello.get_battery())
        self._tello.streamon()
        self._frame_reader = self._tello.get_frame_read()

    def _cmd_vel_callback(self, msg):
        with self._lock:
            self._last_cmd = msg
            self._last_cmd_time = self.get_clock().now()

    def _takeoff_callback(self, request, response):
        if self._tello is None:
            self.get_logger().warn('Takeoff requested in simulate mode; ignoring.')
        elif not self._enable_motors:
            self.get_logger().warn(
                "Takeoff requested but 'enable_motors' is false; ignoring. "
                'Relaunch with enable_motors:=true to allow flight.')
        else:
            self.get_logger().info('Takeoff requested.')
            try:
                self._tello.takeoff()
            except Exception as exc:
                self.get_logger().error('Takeoff failed: %s' % exc)
        return response

    def _land_callback(self, request, response):
        if self._tello is None:
            self.get_logger().warn('Land requested in simulate mode; ignoring.')
        elif not self._enable_motors:
            self.get_logger().warn("Land requested but 'enable_motors' is false; ignoring.")
        else:
            self.get_logger().info('Land requested.')
            try:
                self._tello.land()
            except Exception as exc:
                self.get_logger().error('Land failed: %s' % exc)
        return response

    def _control_loop(self):
        if self._tello is None:
            return

        # Yaw comes from the Tello state stream, so it is published even with motors disabled.
        try:
            self._yaw_pub.publish(Float32(data=float(self._tello.get_yaw())))
        except Exception:
            pass

        if not self._enable_motors:
            return

        with self._lock:
            cmd = self._last_cmd
            age_sec = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9

        if age_sec > self._cmd_timeout_sec:
            left_right, forward_backward, up_down, yaw = 0, 0, 0, 0
        else:
            # ROS uses +y = left and +yaw = counter-clockwise; Tello RC uses
            # +left_right = right and +yaw = clockwise, so those two axes are negated.
            left_right = self._to_tello_speed(-cmd.linear.y)
            forward_backward = self._to_tello_speed(cmd.linear.x)
            up_down = self._to_tello_speed(cmd.linear.z)
            yaw = self._to_tello_speed(-cmd.angular.z)

        try:
            self._tello.send_rc_control(left_right, forward_backward, up_down, yaw)
        except Exception as exc:
            self.get_logger().error('send_rc_control failed: %s' % exc)

    def _to_tello_speed(self, normalized_value):
        clamped = max(-1.0, min(1.0, float(normalized_value)))
        return int(clamped * self._speed_limit)

    def _video_loop(self):
        if self._frame_reader is None:
            return
        frame = self._frame_reader.frame
        if frame is None:
            return

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        msg = self._bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'tello_camera'
        self._image_pub.publish(msg)

    def destroy_node(self):
        if self._tello is not None:
            try:
                self._tello.send_rc_control(0, 0, 0, 0)
                self._tello.streamoff()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TelloControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
