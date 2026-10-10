"""Follows a blue ground line from the Tello camera and reacts to AprilTag (tag36h11) markers.

Line control: the line inside the ROI is sampled in horizontal slices. The bottom slice gives
the lateral offset (corrected by strafing), the bottom-to-top direction gives the heading angle
(corrected by yawing), and the top slice is the look-ahead used to slow down before curves.
Both axes use a PD controller with a low-pass filtered derivative.

States:
  FOLLOW     PD control on the blue line; a close-enough AprilTag triggers a maneuver.
  TURN       Yaw 90 deg left/right, measured with the Tello's own yaw (tello/yaw).
  SEEK_LINE  Fly straight forward until the blue line is seen again, then FOLLOW.
  LANDED     A land tag was seen; tello/land was called and only zero commands are sent.
"""

import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_srvs.srv import Empty
from cv_bridge import CvBridge

FOLLOW = 'FOLLOW'
TURN = 'TURN'
SEEK_LINE = 'SEEK_LINE'
LANDED = 'LANDED'


def _wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def _clamp(value, limit):
    return max(-limit, min(limit, value))


class _PD:
    """PD term with a low-pass filtered derivative; reset whenever the measurement is lost."""

    def __init__(self, kp, kd, d_alpha, max_dt=0.5):
        self.kp = kp
        self.kd = kd
        self.d_alpha = d_alpha
        self.max_dt = max_dt
        self.reset()

    def reset(self):
        self._prev_error = None
        self._prev_time = None
        self.derivative = 0.0

    def update(self, error, now_sec):
        if self._prev_error is not None:
            dt = now_sec - self._prev_time
            if 0.0 < dt <= self.max_dt:
                raw = (error - self._prev_error) / dt
                self.derivative += self.d_alpha * (raw - self.derivative)
            else:
                self.derivative = 0.0
        self._prev_error = error
        self._prev_time = now_sec
        return self.kp * error + self.kd * self.derivative


class TelloLineFollowerNode(Node):

    def __init__(self):
        super().__init__('tello_line_follower')

        self.declare_parameter('image_topic', 'tello/camera/image_raw')
        self.declare_parameter('cmd_vel_topic', 'tello/cmd_vel')
        self.declare_parameter('yaw_topic', 'tello/yaw')
        self.declare_parameter('land_service', 'tello/land')
        self.declare_parameter('debug_image_topic', 'tello/line_follower/debug_image')
        self.declare_parameter('publish_debug_image', True)

        # Frames are resized to this width before processing (0 = full resolution).
        # Pixel thresholds below are always in full-resolution pixels.
        self.declare_parameter('process_width', 480)

        self.declare_parameter('hsv_lower', [90, 80, 50])
        self.declare_parameter('hsv_upper', [130, 255, 255])

        self.declare_parameter('roi_top_ratio', 0.5)
        self.declare_parameter('min_contour_area', 300.0)
        self.declare_parameter('num_slices', 5)

        # Heading (yaw) PD on the line angle, in radians.
        self.declare_parameter('kp_yaw', 1.0)
        self.declare_parameter('kd_yaw', 0.1)
        self.declare_parameter('max_angular', 1.0)
        # Lateral (strafe) PD on the near offset, normalized to [-1, 1].
        self.declare_parameter('kp_lateral', 0.6)
        self.declare_parameter('kd_lateral', 0.1)
        self.declare_parameter('max_lateral', 0.5)
        self.declare_parameter('d_filter_alpha', 0.5)

        # Forward speed, reduced by the line angle and the look-ahead offset.
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('slowdown_gain', 0.7)
        self.declare_parameter('min_speed_scale', 0.3)
        self.declare_parameter('slowdown_angle_deg', 45.0)

        self.declare_parameter('search_yaw_speed', 0.2)

        # AprilTags. Use [-1] to disable a list (ROS can't type an empty integer list).
        self.declare_parameter('left_turn_ids', [2])
        self.declare_parameter('right_turn_ids', [3])
        self.declare_parameter('land_ids', [4])
        self.declare_parameter('tag_trigger_size_px', 80.0)
        self.declare_parameter('tag_cooldown_sec', 3.0)
        self.declare_parameter('turn_angle_deg', 90.0)
        self.declare_parameter('turn_tolerance_deg', 3.0)
        self.declare_parameter('turn_yaw_speed', 0.6)
        self.declare_parameter('turn_timeout_sec', 8.0)
        self.declare_parameter('seek_speed', 0.3)
        self.declare_parameter('seek_timeout_sec', 5.0)

        image_topic = self.get_parameter('image_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        yaw_topic = self.get_parameter('yaw_topic').value
        land_service = self.get_parameter('land_service').value
        debug_image_topic = self.get_parameter('debug_image_topic').value
        self._publish_debug_image = self.get_parameter('publish_debug_image').value

        self._process_width = int(self.get_parameter('process_width').value)

        self._hsv_lower = np.array(self.get_parameter('hsv_lower').value, dtype=np.uint8)
        self._hsv_upper = np.array(self.get_parameter('hsv_upper').value, dtype=np.uint8)

        self._roi_top_ratio = float(self.get_parameter('roi_top_ratio').value)
        self._min_contour_area = float(self.get_parameter('min_contour_area').value)
        self._num_slices = max(2, int(self.get_parameter('num_slices').value))

        d_alpha = float(self.get_parameter('d_filter_alpha').value)
        self._yaw_pd = _PD(float(self.get_parameter('kp_yaw').value),
                           float(self.get_parameter('kd_yaw').value), d_alpha)
        self._lateral_pd = _PD(float(self.get_parameter('kp_lateral').value),
                               float(self.get_parameter('kd_lateral').value), d_alpha)
        self._max_angular = float(self.get_parameter('max_angular').value)
        self._max_lateral = float(self.get_parameter('max_lateral').value)

        self._linear_speed = float(self.get_parameter('linear_speed').value)
        self._slowdown_gain = float(self.get_parameter('slowdown_gain').value)
        self._min_speed_scale = float(self.get_parameter('min_speed_scale').value)
        self._slowdown_angle = math.radians(
            float(self.get_parameter('slowdown_angle_deg').value))

        self._search_yaw_speed = float(self.get_parameter('search_yaw_speed').value)

        self._left_turn_ids = set(self.get_parameter('left_turn_ids').value)
        self._right_turn_ids = set(self.get_parameter('right_turn_ids').value)
        self._land_ids = set(self.get_parameter('land_ids').value)
        self._tag_trigger_size_px = float(self.get_parameter('tag_trigger_size_px').value)
        self._tag_cooldown_sec = float(self.get_parameter('tag_cooldown_sec').value)
        self._turn_angle_deg = float(self.get_parameter('turn_angle_deg').value)
        self._turn_tolerance_deg = float(self.get_parameter('turn_tolerance_deg').value)
        self._turn_yaw_speed = float(self.get_parameter('turn_yaw_speed').value)
        self._turn_timeout_sec = float(self.get_parameter('turn_timeout_sec').value)
        self._seek_speed = float(self.get_parameter('seek_speed').value)
        self._seek_timeout_sec = float(self.get_parameter('seek_timeout_sec').value)

        self._bridge = CvBridge()
        self._last_error_sign = 1.0
        self._kernel = np.ones((3, 3), np.uint8)

        self._tag_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
            cv2.aruco.DetectorParameters())

        self._state = FOLLOW
        self._state_start = self.get_clock().now()
        self._cooldown_until = self.get_clock().now()
        self._turn_dir = 0.0
        self._turn_start_yaw = None
        self._turned_deg = 0.0
        self._yaw = None

        self._image_sub = self.create_subscription(
            Image, image_topic, self._image_callback, qos_profile_sensor_data)
        self._yaw_sub = self.create_subscription(Float32, yaw_topic, self._yaw_callback, 10)
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self._land_client = self.create_client(Empty, land_service)
        self._debug_pub = None
        if self._publish_debug_image:
            self._debug_pub = self.create_publisher(Image, debug_image_topic, 10)

        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            'tello_line_follower started. Tags: left=%s right=%s land=%s'
            % (sorted(self._left_turn_ids), sorted(self._right_turn_ids),
               sorted(self._land_ids)))

    def _on_set_parameters(self, params):
        # Gains and speeds can be tuned live (e.g. from tello_gui); everything else needs a restart.
        setters = {
            'kp_yaw': lambda v: setattr(self._yaw_pd, 'kp', v),
            'kd_yaw': lambda v: setattr(self._yaw_pd, 'kd', v),
            'kp_lateral': lambda v: setattr(self._lateral_pd, 'kp', v),
            'kd_lateral': lambda v: setattr(self._lateral_pd, 'kd', v),
            'linear_speed': lambda v: setattr(self, '_linear_speed', v),
            'seek_speed': lambda v: setattr(self, '_seek_speed', v),
            'turn_yaw_speed': lambda v: setattr(self, '_turn_yaw_speed', v),
            'left_turn_ids': lambda v: setattr(self, '_left_turn_ids', set(v)),
            'right_turn_ids': lambda v: setattr(self, '_right_turn_ids', set(v)),
            'land_ids': lambda v: setattr(self, '_land_ids', set(v)),
        }
        for param in params:
            if param.name in ('left_turn_ids', 'right_turn_ids', 'land_ids'):
                setters[param.name](list(param.value))
                self.get_logger().info('%s set to %s' % (param.name, sorted(param.value)))
            elif param.name in setters:
                setters[param.name](float(param.value))
                self.get_logger().info('%s set to %.3f' % (param.name, param.value))
        return SetParametersResult(successful=True)

    def _yaw_callback(self, msg):
        self._yaw = float(msg.data)

    def _set_state(self, state):
        if state != self._state:
            self.get_logger().info('State %s -> %s' % (self._state, state))
        self._state = state
        self._state_start = self.get_clock().now()
        self._yaw_pd.reset()
        self._lateral_pd.reset()

    def _state_elapsed(self):
        return (self.get_clock().now() - self._state_start).nanoseconds / 1e9

    def _image_callback(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error('Failed to convert image: %s' % exc)
            return

        full_width = frame.shape[1]
        if 0 < self._process_width < full_width:
            scale = self._process_width / float(full_width)
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0

        stamp = msg.header.stamp
        now_sec = stamp.sec + stamp.nanosec * 1e-9
        if now_sec == 0.0:
            now_sec = self.get_clock().now().nanoseconds * 1e-9

        height, width = frame.shape[:2]
        roi_top = int(height * self._roi_top_ratio)
        roi = frame[roi_top:height, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        min_area = self._min_contour_area * scale * scale
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        largest = max(contours, key=cv2.contourArea) if contours else None
        line = self._measure_line(largest, mask.shape) if largest is not None else None

        tags = self._detect_tags(frame, scale)
        if self._state == FOLLOW:
            self._check_tags(tags)

        twist = Twist()
        if self._state == FOLLOW:
            self._follow_line(twist, line, width, now_sec)
        elif self._state == TURN:
            self._turn(twist)
        elif self._state == SEEK_LINE:
            if line is not None:
                self._finish_maneuver()
                self._follow_line(twist, line, width, now_sec)
            elif self._state_elapsed() > self._seek_timeout_sec:
                self.get_logger().warn('Line not found after seeking forward; searching in place.')
                self._finish_maneuver()
                self._follow_line(twist, None, width, now_sec)
            else:
                twist.linear.x = self._seek_speed
        # LANDED: zero twist.

        self._cmd_pub.publish(twist)

        if self._debug_pub is not None:
            self._publish_debug(frame, mask, roi_top, contours, largest, line, tags, twist)

    def _measure_line(self, contour, mask_shape):
        """Return slice centroids (bottom first) of the followed blob, in ROI coordinates."""
        blob = np.zeros(mask_shape, np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED)

        roi_height = mask_shape[0]
        points = []
        for i in range(self._num_slices):
            y1 = roi_height - int(i * roi_height / self._num_slices)
            y0 = roi_height - int((i + 1) * roi_height / self._num_slices)
            m = cv2.moments(blob[y0:y1, :], binaryImage=True)
            if m['m00'] > 0:
                points.append((m['m10'] / m['m00'], y0 + m['m01'] / m['m00']))
        return points or None

    def _follow_line(self, twist, line, width, now_sec):
        if line is None:
            self._yaw_pd.reset()
            self._lateral_pd.reset()
            # Spin toward the side the line was last seen (+error = right = negative angular.z).
            twist.angular.z = -self._search_yaw_speed * self._last_error_sign
            return

        half_width = width / 2.0
        near_x, near_y = line[0]
        far_x, far_y = line[-1]

        # +offset / +angle = line to the right / leaning right (needs a right correction).
        near_offset = (near_x - half_width) / half_width
        far_offset = (far_x - half_width) / half_width
        angle = math.atan2(far_x - near_x, near_y - far_y) if len(line) > 1 else 0.0
        self._last_error_sign = 1.0 if far_offset >= 0 else -1.0

        # ROS: +angular.z = turn left, +linear.y = move left.
        twist.angular.z = _clamp(-self._yaw_pd.update(angle, now_sec), self._max_angular)
        twist.linear.y = _clamp(-self._lateral_pd.update(near_offset, now_sec), self._max_lateral)

        # Look-ahead: slow down when the line bends or the far part drifts off-center.
        bend = max(abs(angle) / self._slowdown_angle, abs(far_offset))
        speed_scale = max(self._min_speed_scale, 1.0 - self._slowdown_gain * bend)
        twist.linear.x = self._linear_speed * speed_scale

    def _detect_tags(self, frame, scale):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._tag_detector.detectMarkers(gray)
        tags = []
        if ids is None:
            return tags
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            pts = tag_corners.reshape(4, 2)
            side_px = float(np.mean(np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)))
            tags.append((int(tag_id), pts, side_px / scale))
        return tags

    def _check_tags(self, tags):
        if self.get_clock().now() < self._cooldown_until:
            return
        for tag_id, _, side_px in sorted(tags, key=lambda t: -t[2]):
            if side_px < self._tag_trigger_size_px:
                continue
            if tag_id in self._land_ids:
                self._land(tag_id)
                return
            if tag_id in self._left_turn_ids:
                self._start_turn(tag_id, 1.0)
                return
            if tag_id in self._right_turn_ids:
                self._start_turn(tag_id, -1.0)
                return

    def _start_turn(self, tag_id, direction):
        self.get_logger().info(
            'Tag %d: turning %s %.0f deg.'
            % (tag_id, 'left' if direction > 0 else 'right', self._turn_angle_deg))
        self._turn_dir = direction
        self._turn_start_yaw = self._yaw
        self._turned_deg = 0.0
        self._set_state(TURN)

    def _turn(self, twist):
        if self._turn_start_yaw is None:
            self._turn_start_yaw = self._yaw
        if self._turn_start_yaw is not None and self._yaw is not None:
            # Use the magnitude so it doesn't matter which way the Tello counts yaw.
            self._turned_deg = abs(_wrap_deg(self._yaw - self._turn_start_yaw))

        remaining = self._turn_angle_deg - self._turned_deg
        if remaining <= self._turn_tolerance_deg:
            self.get_logger().info('Turn done (%.0f deg); seeking line.' % self._turned_deg)
            self._set_state(SEEK_LINE)
            return
        if self._state_elapsed() > self._turn_timeout_sec:
            self.get_logger().warn(
                'Turn timed out after %.0f deg (is tello/yaw being published?); seeking line.'
                % self._turned_deg)
            self._set_state(SEEK_LINE)
            return

        # Slow down over the last 30 deg to limit overshoot.
        scale = max(0.3, min(1.0, remaining / 30.0))
        twist.angular.z = self._turn_dir * self._turn_yaw_speed * scale

    def _finish_maneuver(self):
        self._cooldown_until = self.get_clock().now() + rclpy.duration.Duration(
            seconds=self._tag_cooldown_sec)
        self._set_state(FOLLOW)

    def _land(self, tag_id):
        self.get_logger().info('Tag %d: landing.' % tag_id)
        self._set_state(LANDED)
        self._cmd_pub.publish(Twist())
        if not self._land_client.service_is_ready():
            self.get_logger().error('Land service is not available!')
            return
        self._land_client.call_async(Empty.Request())

    def _publish_debug(self, frame, mask, roi_top, contours, largest, line, tags, twist):
        debug_frame = frame.copy()
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        debug_frame[roi_top:roi_top + mask.shape[0], :] = cv2.addWeighted(
            debug_frame[roi_top:roi_top + mask.shape[0], :], 0.5, mask_colored, 0.5, 0)

        cv2.line(debug_frame, (0, roi_top), (debug_frame.shape[1], roi_top), (255, 255, 0), 1)

        # Outline every blue region; the one being followed gets a thicker outline and a box.
        offset = (0, roi_top)
        cv2.drawContours(debug_frame, contours, -1, (0, 255, 0), 2, offset=offset)
        if largest is not None:
            cv2.drawContours(debug_frame, [largest], -1, (0, 255, 255), 2, offset=offset)
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(debug_frame, (x, roi_top + y), (x + w, roi_top + y + h), (0, 0, 255), 1)

        # Slice centroids: red = near (offset), blue = far (look-ahead), white = heading.
        if line is not None:
            pts = [(int(x), int(roi_top + y)) for x, y in line]
            cv2.line(debug_frame, pts[0], pts[-1], (255, 255, 255), 2)
            for p in pts[1:-1]:
                cv2.circle(debug_frame, p, 3, (255, 255, 255), -1)
            cv2.circle(debug_frame, pts[-1], 5, (255, 0, 0), -1)
            cv2.circle(debug_frame, pts[0], 5, (0, 0, 255), -1)

        # Tags: magenta once close enough to trigger, orange while still too far.
        for tag_id, pts, side_px in tags:
            if tag_id in self._land_ids:
                action = 'LAND'
            elif tag_id in self._left_turn_ids:
                action = 'LEFT'
            elif tag_id in self._right_turn_ids:
                action = 'RIGHT'
            else:
                action = '-'
            color = (255, 0, 255) if side_px >= self._tag_trigger_size_px else (0, 165, 255)
            cv2.polylines(debug_frame, [pts.astype(np.int32)], True, color, 2)
            x, y = pts.min(axis=0).astype(int)
            cv2.putText(debug_frame, 'id %d %s %.0fpx' % (tag_id, action, side_px),
                        (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        status = self._state
        if self._state == TURN:
            status += ' %.0f/%.0f deg' % (self._turned_deg, self._turn_angle_deg)
        if self._yaw is not None:
            status += '  yaw %.0f' % self._yaw
        cmd = 'x %+.2f  y %+.2f  yaw %+.2f' % (twist.linear.x, twist.linear.y, twist.angular.z)
        for i, text in enumerate((status, cmd)):
            cv2.putText(debug_frame, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        try:
            debug_msg = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            self._debug_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().error('Failed to publish debug image: %s' % exc)


def main(args=None):
    rclpy.init(args=args)
    node = TelloLineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
