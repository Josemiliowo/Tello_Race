"""Qt window: Tello camera view, live PD gain sliders for tello_line_follower, takeoff/land."""

import sys
import threading

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from python_qt_binding.QtCore import QObject, Qt, QTimer, Signal
from python_qt_binding.QtGui import QImage, QKeySequence, QPixmap
from python_qt_binding.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QShortcut, QSlider, QVBoxLayout, QWidget)

# (parameter, label, max). Sliders go from 0 to max in steps of 0.01.
TUNABLES = [
    ('kp_yaw', 'Kp yaw', 3.0),
    ('kd_yaw', 'Kd yaw', 1.0),
    ('kp_lateral', 'Kp lateral', 2.0),
    ('kd_lateral', 'Kd lateral', 1.0),
    ('linear_speed', 'Forward speed', 1.0),
]
STEPS_PER_UNIT = 100

# AprilTag ids offered in the GUI, and the follower parameter behind each action column.
TAG_IDS = [1, 2, 3, 4]
TAG_ACTIONS = [('left_turn_ids', 'Left'), ('right_turn_ids', 'Right'), ('land_ids', 'Land')]


class _Bridge(QObject):
    """Carries data from ROS threads to the Qt thread."""

    image = Signal(object)
    status = Signal(str)
    gains = Signal(object)
    tags = Signal(object)


class TelloGuiNode(Node):

    def __init__(self, bridge):
        super().__init__('tello_gui')
        self.declare_parameter('follower_node', 'tello_line_follower')
        self.declare_parameter('image_topics', [
            'tello/line_follower/debug_image', 'tello/camera/image_raw'])
        self.declare_parameter('takeoff_service', 'tello/takeoff')
        self.declare_parameter('land_service', 'tello/land')

        self._bridge = bridge
        follower = self.get_parameter('follower_node').value.strip('/')
        self.image_topics = list(self.get_parameter('image_topics').value)

        self._set_client = self.create_client(SetParameters, '/%s/set_parameters' % follower)
        self._get_client = self.create_client(GetParameters, '/%s/get_parameters' % follower)
        self._takeoff_client = self.create_client(
            Empty, self.get_parameter('takeoff_service').value)
        self._land_client = self.create_client(Empty, self.get_parameter('land_service').value)
        self._image_sub = None
        self.select_image_topic(self.image_topics[0])

    def select_image_topic(self, topic):
        if self._image_sub is not None:
            self.destroy_subscription(self._image_sub)
        self._image_sub = self.create_subscription(
            Image, topic, self._image_callback, qos_profile_sensor_data)

    def _image_callback(self, msg):
        if msg.encoding not in ('bgr8', 'rgb8'):
            return
        frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        self._bridge.image.emit((frame.copy(), msg.width, msg.height, msg.step, msg.encoding))

    def fetch_gains(self):
        if not self._get_client.service_is_ready():
            return False
        names = [name for name, _, _ in TUNABLES]
        future = self._get_client.call_async(GetParameters.Request(names=names))

        def done(fut):
            values = fut.result().values
            self._bridge.gains.emit(
                {n: v.double_value for n, v in zip(names, values)
                 if v.type == ParameterType.PARAMETER_DOUBLE})
        future.add_done_callback(done)
        return True

    def fetch_tags(self):
        if not self._get_client.service_is_ready():
            return False
        names = [name for name, _ in TAG_ACTIONS]
        future = self._get_client.call_async(GetParameters.Request(names=names))

        def done(fut):
            values = fut.result().values
            self._bridge.tags.emit(
                {n: set(v.integer_array_value) for n, v in zip(names, values)})
        future.add_done_callback(done)
        return True

    def set_tag_ids(self, name, ids):
        if not self._set_client.service_is_ready():
            self._bridge.status.emit('Line follower not running; %s not sent.' % name)
            return
        # The follower uses [-1] for "none" because ROS can't type an empty integer list.
        param = Parameter(name=name, value=ParameterValue(
            type=ParameterType.PARAMETER_INTEGER_ARRAY,
            integer_array_value=sorted(ids) or [-1]))
        self._set_client.call_async(SetParameters.Request(parameters=[param]))

    def set_gain(self, name, value):
        if not self._set_client.service_is_ready():
            self._bridge.status.emit('Line follower not running; %s not sent.' % name)
            return
        param = Parameter(name=name, value=ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)))
        self._set_client.call_async(SetParameters.Request(parameters=[param]))

    def call(self, client, label):
        if not client.service_is_ready():
            self._bridge.status.emit('%s service not available (is tello_controller up?)' % label)
            return
        self._bridge.status.emit('%s requested...' % label)
        future = client.call_async(Empty.Request())
        future.add_done_callback(lambda _: self._bridge.status.emit('%s sent.' % label))

    def takeoff(self):
        self.call(self._takeoff_client, 'Takeoff')

    def land(self):
        self.call(self._land_client, 'Land')


class TelloGuiWindow(QWidget):

    def __init__(self, node, bridge):
        super().__init__()
        self._node = node
        self.setWindowTitle('Tello line follower')

        self._video = QLabel('Waiting for video...')
        self._video.setAlignment(Qt.AlignCenter)
        self._video.setMinimumSize(480, 360)
        self._video.setStyleSheet('background: #202020; color: #aaa;')

        topic_box = QComboBox()
        topic_box.addItems(node.image_topics)
        topic_box.currentTextChanged.connect(node.select_image_topic)

        left = QVBoxLayout()
        left.addWidget(topic_box)
        left.addWidget(self._video, 1)

        self._sliders = {}
        self._value_labels = {}
        self._pending = {}
        gains_box = QGroupBox('Line follower gains (applied live)')
        grid = QGridLayout(gains_box)
        for row, (name, label, maximum) in enumerate(TUNABLES):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, int(maximum * STEPS_PER_UNIT))
            slider.setEnabled(False)
            value_label = QLabel('-')
            value_label.setMinimumWidth(40)
            slider.valueChanged.connect(lambda v, n=name: self._on_slider(n, v))
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value_label, row, 2)
            self._sliders[name] = slider
            self._value_labels[name] = value_label

        # One row per tag id, one checkbox per action. A tag can only have one action.
        tags_box = QGroupBox('AprilTag actions (applied live)')
        tag_grid = QGridLayout(tags_box)
        self._tag_checks = {}
        for col, (_, label) in enumerate(TAG_ACTIONS, start=1):
            tag_grid.addWidget(QLabel(label), 0, col, Qt.AlignCenter)
        for row, tag_id in enumerate(TAG_IDS, start=1):
            tag_grid.addWidget(QLabel('Tag %d' % tag_id), row, 0)
            for col, (param, _) in enumerate(TAG_ACTIONS, start=1):
                check = QCheckBox()
                check.setEnabled(False)
                check.toggled.connect(
                    lambda on, i=tag_id, p=param: self._on_tag_toggled(i, p, on))
                tag_grid.addWidget(check, row, col, Qt.AlignCenter)
                self._tag_checks[(tag_id, param)] = check

        # Send slider changes at most every 150 ms instead of on every pixel of movement.
        self._send_timer = QTimer(self)
        self._send_timer.setSingleShot(True)
        self._send_timer.setInterval(150)
        self._send_timer.timeout.connect(self._flush_gains)

        takeoff_btn = QPushButton('TAKEOFF')
        takeoff_btn.setMinimumHeight(50)
        takeoff_btn.setStyleSheet('background: #2e7d32; color: white; font-weight: bold;')
        takeoff_btn.clicked.connect(node.takeoff)
        land_btn = QPushButton('LAND  (Space)')
        land_btn.setMinimumHeight(50)
        land_btn.setStyleSheet('background: #c62828; color: white; font-weight: bold;')
        land_btn.clicked.connect(node.land)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=node.land)

        buttons = QHBoxLayout()
        buttons.addWidget(takeoff_btn)
        buttons.addWidget(land_btn)

        self._status = QLabel('Waiting for tello_line_follower...')
        self._status.setWordWrap(True)

        right = QVBoxLayout()
        right.addWidget(gains_box)
        right.addWidget(tags_box)
        right.addStretch(1)
        right.addLayout(buttons)
        right.addWidget(self._status)

        layout = QHBoxLayout(self)
        layout.addLayout(left, 3)
        layout.addLayout(right, 2)

        bridge.image.connect(self._show_image)
        bridge.status.connect(self._status.setText)
        bridge.gains.connect(self._load_gains)
        bridge.tags.connect(self._load_tags)

        # Poll until the follower is up, then load its current gains into the sliders.
        self._fetch_timer = QTimer(self)
        self._fetch_timer.timeout.connect(self._try_fetch)
        self._fetch_timer.start(500)

    def _try_fetch(self):
        if self._node.fetch_gains() and self._node.fetch_tags():
            self._fetch_timer.stop()

    def _load_gains(self, gains):
        for name, value in gains.items():
            slider = self._sliders[name]
            slider.blockSignals(True)
            slider.setValue(int(round(value * STEPS_PER_UNIT)))
            slider.blockSignals(False)
            slider.setEnabled(True)
            self._value_labels[name].setText('%.2f' % value)
        self._status.setText('Connected to tello_line_follower.')

    def _load_tags(self, tags):
        for (tag_id, param), check in self._tag_checks.items():
            check.blockSignals(True)
            check.setChecked(tag_id in tags.get(param, set()))
            check.blockSignals(False)
            check.setEnabled(True)

    def _on_tag_toggled(self, tag_id, param, on):
        changed = {param}
        if on:
            # Exclusive per tag: untick this tag's other actions.
            for other, _ in TAG_ACTIONS:
                check = self._tag_checks[(tag_id, other)]
                if other != param and check.isChecked():
                    check.blockSignals(True)
                    check.setChecked(False)
                    check.blockSignals(False)
                    changed.add(other)
        for name in changed:
            ids = {i for i in TAG_IDS if self._tag_checks[(i, name)].isChecked()}
            self._node.set_tag_ids(name, ids)

    def _on_slider(self, name, steps):
        value = steps / float(STEPS_PER_UNIT)
        self._value_labels[name].setText('%.2f' % value)
        self._pending[name] = value
        if not self._send_timer.isActive():
            self._send_timer.start()

    def _flush_gains(self):
        for name, value in self._pending.items():
            self._node.set_gain(name, value)
        self._pending.clear()

    def _show_image(self, data):
        frame, width, height, step, encoding = data
        fmt = QImage.Format_BGR888 if encoding == 'bgr8' else QImage.Format_RGB888
        image = QImage(frame.data, width, height, step, fmt)
        pixmap = QPixmap.fromImage(image).scaled(
            self._video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._video.setPixmap(pixmap)


def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    bridge = _Bridge()
    node = TelloGuiNode(bridge)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    window = TelloGuiWindow(node, bridge)
    window.resize(1100, 560)
    window.show()

    # Let Ctrl+C in the terminal close the window.
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)
    import signal
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    try:
        app.exec_()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
