import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/emilio/Control/Tello/tello_ws/install/tello_line_follower'
