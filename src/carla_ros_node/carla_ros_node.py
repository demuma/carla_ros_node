#!/usr/bin/env python

# Python modules
from cProfile import label
from enum import auto
from time import time
from token import GREATER
from bcrypt import kdf
from cv2 import integral
import matplotlib
from numpy import diff
import matplotlib.pyplot as plt

from sklearn.decomposition import sparse_encode
import rospy
import sys
import numpy
import datetime
import math
import os

# Message types
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Accel, Quaternion
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl
from carla import VehicleControl

# PID controller variables
global lidar, laser_scan, control, target_velocity, velocity, acceleration, orientation, manual_control, auto_pilot, target_velocity, set_velocity, prev_velocity, error_velocity_change, error_velocity_sum, error_velocity, dt
lidar = PointCloud2()
laser_scan = LaserScan()
control = Twist()
velocity = float()
acceleration = Accel()
orientation = Quaternion()
manual_control = Bool()
auto_pilot = Bool()
target_velocity = float()
error_velocity_change = float()
set_velocity = float()
prev_velocity = float()
prev_velocity = 0.0
target_velocity = 100 / 3.6
I = 0.0
error_velocity = 0.0
dt = 0.05 # FPS

# Plot
global time_step
time_step = 0.0

# Distances
global front_distance, back_distance, right_distance, left_distance

# Class for AFRP4
class carla_ros_node:

    def __init__(self):

        # Set manual override to true (enable manual control)
        # Topic: /carla/ego_vehicle/vehicle_control_manual_override
        # Data: std_msgs/Bool
        self.pub_manual_control = rospy.Publisher('/carla/ego_vehicle/vehicle_control_manual_override', Bool, queue_size=10)

        # Send drive command to node carla_twist_to_control
        # Topic: /carla/ego_vehicle/twist
        # Data type: geometry_msg/Twist
        self.pub_drive_control = rospy.Publisher('/carla/ego_vehicle/twist', Twist, queue_size=10)

        # Receive laser scan data from the pointcloud to laser scan converter node
        # Topic: /laser_scan
        # Data type: sensor_msgs/LaserScan
        self.sub_laser_scan = rospy.Subscriber('/carla/ego_vehicle/laser_scan', LaserScan, laser_scan_callback)

        # Receive lidar point cloud from ego_vehicle
        # Topic: /carla/ego_vehicle/lidar
        # Data type: sensor_msgs/
        # self.sub_lidar_data = rospy.Subscriber('/carla/ego_vehicle/lidar', PointCloud2, lidar_callback)

        # Set auto pilot to true (enable manual control)
        # Topic: /carla/ego_vehicle/enable_autopilot
        # Data type: std_msgs/Bool
        self.pub_auto_pilot = rospy.Publisher('/carla/ego_vehicle/enable_autopilot', Bool, queue_size=10)

        # Get current vehicle status (velocity, acceleration, orientation in Quaternions)
        # Topic: /carla/ego_vehicle/vehicle_status
        # Data type: CarlaEgoVehicleStatus
        self.sub_vehicle_status = rospy.Subscriber('/carla/ego_vehicle/vehicle_status', CarlaEgoVehicleStatus, vehicle_status_callback)

        # Initializing node and setting up publisher
        rospy.init_node('carla_ros_node', anonymous=False)

    # Enable manual control in order to send control messages
    def manual_control(self, status):
        manual_control.data = status
        self.pub_manual_control.publish(manual_control)
        #print("Manual control enabled")

    # Disable manual control in order to send control messages
    def disable_manual_control(self):
        manual_control.data = False
        self.pub_manual_control.publish(manual_control)
        print("Manual control disabled")

    # Enable manual control in order to send control messages
    def enable_auto_pilot(self):
        auto_pilot.data = True
        self.pub_auto_pilot.publish(auto_pilot)
        print("Autopilot enabled")

    # Enable manual control in order to send control messages
    def disable_auto_pilot(self):
        auto_pilot.data = False
        self.pub_auto_pilot.publish(auto_pilot)
        print("Autopilot disabled")

    # Send cmd_vel messages to twist to control node 
    def send_control_message(self, lin_x, ang_z):
        control.linear.x = lin_x
        control.angular.z = ang_z
        self.pub_drive_control.publish(control)

def laser_scan_callback(msg):

    global front_distance, back_distance, right_distance, left_distance

    back_distance = abs(msg.ranges[0]-2.70)
    right_distance = abs(msg.ranges[124]-1.50)
    front_distance = abs(msg.ranges[249]-2.96)
    left_distance = abs(msg.ranges[374]-1.55)
    
    #print('Front: %.2fm  Right: %.2fm   Back: %.2fm  Left: %.2fm' %(front_distance, right_distance, back_distance, left_distance))

def vehicle_status_callback(msg):
    global velocity, acceleration, prev_velocity, time_step, dt
    prev_velocity = velocity
    velocity = msg.velocity
    acceleration = msg.acceleration.linear.x
    time_step += dt
    #print('Current velocity: %4.2fm/s Current acceleration: %5.2fm/s3 Front: %4.2fm  Right: %4.2fm   Back: %4.2fm  Left: %4.2fm' %(velocity, acceleration, front_distance, right_distance, back_distance, left_distance), end='\r')

def pid_control():
     
    global dt, velocity, target_velocity, error_velocity_change, set_velocity, prev_velocity, I, error_velocity

    # Gains
    kP = 1.8  #0.8
    kI = 0.1  #0.1
    kD = 0.0  #0.5

    # Error
    error_velocity_change = velocity - prev_velocity
    error_velocity = target_velocity - velocity
    
    # Components
    P = kP * error_velocity
    I += kI * error_velocity * dt
    D = kD * error_velocity_change / dt

    # PID output
    PID = P + I + D

    #print('Diff: %2.2f' %error_velocity_change, end='\r')
    print('Target Velocity: %2.2f PID: %2.2f Current velocity: %2.2f PID Velocity: %2.2f' %(target_velocity*3.6, PID, velocity*3.6, PID*3.6), end='\r')
    
    # Print PID components
    #print('PID: %3.2f P: %3.2f I: %3.2f D: %3.2f' %(PID. P, I, D), end='\r')

    return PID

if __name__ == '__main__':

    # Plot
    ax = plt.axes()
    ax.set_xlabel("Time in sec")
    ax.set_ylabel("Velocity in km/h")
    plt.axhline(y=target_velocity*3.6, color='g', linestyle='-', label="Target Velocity")
    plt.title("Plot of Velocities")

    # Instantiate new class object
    vehicle  = carla_ros_node()

    # Set rate
    rate = rospy.Rate(20)

    # Disable manual control for ego_vehicle
    vehicle.manual_control(False)

    # Control loop
    while not rospy.is_shutdown():

        # Plot current velocity
        ax.scatter([time_step], [velocity*3.6], c="red", s=4, label='velocities')
        ax.plot([time_step], [velocity*3.6])
        ax.autoscale()
        plt.plot()
        plt.draw()
        plt.pause(0.01)
        plt.legend(["Target Velocity"])

        # Send velocity from PID controller
        vehicle.send_control_message(pid_control(), 0)
        
        rate.sleep()
        #rospy.spin()