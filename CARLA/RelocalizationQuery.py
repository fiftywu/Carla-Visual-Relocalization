# obtain the static dataset generated from carla simulator version 0.8
from __future__ import print_function

import argparse
import logging
import random
import time
import sys

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

import numpy as np
import os
import shutil


def run_carla_client(args):
    # Here we will run 1 episodes with 1000 frames each.
    number_of_episodes = 1
    frames_per_episode = 2001
    # 原来1000

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')
        for episode in range(0, number_of_episodes):
            # Start a new episode.
            if args.settings_filepath is None:
                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=args.NumberOfVehicles,
                    NumberOfPedestrians=args.NumberOfPedestrians,
                    WeatherId=args.weatherId,
                    QualityLevel=args.quality_level)

                # Now we want to add a few cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.

                # The default camera captures RGB images of the scene.
                camera0 = Camera('RGB')
                # Set image resolution in pixels.
                camera0.set_image_size(args.image_width, args.image_width)
                # Set its position relative to the car in meters.
                camera0.set_position(1.80, 0, 1.30)
                settings.add_sensor(camera0)

                # Let's add another camera producing ground-truth depth.
                camera1 = Camera('Depth', PostProcessing='Depth')
                camera1.set_image_size(args.image_width, args.image_width)
                camera1.set_position(1.80, 0, 1.30)
                settings.add_sensor(camera1)

                # Let's add another camera producing ground-truth semantic segmentation.
                camera2 = Camera('SemanticSegmentation', PostProcessing='SemanticSegmentation')
                camera2.set_image_size(args.image_width, args.image_width)
                camera2.set_position(1.80, 0, 1.30)
                settings.add_sensor(camera2)

                if args.lidar:
                    lidar = Lidar('Lidar32')
                    lidar.set_position(0, 0, 2.50)
                    lidar.set_rotation(0, 0, 0)
                    lidar.set(
                        Channels=32,
                        Range=50,
                        PointsPerSecond=100000,
                        RotationFrequency=10,
                        UpperFovLimit=10,
                        LowerFovLimit=-30)
                    settings.add_sensor(lidar)

            else:

                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = args.playerStart

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode...')
            client.start_episode(player_start)

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                print('Frame : ', frame)
                save_bool = True

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # Save Trajectory
                save_trajectory(frame, measurements)

                # Read trajectory to check if it is the same #bbescos
                file = open(args.trajectoryFile, 'r')
                lines = file.readlines()
                line = lines[frame]
                words = line.split()
                pos_x_s = float(words[1])
                if abs(pos_x_s - measurements.player_measurements.transform.location.x) > 0.1*5:
                    save_bool = False
                    print(pos_x_s, measurements.player_measurements.transform.location.x)
                pos_y_s = float(words[2])
                if abs(pos_y_s - measurements.player_measurements.transform.location.y) > 0.1*5:
                    save_bool = False
                    print(pos_y_s, measurements.player_measurements.transform.location.y)
                file.close()

                # Save the images to disk if requested.
                if args.save_images_to_disk and frame % 10 == 0 and frame > 29 and save_bool:
                    for name, measurement in sensor_data.items():
                        filename = args.out_filename_format.format(episode, name, frame)
                        measurement.save_to_disk(filename)

                # Now we have to send the instructions to control the vehicle.
                # If we are in synchronous mode the server will pause the
                # simulation until we send this control.
                if not args.autopilot:

                    client.send_control(
                        steer=random.uniform(-1.0, 1.0),
                        throttle=0.5,
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)
                else:

                    # Together with the measurements, the server has sent the
                    # control that the in-game autopilot would do this frame. We
                    # can enable autopilot by sending back this control to the
                    # server.
                    control = measurements.player_measurements.autopilot_control
                    # Read control file
                    file = open(args.controlFile, 'r')
                    lines = file.readlines()
                    line = lines[frame]
                    words = line.split()
                    steer = float(words[1])
                    throttle = float(words[2])
                    brake = float(words[3])
                    hand_brake = (words[4] == 'True')
                    reverse = (words[5] == 'True')
                    ##
                    # steer = round(steer,10)
                    # throttle = round(throttle, 10)
                    # brake = round(brake, 10)
                    ##
                    file.close()
                    control.steer = steer
                    control.throttle = throttle
                    control.brake = brake
                    control.hand_brake = hand_brake
                    control.reverse = reverse
                    save_control(frame, control)
                    client.send_control(control)


def save_trajectory(frame, measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements

    pos_x = player_measurements.transform.location.x
    pos_y = player_measurements.transform.location.y

    file = open("Trajectory_s.txt", "a")
    file.write("%5i %5.1f %5.1f\n" % (frame, pos_x, pos_y))
    file.close()


def save_control(frame, control):
    steer = control.steer
    throttle = control.throttle
    brake = control.brake
    hand_brake = control.hand_brake
    reverse = control.reverse

    file = open("Control_s.txt", "a")
    file.write("%5i %1.50f %2.2f %2.2f %r %r \n" % (frame, steer, throttle, brake, hand_brake, reverse))
    file.close()


def run_carla_client_static(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    args.out_filename_format = '_out_s/episode_{:0>4d}/{:s}/{:0>6d}'
    args.out_SLAM_filename_format = '_out_s/episode_{:0>4d}/SLAM/{:s}/{:0>6d}'

    while True:
        try:
            run_carla_client(args)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


class ArgsClass():
    def __init__(self, weather_id, player_start_id, control_file, trajectory_file, image_size, NumberOfVehicles, NumberOfPedestrians):
        self.autopilot = True
        self.controlFile = control_file
        self.debug = False
        self.host = 'localhost'
        self.lidar = False
        self.playerStart = player_start_id
        self.port = 2000
        self.quality_level = 'Epic'
        self.save_images_to_disk = True
        self.settings_filepath = None
        self.trajectoryFile = trajectory_file
        self.weatherId = weather_id
        self.out_filename_format = None
        self.out_SLAM_filename_format = None
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.NumberOfVehicles = NumberOfVehicles
        self.NumberOfPedestrians = NumberOfPedestrians


if __name__ == '__main__':
    Dir_Town01 = 'D:/DownLoad/CARLA_0.8.2/PythonClient/scripts/CARLA/RelocalizationData/Town01/'
    Dir_Town02 = 'D:/DownLoad/CARLA_0.8.2/PythonClient/scripts/CARLA/RelocalizationData/Town02/'
    source_dir = 'D:/DownLoad/CARLA_0.8.2/PythonClient/scripts/CARLA/_out_s/episode_0000/'
    ctrl = 'D:/DownLoad/CARLA_0.8.2/PythonClient/scripts/CARLA/Control_s.txt'
    trj = 'D:/DownLoad/CARLA_0.8.2/PythonClient/scripts/CARLA/Trajectory_s.txt'

    # Shortcut per 20 frame
    image_size = [256, 256]
    dynamic_condition = [0, 0]  #  [[75,300], [200, 650], [150,500], [100, 350], [50, 200]] Vehicles Pedestrians

    first_town = False

    if first_town:
        # Town01Train static dataset generation
            train_list = os.listdir(Dir_Town01)
            for list_ in train_list:
            # for list_ in ['W000_P010_V000_P000', ]
                ctrl_file = os.path.join(Dir_Town01, list_, 'Control.txt')
                trj_file = os.path.join(Dir_Town01, list_, 'Trajectory.txt')
                weather = int(list_[1:4])
                player_start = int(list_[6:9])
                print('weather: ',weather, 'start_position: ', player_start, 'dynamic_condition: ', dynamic_condition)
                args = ArgsClass(weather, player_start, ctrl_file, trj_file, image_size, dynamic_condition[0], dynamic_condition[1])
                run_carla_client_static(args)
                destination_dir = os.path.join(Dir_Town01, "W%03d_P%03d_V%03d_P%03d" % (weather, player_start, dynamic_condition[0], dynamic_condition[1]))

                if not os.path.isdir(destination_dir):
                    os.makedirs(destination_dir)
                folders = os.listdir(source_dir)
                for folder in folders:
                    shutil.move(os.path.join(source_dir, folder), destination_dir)
                shutil.move(ctrl, destination_dir)
                shutil.move(trj, destination_dir)

                # # Because the control error so delete the unmatched pictures
                # if len(os.listdir(os.path.join(destination_dir,'RGB'))) != len(os.listdir(os.path.join(Dir_Town01, list_, 'RGB'))):
                #     figure_list = os.listdir(os.path.join(Dir_Town01, list_, 'RGB'))
                #     dynamic_list = os.listdir(os.path.join(destination_dir, 'RGB'))
                #     for figure in figure_list:
                #         if figure not in dynamic_list:
                #             # delete dynamic figures accordingly
                #             os.remove(os.path.join(Dir_Town01, list_, 'RGB', figure))
                #             os.remove(os.path.join(Dir_Town01, list_, 'Depth', figure))
                #             os.remove(os.path.join(Dir_Town01, list_, 'SemanticSegmentation', figure))

    else:
        # Town02Test static dataset generation
            train_list = os.listdir(Dir_Town02)
            list_ = 'W000_P100_V075_P300'
            ctrl_file = os.path.join(Dir_Town02, list_, 'Control.txt')
            trj_file = os.path.join(Dir_Town02, list_, 'Trajectory.txt')
            weather = int(list_[1:4])
            player_start = int(list_[6:9])
            print('weather: ', weather, 'start_position: ', player_start, 'dynamic_condition', dynamic_condition)
            args = ArgsClass(weather, player_start, ctrl_file, trj_file, image_size, dynamic_condition[0], dynamic_condition[1])
            run_carla_client_static(args)
            destination_dir = os.path.join(Dir_Town02, "W%03d_P%03d_V%03d_P%03d" % (weather, player_start, dynamic_condition[0], dynamic_condition[1]))
            if not os.path.isdir(destination_dir):
                os.makedirs(destination_dir)
            folders = os.listdir(source_dir)
            for folder in folders:
                shutil.move(os.path.join(source_dir, folder), destination_dir)
            shutil.move(ctrl, destination_dir)
            shutil.move(trj, destination_dir)

            # # Because the control error so delete the unmatched pictures
            # if len(os.listdir(os.path.join(destination_dir,'RGB'))) != len(os.listdir(os.path.join(Dir_Town02, list_, 'RGB'))):
            #     figure_list = os.listdir(os.path.join(Dir_Town02, list_, 'RGB'))
            #     dynamic_list = os.listdir(os.path.join(destination_dir, 'RGB'))
            #     for figure in figure_list:
            #         if figure not in dynamic_list:
            #             # delete dynamic figures accordingly
            #             os.remove(os.path.join(Dir_Town02, list_, 'RGB', figure))
            #             os.remove(os.path.join(Dir_Town02, list_, 'Depth', figure))
            #             os.remove(os.path.join(Dir_Town02, list_, 'SemanticSegmentation', figure))