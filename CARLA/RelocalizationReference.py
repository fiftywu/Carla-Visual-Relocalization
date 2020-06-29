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
    # Here we will run 1 episodes with 1000 frames.
    number_of_episodes = 1
    # 原来作者设置的是1000
    frames_per_episode = 2001

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
            print('player_start', player_start)
            print('weather_Id', args.weatherId)

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode...')
            client.start_episode(player_start)
            print('client: ', client)

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                print('Frame : ', frame)
                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # Save Trajectory
                save_trajectory(frame, measurements)

                # Save the images to disk if requested. We save 1 frame out of 10, from frame 30 on.
                # In the first frames the car is 'flying' and the lightning is not correct.
                if args.save_images_to_disk and frame % 10 == 0 and frame > 29:
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
                    save_control(frame, control)
                    client.send_control(control)


def save_trajectory(frame, measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    pos_x = player_measurements.transform.location.x
    pos_y = player_measurements.transform.location.y
    file = open("Trajectory.txt", "a")
    file.write("%5i %5.1f %5.1f\n" % (frame, pos_x, pos_y))
    file.close()


def save_control(frame, control):
    steer = control.steer
    throttle = control.throttle
    brake = control.brake
    hand_brake = control.hand_brake
    reverse = control.reverse
    file = open("Control.txt", "a")
    file.write("%5i %1.50f %2.2f %2.2f %r %r \n" % (frame, steer, throttle, brake, hand_brake, reverse))
    file.close()


def run_carla_client_dynamic(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'
    while True:
        try:
            run_carla_client(args)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


class ArgsClass():
    def __init__(self, weather_id, player_start_id, image_size, NumberOfVehicles, NumberOfPedestrians):
        self.autopilot = True
        self.debug = False
        self.host = 'localhost'
        self.lidar = False
        self.playerStart = player_start_id
        self.port = 2000
        self.quality_level = 'Epic'
        self.save_images_to_disk = True
        self.settings_filepath = None
        self.weatherId = weather_id
        self.out_filename_format = None
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.NumberOfVehicles = NumberOfVehicles
        self.NumberOfPedestrians = NumberOfPedestrians


if __name__ == '__main__':
    print('start to obtain...')

# If not exist the dir then create
    Dir_Town01 = 'D:/DownLoad/CARLA_0.8.2/PythonClient/scripts/CARLA/RelocalizationData/Town01'
    Dir_Town02 = 'D:/DownLoad/CARLA_0.8.2/PythonClient/scripts/CARLA/RelocalizationData/Town02'
    source_dir = 'D:/DownLoad/CARLA_0.8.2/PythonClient/scripts/CARLA/_out/episode_0000/'
    ctrl = 'D:/DownLoad/CARLA_0.8.2/PythonClient/scripts/CARLA/Control.txt'
    trj = 'D:/DownLoad/CARLA_0.8.2/PythonClient/scripts/CARLA/Trajectory.txt'

    if not os.path.isdir(Dir_Town01):
        print('not found', Dir_Town01, 'so create it')
        os.makedirs(Dir_Town01)


# Shortcut per 20 frame
    weather = 0  # max num of weather is 14, from 0 to 13
    player_start = 100
    image_size = [256, 256]
    dynamic_condition = [75, 300]  # [[150,500], [100, 350], [50, 200]] Vehicles Pedestrians

    first_town = False

    if first_town:
        # Town01Train dataset generation
        print('# weather: ',weather,' player_start: ', player_start)
        args = ArgsClass(weather, player_start, image_size, dynamic_condition[0], dynamic_condition[1])
        run_carla_client_dynamic(args)
        # cache the output in scripts/CARLA/_out/episode_0000
        # then move these data into specialized dir
        destination_dir = os.path.join(Dir_Town02, "W%03d_P%03d_V%03d_P%03d" % (weather, player_start, dynamic_condition[0], dynamic_condition[1]))
        if not os.path.isdir(destination_dir):
            os.makedirs(destination_dir)
        folders = os.listdir(source_dir)
        for folder in folders:
            shutil.move(os.path.join(source_dir, folder), destination_dir)
        shutil.move(ctrl, destination_dir)
        shutil.move(trj, destination_dir)

    else:
        # Town02Test dataset generation
        print('# weather: ',weather,' player_start: ', player_start,
              'NumberOfVehicles',dynamic_condition[0], 'NumberOfPedestrians', dynamic_condition[1])
        args = ArgsClass(weather, player_start, image_size, dynamic_condition[0], dynamic_condition[1])
        run_carla_client_dynamic(args)
        destination_dir = os.path.join(Dir_Town02, "W%03d_P%03d_V%03d_P%03d" % (weather, player_start, dynamic_condition[0], dynamic_condition[1]),)
        if not os.path.isdir(destination_dir):
            os.makedirs(destination_dir)
        folders = os.listdir(source_dir)
        for folder in folders:
            shutil.move(source_dir+folder, destination_dir)
        shutil.move(ctrl, destination_dir)
        shutil.move(trj, destination_dir)
