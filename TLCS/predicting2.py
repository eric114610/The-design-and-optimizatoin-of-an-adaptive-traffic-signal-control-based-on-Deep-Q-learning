from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile
import time

from predicting_simulation2 import Simulation
from generator import TrafficGenerator
from model import PredictModel
from visualization import Visualization
from utils import import_predict_configuration, set_sumo, set_predict_path

from firebase_get import firebase_Get
import serial

class CarDistribute:
    def __init__(self, l):
        self._car = l[0]
        self._bus = l[1]
        self._truck = l[2]
        self._total = l[0] + l[1] + l[2]

class CarQueue:
    def __init__(self, car_q):
        self.N_Straight = self._buildDirection(car_q[0:3])
        self.S_Straight = self._buildDirection(car_q[6:9])
        self.W_Straight = self._buildDirection(car_q[12:15])
        self.E_Straight = self._buildDirection(car_q[18:21])
        self.N_Turn = self._buildDirection(car_q[3:6])
        self.S_Turn = self._buildDirection(car_q[9:12])
        self.W_Turn = self._buildDirection(car_q[15:18])
        self.E_Turn = self._buildDirection(car_q[21:])

    def _buildDirection(self, l):
        return CarDistribute(l)



import argparse

def parse_arguments2():
    # Create the parser
    parser = argparse.ArgumentParser(description='Traffic Simulation Parameters')
    
    # Add the num_lanes argument with a default value of 3
    parser.add_argument(
        '--num_lanes',
        type=int,
        default=2,
        help='Number of lanes in the simulation (default: 3)'
    )
    
    # Add the car_type argument
    parser.add_argument(
        '--car_type',
        type=str,
        required=True,
        choices=['cars', 'all_car', 'motor'],
        help='Type of car to simulate (cars, all_car, or motor)'
    )

    parser.add_argument(
        '--turn_left',
        type=bool,
        default=False,
        help='has turning light or false'
    )
    
    # Parse the arguments
    args = parser.parse_args()
    print(args)
    return args


import random
def generate_random_numbers():
    while True:
        numbers = [random.randint(0, 4) for _ in range(3)]
        total_sum = sum(numbers)
        if 1 < total_sum < 15:
            return sorted(numbers, reverse=True), total_sum
        

def change_mode(L, tmp_args, Models, tmp_simulations, old_predict):
    print("command:", L)
    # predict = False
    predict = old_predict

    if L == "b'TL\\n'":
        tmp_args.turn_left = 1-tmp_args.turn_left
    elif L == "b'MT\\n'":
        if tmp_args.car_type != "motor":
            tmp_args.car_type = "motor"
        else:
            tmp_args.car_type == "cars"
    elif L == "b'END\\n'":
        predict = False
    elif L == "b'START\\n'":
        predict = True
    else:
        tmp_args.num_lanes = int(L[2:-3])
        Simulations = []

        for i,v in enumerate(Models):
            TmpSimulation = Simulation(
                v,
                config['max_steps'],
                config['green_duration'],
                config['yellow_duration'],
                config['num_states'],
                config['num_actions'],
                tmp_args.num_lanes
            )
            Simulations.append(TmpSimulation)
        tmp_simulations = Simulations

    return tmp_args, tmp_simulations, predict


if __name__ == "__main__":

    config = import_predict_configuration(config_file='predicting_settings.ini')
    #sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    # print(config['model_to_load'])
    

    model_path, plot_path = set_predict_path(config['models_path_name'], config['model_to_load'], 'predicts')
    #print(model_path, plot_path)

    args = parse_arguments2()   


    Models = []

    for i,v in enumerate(model_path):
        Model = PredictModel(
            input_dim=config['num_states'],
            model_path=model_path[i]
        )
        Models.append(Model)

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated'],
        args
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )

    Simulations = []

    for i,v in enumerate(Models):
        
        TmpSimulation = Simulation(
            v,
            config['max_steps'],
            config['green_duration'],
            config['yellow_duration'],
            config['num_states'],
            config['num_actions'],
            args.num_lanes
        )
        Simulations.append(TmpSimulation)

    # firebase_db = firebase_Get()
    ser = serial.Serial('COM3', 115200, timeout=0.05)

    print("End Init")

    # predict = True
    # first = True
    # result = 0
    Gfirst = True
    predict = True

    
    print("---------------------------WAITING----------------------------")
    kalsdnk = input()

    while True:

        with open("close.txt", "r") as C:
            content = C.read()
            if content == "END":
                break


        if not Gfirst:
            args = tmp_args
            Simulations = tmp_Simulations

            print(args)

            while(not predict):
                L = str(ser.readline())
                if L != "b''":
                    tmp_args, tmp_Simulations, predict = change_mode(L, tmp_args, Models, tmp_Simulations, predict)
                    break

        else:
            tmp_args = args
            tmp_Simulations = Simulations

        Gfirst = False
        
        first = True
        result = 0
        count=0

        while predict: #predict:
            start_time = time.perf_counter()

            with open("close.txt", "r") as C:
                content = C.read()
                if content == "END":
                    break

            L = str(ser.readline())
            # print(L, type(L))
            if L != "b''":
                tmp_args, tmp_Simulations, predict = change_mode(L, tmp_args, Models, tmp_Simulations, predict)

            # car_queue = firebase_db.get()
            # print(car_queue)
            with open("close.txt", "r") as C:
                content = C.read()
                if content == "END":
                    break

            if not predict:
                break

            if count==0:
                # with open("test.txt", "w") as file:
                #     for i in range(8):
                #         random_numbers, total_sum = generate_random_numbers()
                #         for item in random_numbers:
                #             file.write(f"{item}\n")
                #         file.write(f"{total_sum}\n")

                with open("test.txt", "r") as file:
                    content = file.read()

                car_queue = [int(item.strip()) for item in content.split("\n")[:-1]]
                # print(car_queue)
            else:
                car_queue = [0]*32
            # print(my_list*8)
            
            #print(car_q)
            car_queue = CarQueue(car_queue)
            
            if args.turn_left:
                result = Simulations[0].run(car_queue)
            else:
                result = Simulations[1].run(car_queue)
            

            if result:
                print(result)

            if result < 5 and result > 0:
                count=5
            elif result > 0:
                count=4

            if result == 1:
                if not args.turn_left:
                    ser.write(b'x')
                else:
                    ser.write(b'g')
            elif result == 5 or result == 6:
                ser.write(b'y')
            elif result == 2:
                ser.write(b'b')
            elif result != 0:
                ser.write(b'r')

            ser.readline()
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            #print(elapsed_time)
            if not first:
                time.sleep(max(1-elapsed_time, 0))
            
            first = False
            count -= 1

        print("Simulation Ended")

    print("System Ended")