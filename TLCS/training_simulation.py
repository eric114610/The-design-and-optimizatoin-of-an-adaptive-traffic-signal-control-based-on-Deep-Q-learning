import traci
import numpy as np
import random
import timeit
import os
import math

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs, args):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self.args = args
        self.car_limit = args.num_lanes*5
        self.turn_limit = 5
        self.straight_lanes = []
        self.turn_lanes = []
        self.lanes_init()


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state(old_action)

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)+1
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = int(math.log(wait_time))*10
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def _get_state(self, old_action):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()
        lane_car = np.zeros(16)

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos

            if lane_pos < 0:
                lane_cell = 9
            elif lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 35:
                lane_cell = 4
            elif lane_pos < 45:
                lane_cell = 5
            elif lane_pos < 55:
                lane_cell = 6
            elif lane_pos < 65:
                lane_cell = 7
            elif lane_pos < 80:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            
            car_type = 0
            
            if self.args.car_type != 'all_car':
                car_vclass = traci.vehicle.getVehicleClass(car_id)
                if car_vclass == "truck":
                    car_type = 0.5
                elif car_vclass == "bus":
                    car_type = 1
                elif car_vclass == "moped":
                    car_type = -0.5
                else:
                    car_type = 0

            if lane_cell <= 4:
                if lane_id in self.straight_lanes:
                    if lane_id[0] == 'W':
                        if lane_car[0] + car_type < self.car_limit:
                            lane_car[0] += 1 + car_type
                    elif lane_id[0] == 'N':
                        if lane_car[2] + car_type < self.car_limit:
                            lane_car[2] += 1 + car_type
                    elif lane_id[0] == 'E':
                        if lane_car[4] + car_type < self.car_limit:
                            lane_car[4] += 1 + car_type
                    elif lane_id[0] == 'S':
                        if lane_car[6] + car_type < self.car_limit:
                            lane_car[6] += 1 + car_type

                if lane_id in self.turn_lanes:
                    if lane_id[0] == 'W':
                        if lane_car[1] + car_type < self.turn_limit:
                            lane_car[1] += 1 + car_type
                    elif lane_id[0] == 'N':
                        if lane_car[3] + car_type < self.turn_limit:
                            lane_car[3] += 1 + car_type
                    elif lane_id[0] == 'E':
                        if lane_car[5] + car_type < self.turn_limit:
                            lane_car[5] += 1 + car_type
                    elif lane_id[0] == 'S':
                        if lane_car[7] + car_type < self.turn_limit:
                            lane_car[7] += 1 + car_type
                

        
        flow_count=0
        if old_action==0:
            flow_count = (lane_car[2] + lane_car[6])/self.args.num_lanes
            lane_car[2]=0; lane_car[6] = 0
            state[8] = flow_count
        if old_action==1:
            flow_count = (lane_car[3] + lane_car[7])/self.args.num_lanes
            lane_car[3]=0; lane_car[7]=0
            state[9] = flow_count
        if old_action==2:
            flow_count = (lane_car[0] + lane_car[4])/self.args.num_lanes
            lane_car[0]=0; lane_car[4]=0
            state[10] = flow_count
        if old_action==3:
            flow_count = (lane_car[1] + lane_car[5])/self.args.num_lanes
            lane_car[1]=0; lane_car[5]=0
            state[11] = flow_count

        for i in range(8):
            total_s = 0
            if i%2==0:
                for j in np.arange(0, lane_car[i]/self.args.num_lanes, 0.1):
                    total_s += -0.08*(j)*(j)*(j) + 10
            else:
                total_s = lane_car[i]
            state[i] = total_s

        return state


    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = q_s_a[i][action] + self._gamma * (reward + self._gamma * np.amax(q_s_a_d[i]) - q_s_a[i][action])
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode


    def lanes_init(self):
        if(self.args.num_lanes > 0):
            self.straight_lanes.extend(["W2TL_0", "E2TL_0", "N2TL_0", "S2TL_0"])
        if(self.args.num_lanes > 1):
            self.straight_lanes.extend(["W2TL_1", "E2TL_1", "N2TL_1", "S2TL_1"])
        if(self.args.num_lanes > 2):
            self.straight_lanes.extend(["W2TL_2", "E2TL_2", "N2TL_2", "S2TL_2"])
        if(self.args.num_lanes > 3):
            self.straight_lanes.extend(["W2TL_3", "E2TL_3", "N2TL_3", "S2TL_3"])

        if(self.args.num_lanes == 1):
            self.turn_lanes.extend(["W2TL_1", "E2TL_1", "N2TL_1", "S2TL_1"])
        elif(self.args.num_lanes == 2):
            self.turn_lanes.extend(["W2TL_2", "E2TL_2", "N2TL_2", "S2TL_2"])
        elif(self.args.num_lanes == 3):
            self.turn_lanes.extend(["W2TL_3", "E2TL_3", "N2TL_3", "S2TL_3"])
        elif(self.args.num_lanes == 4):
            self.turn_lanes.extend(["W2TL_4", "E2TL_4", "N2TL_4", "S2TL_4"])

    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

