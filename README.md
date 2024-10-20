# The-design-and-optimizatoin-of-an-adaptive-traffic-signal-control-based-on-Deep-Q-learning
Project for 2024 Meichu Hackathon NXP Team 6

Aims to automaticly decide traffic signal using computer vision and Deep Q-learning with camera footage at the intersection.


**Slide** : https://shorturl.at/O1lkE
## Deep Q-learning
Deep Q-learning related code is in "TLCS" folder. 

Run the file training_main.py to start training.
Run simulation.py to start simulating. 

We use SUMO simulator to simulate traffic situation and use it as input to train our model.
Once we fetched the data from SUMO, we derive current state of the envirnment based on waiting queue and traffic flow.

### State
To get the current state of the envirnment and since we were limited by the camera angle, we assume that the maximum depth that the camera can see is 35 meters (or max. 5 cars per lane) from the Stop bar at the intersection.
We use waiting queue at the lanes of red light and traffic flow of the lanes with green light to derive state.

### Environment
You can config your own environment using SUMO, or use default environment for training.

For predicting, use command line argument to config environmnet.

#### arguments
--car_type: str ["cars, motor, all_car"]
cars class - contains cars, trucks, buses
all_car - all cars indentify as regular car
motor - contains cars, trucks, buses, motors

--num_lanes: int - number of lanes for 1 direction (regular lane, not only left turn) 

--turn_left: bool - if have only left turn lane

### Action
There are 4 actions the agent can choose from
- North-South Straight: green for lanes in the north and south arm dedicated to turning right or going straight.
- North-South Left Turn: green for lanes in the north and south arm dedicated to turning left.
- East-West Straight: green for lanes in the east and west arm dedicated to turning right or going straight.
- East-West Left Turn: green for lanes in the east and west arm dedicated to turning left.
If the action agent selected is different from the original action, there will be a yellow light for 4 seconds insert between two green lights.

### Reward
Change in *cumulative waiting time* for all the cars in incoming lanes between actions.
The *cunulative waiting time* for each car isn't linear.
Suppose that a car spent t seconds with speed=0 since the spawn, its *cumulative waiting time* will be int(log(t+1)*10)

### Learning Machanism
We implemented Deep Q-learning to train the model. 
We make use of the Q-learning equation Q(s,a) = Q(s,a) + gamma * (reward + gamma • max Q'(s',a') - Q(s,a)) to update the action values and a deep neural network to learn the state-action function.
The neural network is fully connected with 12 neurons as input (the state), 4 hidden layers of 400 neurons each, and the output layers with 4 neurons representing the 4 possible actions.
For training, experience replay is implemented, the experiences are stored at memory.
For each episode, their will be 100 epochs, for each epoch, a small batch of 400 experiences will be fed to train the neural network.


## Vehicle detection
Detection related code in vehicle-detection-yolov8 directory.

Support both images and video realtime detection, able to detect car, motor, buses, trucks.

Utilize YoloV8 with yolov8x.pt for realtime vehicle detection.
Labelling lane edges for detecting each car's lane.
Each direction needs 2 seperate detection results: leftturn lane and straight lanes.


## GUI Operation Guide:
### Tab Traffic light
Shows traffic signals for North-South direction.
Purple light means cars can left turn now. 

### Tab Control
Start button - start simulation .
End button - end current simulation, set control arguments will take affect.
Left turn button - set if current environment has only left turn lane.
Motor button - set simulation for also detecting motor.
Lane dropdown list - set current lanes


## NXP develope
Platform :
- i.Mx RT1060 EVKB

Software :
- MCUXpresso IDE v24.9.25
- Gui-Guider-1.8.0-GA
- SDK 2.16.0
- Tera Term

We used the Gui Guider tool to create a simple traffic light control panel and a demonstration icon for the traffic light. Communication between the board and our computer is established via a serial port. By utilizing the Python library ```pyserial```, we can control the parameters of our Traffic Light Control System through the LED panel on the board and display the corresponding traffic light indications.

Traffic light :
- Red, Yellow, Green, Blue (Green for left turn)

Controls :
- Simulation : Start, End
- Parameters : Lane(int : 2,3,4), LeftTurn(bool), Motorcycle(bool)

## Example Video
[fixed.mp4](https://github.com/eric114610/The-design-and-optimizatoin-of-an-adaptive-traffic-signal-control-based-on-Deep-Q-learning/blob/main/fixed.mp4) - fixed traffic light SUMO simulation  
[controled.mp4](https://github.com/eric114610/The-design-and-optimizatoin-of-an-adaptive-traffic-signal-control-based-on-Deep-Q-learning/blob/main/controled.mp4) - our system's SUMO simulation  
[output_yolo.avi](https://github.com/eric114610/The-design-and-optimizatoin-of-an-adaptive-traffic-signal-control-based-on-Deep-Q-learning/blob/main/output_yolo.avi) - vehicle detection results video  
