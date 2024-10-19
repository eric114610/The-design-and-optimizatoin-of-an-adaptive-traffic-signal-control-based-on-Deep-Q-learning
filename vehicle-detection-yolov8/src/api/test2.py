from ultralytics import YOLO
import numpy as np
import cv2
import random

def generate_random_numbers():
    while True:
        numbers = [random.randint(0, 4) for _ in range(3)]
        total_sum = sum(numbers)
        if 1 < total_sum < 15:
            return sorted(numbers, reverse=True), total_sum
        
print("A")
model = YOLO("yolov8x.pt")

print("B")
# Define the classes we want to detect
# COCO dataset class indices:
# car: 2, motorcycle: 3, bus: 5, truck: 7
classes_to_detect = [2, 5, 7]

# Define region of interest (ROI) - example coordinates
# Format: [x1, y1, x2, y2] for the rectangle
# You can adjust these coordinates based on your needs
roi = [100, 100, 600, 400]  # example coordinates

# Read the image first to apply ROI
img = cv2.imread("./3.png")
# Create a mask for ROI
mask = np.zeros(img.shape[:2], dtype=np.uint8)
mask[roi[1]:roi[3], roi[0]:roi[2]] = 255

# Apply mask to image
roi_img = img.copy()
roi_img[mask == 0] = 0

# Save ROI image temporarily (optional, for debugging)
# cv2.imwrite("roi_debug.png", roi_img)

# Run inference with class filtering
results = model(roi_img, classes=classes_to_detect)

cv2.imwrite("./out_low.png", results[0].plot())
################################################

import time

numbers = list(range(1, 11))
# Shuffle the list to get a random permutation
total_time=0

while True:

    with open("../../../Intersection-Traffic-Light-Control-System/TLCS/close.txt", "r") as C:
        content = C.read()
        if content == "END":
            break

    total_time = 0
        
    start = time.time()

    img = cv2.imread("4.png")

    results1 = model(img, classes=classes_to_detect)

    # Initialize counters
    vehicle_counts1 = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]

    vehicle_coordinates = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }

    # Process results
    for r in results1[0].boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])

        box = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < 1980 and (-2.5*center_x+5850>center_y) and (-0.8737*center_x+2630<center_y):
            # 1140,3000 1980,900  0,2630
            # Map YOLO class indices to our list indices
            # if cls == 2:    # car
            #     vehicle_counts1[0] += 1
            # elif cls == 5:  # bus
            #     vehicle_counts1[1] += 1
            # elif cls == 7:  # truck
            #     vehicle_counts1[2] += 1
            # vehicle_counts1[3] = vehicle_counts1[0] + vehicle_counts1[1] + vehicle_counts1[2]
            if cls == 2:    # car
                vehicle_counts1[0] += 1
                vehicle_coordinates['car'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 5:  # bus
                vehicle_counts1[1] += 1
                vehicle_coordinates['bus'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 7:  # truck
                vehicle_counts1[2] += 1
                vehicle_coordinates['truck'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            # elif cls == 3:  # motorcycle
            #     vehicle_counts1[3] += 1
            #     vehicle_coordinates['motorcycle'].append({
            #         'bbox': (x1, y1, x2, y2),
            #         'center': (center_x, center_y),
            #         'confidence': conf
            #     })
            vehicle_counts1[3] = vehicle_counts1[0] + vehicle_counts1[1] + vehicle_counts1[2]


    print(f"Vehicle counts 1 [car, bus, truck, total]: {vehicle_counts1}")

    total_time += time.time() - start
    

    # Plot and save the results
    res_plotted1 = results1[0].plot()
    res_plotted2 = results1[0].plot()

    # Draw ROI rectangle on the result (optional)
    # cv2.rectangle(res_plotted1, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
    for vehicle_type, coords in vehicle_coordinates.items():
        for detection in coords:
            center = detection['center']
            cv2.circle(res_plotted1, center, 15, (0, 255, 255), -1)
            cv2.putText(res_plotted1, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 12)

    cv2.imwrite("./out.png", res_plotted1)



    start = time.time()

    vehicle_counts2 = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]

    vehicle_coordinates = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }

    # Process results
    for r in results1[0].boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])

        box = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < 2000 and (-2.5*center_x+5850<center_y):
            # Map YOLO class indices to our list indices
            # if cls == 2:    # car
            #     vehicle_counts2[0] += 1
            # elif cls == 5:  # us
            #     vehicle_counts2[1] += 1
            # elif cls == 7:  # truck
            #     vehicle_counts2[2] += 1
            # vehicle_counts2[3] = vehicle_counts2[0] + vehicle_counts2[1] + vehicle_counts2[2]
            if cls == 2:    # car
                vehicle_counts2[0] += 1
                vehicle_coordinates['car'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 5:  # bus
                vehicle_counts2[1] += 1
                vehicle_coordinates['bus'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 7:  # truck
                vehicle_counts2[2] += 1
                vehicle_coordinates['truck'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })

            vehicle_counts2[3] = vehicle_counts2[0] + vehicle_counts2[1] + vehicle_counts2[2]


    print(f"Vehicle counts 2 [car, bus, truck, total]: {vehicle_counts2}")

    total_time += time.time() - start

    for vehicle_type, coords in vehicle_coordinates.items():
        for detection in coords:
            center = detection['center']
            cv2.circle(res_plotted2, center, 15, (0, 255, 255), -1)
            cv2.putText(res_plotted2, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 12)
            
    cv2.imwrite("./out2.png", res_plotted2)



    ##############################################################

    start = time.time()

    img = cv2.imread("6.png")


    results2 = model(img, classes=classes_to_detect)

    # Initialize counters
    vehicle_counts3 = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]

    vehicle_coordinates = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }

    # Process results
    for r in results2[0].boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])

        box = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < 1980 and (-2.5*center_x+5850>center_y) and (-0.8737*center_x+2630<center_y):
            # Map YOLO class indices to our list indices
            if cls == 2:    # car
                vehicle_counts3[0] += 1
                vehicle_coordinates['car'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 5:  # bus
                vehicle_counts3[1] += 1
                vehicle_coordinates['bus'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 7:  # truck
                vehicle_counts3[2] += 1
                vehicle_coordinates['truck'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            vehicle_counts3[3] = vehicle_counts3[0] + vehicle_counts3[1] + vehicle_counts3[2]

    print(f"Vehicle counts 3 [car, bus, truck, total]: {vehicle_counts3}")

    total_time += time.time() - start

    res_plotted3 = results2[0].plot()
    res_plotted4 = results2[0].plot()

    for vehicle_type, coords in vehicle_coordinates.items():
        for detection in coords:
            center = detection['center']
            cv2.circle(res_plotted3, center, 15, (0, 255, 255), -1)
            cv2.putText(res_plotted3, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 12)
            
    cv2.imwrite("./out3.png", res_plotted3)

    start = time.time()


    vehicle_counts4 = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]

    vehicle_coordinates = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }

    # Process results
    for r in results2[0].boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])

        box = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < 2000 and (-2.5*center_x+5850<center_y):
            # Map YOLO class indices to our list indices
            if cls == 2:    # car
                vehicle_counts4[0] += 1
                vehicle_coordinates['car'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 5:  # bus
                vehicle_counts4[1] += 1
                vehicle_coordinates['bus'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 7:  # truck
                vehicle_counts4[2] += 1
                vehicle_coordinates['truck'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            vehicle_counts4[3] = vehicle_counts4[0] + vehicle_counts4[1] + vehicle_counts4[2]

    print(f"Vehicle counts 4 [car, bus, truck, total]: {vehicle_counts4}")

    total_time += time.time() - start

    for vehicle_type, coords in vehicle_coordinates.items():
        for detection in coords:
            center = detection['center']
            cv2.circle(res_plotted4, center, 15, (0, 255, 255), -1)
            cv2.putText(res_plotted4, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 12)
    
    cv2.imwrite("./out4.png", res_plotted4)

    #################################################
    start = time.time()

    img = cv2.imread("7.png")

    results3 = model(img, classes=classes_to_detect)

    # Initialize counters
    vehicle_counts5 = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]

    vehicle_coordinates = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }
    # Process results
    for r in results3[0].boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])

        box = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < 560 and (-3.11111*center_x+1862>center_y) and (-0.7143*center_x+520<center_y):
        # 560,120  380,680  0,520
            # Map YOLO class indices to our list indices
            if cls == 2:    # car
                vehicle_counts5[0] += 1
                vehicle_coordinates['car'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 5:  # bus
                vehicle_counts5[1] += 1
                vehicle_coordinates['bus'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 7:  # truck
                vehicle_counts5[2] += 1
                vehicle_coordinates['truck'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            vehicle_counts5[3] = vehicle_counts5[0] + vehicle_counts5[1] + vehicle_counts5[2]

    print(f"Vehicle counts 5 [car, bus, truck, total]: {vehicle_counts5}")

    total_time += time.time() - start

    res_plotted5 = results3[0].plot()
    res_plotted6 = results3[0].plot()

    for vehicle_type, coords in vehicle_coordinates.items():
        for detection in coords:
            center = detection['center']
            cv2.circle(res_plotted5, center, 5, (0, 255, 255), -1)
            cv2.putText(res_plotted5, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("./out5.png", res_plotted5)

    start = time.time()

    vehicle_counts6 = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]

    vehicle_coordinates = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }

    # Process results
    for r in results3[0].boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])

        box = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < 560 and (-3.11111*center_x+1862<center_y):
        
            # Map YOLO class indices to our list indices
            if cls == 2:    # car
                vehicle_counts6[0] += 1
                vehicle_coordinates['car'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 5:  # bus
                vehicle_counts6[1] += 1
                vehicle_coordinates['bus'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 7:  # truck
                vehicle_counts6[2] += 1
                vehicle_coordinates['truck'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            vehicle_counts6[3] = vehicle_counts6[0] + vehicle_counts6[1] + vehicle_counts6[2]

    print(f"Vehicle counts 6 [car, bus, truck, total]: {vehicle_counts6}")

    total_time += time.time() - start

    for vehicle_type, coords in vehicle_coordinates.items():
        for detection in coords:
            center = detection['center']
            cv2.circle(res_plotted6, center, 5, (0, 255, 255), -1)
            cv2.putText(res_plotted6, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("./out6.png", res_plotted6)


    #######################################

    start = time.time()

    img = cv2.imread("8.png")

    results4 = model(img, classes=classes_to_detect)

    vehicle_counts7 = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]

    vehicle_coordinates = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }

    # Process results
    for r in results4[0].boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])

        box = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < 560 and (-3.11111*center_x+1862>center_y) and (-0.7143*center_x+520<center_y):
        # 560,120  380,680  0,520
            # Map YOLO class indices to our list indices
            if cls == 2:    # car
                vehicle_counts7[0] += 1
                vehicle_coordinates['car'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 5:  # bus
                vehicle_counts7[1] += 1
                vehicle_coordinates['bus'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 7:  # truck
                vehicle_counts7[2] += 1
                vehicle_coordinates['truck'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            vehicle_counts7[3] = vehicle_counts7[0] + vehicle_counts7[1] + vehicle_counts7[2]

    print(f"Vehicle counts 7 [car, bus, truck, total]: {vehicle_counts7}")

    total_time += time.time() - start

    res_plotted7 = results4[0].plot()
    res_plotted8 = results4[0].plot()

    for vehicle_type, coords in vehicle_coordinates.items():
        for detection in coords:
            center = detection['center']
            cv2.circle(res_plotted7, center, 5, (0, 255, 255), -1)
            cv2.putText(res_plotted7, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("./out7.png", res_plotted7)


    # Initialize counters
    start = time.time()

    vehicle_counts8 = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]

    vehicle_coordinates = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }

    # Process results
    for r in results4[0].boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])

        box = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < 560 and (-3.11111*center_x+1862<center_y):
        
            # Map YOLO class indices to our list indices
            if cls == 2:    # car
                vehicle_counts8[0] += 1
                vehicle_coordinates['car'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 5:  # bus
                vehicle_counts8[1] += 1
                vehicle_coordinates['bus'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 7:  # truck
                vehicle_counts8[2] += 1
                vehicle_coordinates['truck'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            vehicle_counts8[3] = vehicle_counts8[0] + vehicle_counts8[1] + vehicle_counts8[2]

    print(f"Vehicle counts 8 [car, bus, truck, total]: {vehicle_counts8}")

    total_time += time.time() - start

    for vehicle_type, coords in vehicle_coordinates.items():
        for detection in coords:
            center = detection['center']
            cv2.circle(res_plotted8, center, 5, (0, 255, 255), -1)
            cv2.putText(res_plotted8, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("./out8.png", res_plotted8)


    #######################################

    start = time.time()

    img = cv2.imread("9.png")

    results5 = model(img, classes=classes_to_detect)

    vehicle_counts9 = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]

    vehicle_coordinates = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }

    # Process results
    for r in results5[0].boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])

        box = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < 650 and (-2.354*center_x+1595>center_y) and (-0.83046*center_x+637<center_y):
        # 560,120  380,680  0,520
            # Map YOLO class indices to our list indices
            if cls == 2:    # car
                vehicle_counts9[0] += 1
                vehicle_coordinates['car'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 5:  # bus
                vehicle_counts9[1] += 1
                vehicle_coordinates['bus'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 7:  # truck
                vehicle_counts9[2] += 1
                vehicle_coordinates['truck'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            vehicle_counts9[3] = vehicle_counts9[0] + vehicle_counts9[1] + vehicle_counts9[2]

    print(f"Vehicle counts 9 [car, bus, truck, total]: {vehicle_counts9}")

    total_time += time.time() - start

    res_plotted10 = results5[0].plot()
    res_plotted9 = results5[0].plot()

    for vehicle_type, coords in vehicle_coordinates.items():
        for detection in coords:
            center = detection['center']
            cv2.circle(res_plotted9, center, 5, (0, 255, 255), -1)
            cv2.putText(res_plotted9, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("./out9.png", res_plotted9)


    # Initialize counters
    start = time.time()

    vehicle_counts10 = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]

    vehicle_coordinates = {
        'car': [],
        'bus': [],
        'truck': [],
        'motorcycle': []
    }

    # Process results
    for r in results5[0].boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])

        box = r.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        
        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_x < 650 and (-2.354*center_x+1595<center_y):
        
            # Map YOLO class indices to our list indices
            if cls == 2:    # car
                vehicle_counts10[0] += 1
                vehicle_coordinates['car'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 5:  # bus
                vehicle_counts10[1] += 1
                vehicle_coordinates['bus'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            elif cls == 7:  # truck
                vehicle_counts10[2] += 1
                vehicle_coordinates['truck'].append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })
            vehicle_counts10[3] = vehicle_counts10[0] + vehicle_counts10[1] + vehicle_counts10[2]

    print(f"Vehicle counts 10 [car, bus, truck, total]: {vehicle_counts10}")

    total_time += time.time() - start

    for vehicle_type, coords in vehicle_coordinates.items():
        for detection in coords:
            center = detection['center']
            cv2.circle(res_plotted10, center, 5, (0, 255, 255), -1)
            cv2.putText(res_plotted10, f"({center[0]}, {center[1]})", 
                        (center[0] + 10, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("./out10.png", res_plotted10)


    print("Total time:", total_time)
    

    random.shuffle(numbers)


    with open("../../../Intersection-Traffic-Light-Control-System/TLCS/test.txt", "w") as file:
        for i in numbers[:-2]:

            if i==1:
                for item in vehicle_counts1:
                    file.write(f"{item}\n")
            elif i==2:
                for item in vehicle_counts2:
                    file.write(f"{item}\n")

            if i==3:
                for item in vehicle_counts3:
                    file.write(f"{item}\n")
            elif i==4:
                for item in vehicle_counts4:
                    file.write(f"{item}\n")

            if i==5:
                for item in vehicle_counts5:
                    file.write(f"{item}\n")
            elif i==6:
                for item in vehicle_counts6:
                    file.write(f"{item}\n")

            if i==7:
                for item in vehicle_counts7:
                    file.write(f"{item}\n")
            elif i==8:
                for item in vehicle_counts8:
                    file.write(f"{item}\n")
            # random_numbers, total_sum = generate_random_numbers()
            # for item in random_numbers:
            #     file.write(f"{item}\n")
            # file.write(f"{total_sum}\n")
            # random_numbers, total_sum = generate_random_numbers()
            # for item in random_numbers:
            #     file.write(f"{item}\n")
            # file.write(f"{total_sum}\n")
            # random_numbers, total_sum = generate_random_numbers()
            # for item in random_numbers:
            #     file.write(f"{item}\n")
            # file.write(f"{total_sum}\n")
            # random_numbers, total_sum = generate_random_numbers()
            # for item in random_numbers:
            #     file.write(f"{item}\n")
            # file.write(f"{total_sum}\n")
            # random_numbers, total_sum = generate_random_numbers()
            # for item in random_numbers:
            #     file.write(f"{item}\n")
            # file.write(f"{total_sum}\n")
            # random_numbers, total_sum = generate_random_numbers()
            # for item in random_numbers:
            #     file.write(f"{item}\n")
            # file.write(f"{total_sum}\n")
            # random_numbers, total_sum = generate_random_numbers()
            # for item in random_numbers:
            #     file.write(f"{item}\n")
            # file.write(f"{total_sum}\n")
            # random_numbers, total_sum = generate_random_numbers()
            # for item in random_numbers:
            #     file.write(f"{item}\n")
            # file.write(f"{total_sum}\n")
    break



















# video_path = "./input.mp4"
# cap = cv2.VideoCapture(video_path)

# # Get video properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # output_path = './output.mp4'
# # output = cv2.VideoWriter(output_path, 
# #                         cv2.VideoWriter_fourcc(*'mp4v'),
# #                         fps, 
# #                         (frame_width, frame_height))
# output_path = 'output.avi'  # Note: changed to .avi
# output = cv2.VideoWriter(
#     output_path,
#     cv2.VideoWriter_fourcc(*'XVID'),
#     fps,
#     (frame_width, frame_height)
# )

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break
        
#     # Create mask for ROI
    
#     # Run inference
#     results = model(frame, classes=classes_to_detect)
    
#     # Initialize vehicle counts and coordinates for this frame
#     vehicle_counts = [0, 0, 0, 0]  # [num_car, num_bus, num_truck, num_motor]
#     vehicle_coordinates = {
#         'car': [],
#         'bus': [],
#         'truck': [],
#         'motorcycle': []
#     }
    
#     # Process detections
#     for r in results[0].boxes:
#         cls = int(r.cls[0])
#         conf = float(r.conf[0])
#         box = r.xyxy[0].cpu().numpy()
#         x1, y1, x2, y2 = map(int, box)
#         center_x = (x1 + x2) // 2 * 1920//1280
#         center_y = (y1 + y2) // 2 * 1920//1280
#         center_x_f = (x1 + x2) // 2
#         center_y_f = (y1 + y2) // 2
        
#         # Store information based on class

#         if center_x < 1010 and (-1.243*center_x+1443 < center_y):
#             if cls == 2:    # car
#                 vehicle_counts[0] += 1
#                 vehicle_coordinates['car'].append({
#                     'bbox': (x1, y1, x2, y2),
#                     'center': (center_x_f, center_y_f),
#                     'confidence': conf
#                 })
#             elif cls == 5:  # bus
#                 vehicle_counts[1] += 1
#                 vehicle_coordinates['bus'].append({
#                     'bbox': (x1, y1, x2, y2),
#                     'center': (center_x_f, center_y_f),
#                     'confidence': conf
#                 })
#             elif cls == 7:  # truck
#                 vehicle_counts[2] += 1
#                 vehicle_coordinates['truck'].append({
#                     'bbox': (x1, y1, x2, y2),
#                     'center': (center_x_f, center_y_f),
#                     'confidence': conf
#                 })
#             elif cls == 3:  # motorcycle
#                 vehicle_counts[3] += 1
#                 vehicle_coordinates['motorcycle'].append({
#                     'bbox': (x1, y1, x2, y2),
#                     'center': (center_x_f, center_y_f),
#                     'confidence': conf
#                 })
    
#     # Draw results on frame
#     res_plotted = results[0].plot()
    
#     # Draw ROI rectangle
#     # cv2.rectangle(res_plotted, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
    
#     # Draw vehicle counts
#     text = f"Cars: {vehicle_counts[0]} Bus: {vehicle_counts[1]} Trucks: {vehicle_counts[2]} Motorcycles: {vehicle_counts[3]}"
#     cv2.putText(res_plotted, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
#     # Draw center points and coordinates
#     for vehicle_type, coords in vehicle_coordinates.items():
#         for detection in coords:
#             center = detection['center']
#             cv2.circle(res_plotted, center, 5, (0, 255, 255), -1)
#             cv2.putText(res_plotted, f"({center[0]}, {center[1]})", 
#                         (center[0] + 10, center[1]), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
#     # Write frame to output video
#     output.write(res_plotted)
    
    

# # Release everything
# cap.release()
# output.release()