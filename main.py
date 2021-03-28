#!/usr/bin/env python
# manual

# PLEASE READ README.md FILE FOR RUNNING THIS CODE.
# https://drive.google.com/file/d/1KjM74c_2LvWHl2namujDJYp4JZWCheA8/view
import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
import cv2
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import os
from PIL import Image
from stop_sign_utils import *
from astar import AStarAgent

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="abc")
parser.add_argument("--no-pause", action="store_true", help="don't pause on failure")
args = parser.parse_args()

agent = AStarAgent()
path = None
env = DuckietownEnv(map_name=args.map_name, domain_rand=False, draw_bbox=False)

obs = env.reset()
env.render()

total_reward = 0

tmp_dist = 0
tmp_angle = 0

start_time = time.time()
start_time2 = time.time()
take_psg = False
tmp_time = 0

MIN_MATCH_COUNT = 20
FLANN_INDEX_KDTREE = 0
k_p = 10
k_p_1 = 15
k_d = 1
k_d_1 = 0.5

take_psg = True

detector = cv2.xfeatures2d.SIFT_create()
flannParam = dict(algorithm = FLANN_INDEX_KDTREE,tree =5)
flann = cv2.FlannBasedMatcher(flannParam,{})
trainImg = cv2.imread("proto3.png", 0) 
trainKP, trainDecs = detector.detectAndCompute(trainImg,None)

while True:
    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    diff_center = lane_pose.dist
    angle_straight_rads = lane_pose.angle_rad

    speed = 0.2  
    timer = time.time() - tmp_time

    direction = (k_p * diff_center + k_d * angle_straight_rads + k_p_1 *\
                (diff_center - tmp_dist) / timer + k_d_1 * \
                (angle_straight_rads - tmp_angle) / timer ) 

    tmp_time = time.time()
    tmp_dist = diff_center
    tmp_angle = angle_straight_rads

    obs, reward, done, info = env.step([speed, direction])
    total_reward += reward

    im = Image.fromarray(obs)
    img = im.convert('RGB') 
    cv_img = np.array(img)
    cv_img = cv_img[:, :, ::-1].copy() 

    cv_img_BGR = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    cv_img_grayScaled = cv2.cvtColor(cv_img_BGR, cv2.COLOR_BGR2GRAY) 
    queryKP, queryDesc = detector.detectAndCompute(cv_img_grayScaled, None)
    matches = flann.knnMatch(queryDesc, trainDecs, k=2) 
    if env.step_count == 550:
        agent.remove_grid_from_maze(2,3)
        path = agent.find_new_path((1,3),(5,5))
    highMatch = []
    for m, n in matches:
        if (m.distance < 0.65 * n.distance):
            highMatch.append(m) 
    print("match len:", len(highMatch))
    if (len(highMatch) > MIN_MATCH_COUNT):
        while time.time() - start_time < 5: #wait 5 sec. go 5 sec.
            tp = []
            qp = []
            for m in highMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp, qp = np.float32((tp,qp))
            H,status = cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
            h,w = trainImg.shape
            trainingBorder = np.float32([[[0,0],[0,h-1],[w-1,h-1],[0,w-1]]])
            queryBorder = cv2.perspectiveTransform(trainingBorder,H)
            if take_psg:
                agent.remove_grid_from_maze(3,2)
                path = agent.find_new_path((3,1),(5,5))
                cv2.putText(cv_img_BGR, "Arrived to Passenger", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                cv2.putText(cv_img_BGR, "Pick-up Point", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                cv2.putText(cv_img_BGR, "Path is planning...", (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                cv2.polylines(cv_img_BGR,[np.int32(queryBorder)],True,(255,0,0),5)
                cv_img_RGB= cv2.cvtColor(cv_img_BGR, cv2.COLOR_BGR2RGB)
                cv2.imshow('Sign Detection Result', cv_img_RGB)
                cv2.waitKey(3)
                take_psg = False
                time.sleep(5)
            else:
                cv2.putText(cv_img_BGR, "Arrived to Passenger", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                cv2.putText(cv_img_BGR, "Drop-off Point", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                cv2.polylines(cv_img_BGR,[np.int32(queryBorder)],True,(255,0,0),5)
                cv_img_RGB= cv2.cvtColor(cv_img_BGR, cv2.COLOR_BGR2RGB)
                cv2.imshow('Sign Detection Results', cv_img_RGB)
                cv2.waitKey(3)
                time.sleep(5)

            start_time2 = time.time()
            while time.time() - start_time2 < 1.5:
                print("Now continuing with planner")
                wheel_distance = 0.102
                min_rad = 0.08
                action = np.array( [0.0, 0.0])   
                action += np.array([0.20, 0])

                v1 = action[0]
                v2 = action[1]
                # Limit radius of curvature
                if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
                    # adjust velocities evenly such that condition is fulfilled
                    delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
                    v1 += delta_v
                    v2 -= delta_v
                action[0] = v1
                action[1] = v2
                obs, reward, done, info = env.step(action)
                env.render()
                print("Planner ended")
                    
        start_time2 = time.time()
        start_time = time.time()

    else:
        print("Sign not detected yet! - %d/%d") #%(len(highMatch),MIN_MATCH_COUNT)
    cv_img_RGB = cv2.cvtColor(cv_img_BGR, cv2.COLOR_BGR2RGB)
    if path is not None:
        print(path)
        cv2.putText(cv_img_RGB, str("Generated path is:"), (0, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
        cv2.putText(cv_img_RGB, str(path[:len(path)//2]), (0, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
        cv2.putText(cv_img_RGB, str(path[len(path)//2:]), (0, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255))
    cv2.imshow('A* planning', cv_img_RGB)
    cv2.waitKey(1)

    env.render()
