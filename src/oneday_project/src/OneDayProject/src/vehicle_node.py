#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import pickle

import numpy as np
import random
import rospy
import math
import copy
import rospkg
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import scipy.io as sio
import tf
from scipy.stats import norm, multivariate_normal

from geometry_msgs.msg import Twist, Point32, PolygonStamped, Polygon, Vector3, Pose, Quaternion, Point
from visualization_msgs.msg import MarkerArray, Marker

from std_msgs.msg import Float32, Float64, Header, ColorRGBA, UInt8, String, Float32MultiArray, Int32MultiArray
from hmmlearn.hmm import GMMHMM, GaussianHMM

from msgs.msg import dataset_array_msg, dataset_msg, map_array_msg, map_msg, point_msg

## Environments Class
class Environments(object):
    def __init__(self):
        rospy.init_node('Environments')

        self.init_variable() # 변수들 초기화하는 메서드 호출
        self.set_subscriber() # 키보드 입력을 받을 때, cmd_vel이라는 이름을 가지고 구독하게 됨.u:정지. k는 재시작
        self.set_publisher() #네모박스 띄우는 토픽, 각 오브젝트마다,,, 그리고 prediction 정보를 띄우는 토픽으로 이루어짐.
        self.load_map() # 맵 데이터를 로드하는 메소드를 호출


        r = rospy.Rate(20) # 초당 20회의 루프 실행률을 설정
        while not rospy.is_shutdown():

            self.loop()
            r.sleep()


    def load_map(self): # 로드 맵 메소드.
        # 차선의 위치 데이터를 포함하는 넘파이 배열들을 저장한다.
        self.map_file = [] # 여러 맵 파일 이름을 리스트로 정의
        # "ROS" 파라미터 서버에서 map_path를 조회하여 로컬 시스템에서 맵 파일 경로를 가져온다.
        map_path = rospy.get_param("map_path") # 맵 경로 호출 및 저장, 각 맵 파일을 순회하면서 .mat 형식 파일 로드
        matfiles = ["waypoints_0_rev.mat", # 1차선 
                    "waypoints_1_rev.mat", # 2차선
                    "waypoints_2_rev.mat", # 합류차선
                    "waypoints_3_rev.mat" # 분기차선
                    ]

        # 124.20403395607009 738.5587856439931 1356.4053875886939
        station_offset = [0, 0, 0, 0]

        for i, matfile in enumerate(matfiles): # 맵 파일 배열을 순회하며 각 맵 파일을 하나씩 로드한다.
            mat = sio.loadmat(map_path+matfile) # MATLAB 파일을 로드

            easts = mat["east"][0] # 각 파일에서 동쪽, 북쪽, 해당 지점 station 값 추출
            norths = mat["north"][0]
            stations = mat["station"][0]+station_offset[i] # 최종 스테이션 값 조정

            # if i==2:
            self.map_file.append(np.stack([easts, norths, stations],axis=-1)) # 여기에 각 차선에 대한 정보가 저장된다. 글로벌 ests, norths에 저장, 스테이션에 대해 저장


        self.D_list = [ 0, -3.85535188 ,-7.52523438, -7.37178602] # 항상 옆 차선 붙어있기 때문에,,, 그냥 d는 일정하다고,,, frenet에서 편의를 위해서
        self.D_list = np.array(self.D_list)

       # Hidden Markov Model 관련 설정 초기화. 차량 데이터는 pickle 파일로부터 로드
       # 각 차량의 상태 정보를 self.vehicle에 저장
    def init_variable(self):  # 시뮬레이션 일시 저장...
        self.pause = False
        self.time = 11
        self.br = tf.TransformBroadcaster() # 좌표 변환 정보를 다른 노드에게 방송하는 객체 생성, 차량 위치 변화 시각화 또는 다른 노드에서 활용 가능

        SamplePath = rospy.get_param("SamplePath") # 차량 데이터가 저장된 디렉토리 경로 가져온다.
        SampleList = sorted(glob.glob(SamplePath+"/*.pickle")) # 모든 pickle 파일을 찾아 알파벳 순으로 정렬, 차량 시뮬레이션 데이터 포함하고 있다.
        SampleId = (int)(rospy.get_param("SampleId")) # 특정 차량 데이터 파일 인덱스를 가져온다.

        ######################### Load Vehicle #######################

        with open(SampleList[SampleId], 'rb') as f: # 선택된 차량 데이터 파일을 열고 호출해 파일 내용을 역질렬화 한다.. 이는 self.vehicle에 저장된다.
            self.vehicles = pickle.load(f)
        # self.Logging[veh.track_id].append([veh.lane_id, veh.target_lane_id, veh.s,
        #         #                          veh.d, veh.pose[0], veh.pose[1], veh.pose[2], veh.v, veh.yawrate, MODE[veh.mode], veh.ax, veh.steer, veh.length, veh.width])

        ########################## HMM ###############################
        #with open("/home/mmc_ubuntu/Work/system-infra/Simulation/log/model_LC.pickle", 'rb') as f:
        #    self.hmm_lc = pickle.load(f)

        #with open("/home/mmc_ubuntu/Work/system-infra/Simulation/log/model_LK.pickle", 'rb') as f:
        #    self.hmm_lk = pickle.load(f)


    def loop(self): #self.pause가 False인 경우 self.publish와 self.pub_map을 호출

        if self.pause:
            pass
        else:
            self.publish() # 차량 정보를 Ros 토픽을 통해 발행
            self.pub_map() # 맵 데이터를 ROS 토픽을 통해 발행
            self.time+=1

        if self.time>=(len(self.vehicles[0])-1):
            rospy.signal_shutdown("End of the logging Time")
            # asdf


    def callback_plot(self, data):

        if data.linear.x>0 and data.angular.z>0: #u
            self.pause = True
        else:
            self.pause = False

            # 차량 데이터를 'dataset_array_msg' 포맷으로 포장하여 '/history' 토픽에 발행
    def publish(self, is_delete = False):


        ObjectsData = dataset_array_msg()

        for i in range(len(self.vehicles)):


            ObjectData = dataset_msg()
            ObjectData.id = i
            ObjectData.lane_id = self.vehicles[i][self.time][0]
            ObjectData.length = self.vehicles[i][self.time][12]
            ObjectData.width = self.vehicles[i][self.time][13]

            for t in range(self.time-10, self.time+1):
                ObjectData.x.append(self.vehicles[i][t][4])
                ObjectData.y.append(self.vehicles[i][t][5])
                ObjectData.yaw.append(self.vehicles[i][t][6])
                ObjectData.vx.append(self.vehicles[i][t][7])
                ObjectData.s.append(self.vehicles[i][t][2])
                ObjectData.d.append(self.vehicles[i][t][3])


            ObjectsData.data.append(ObjectData)

        self.history_pub.publish(ObjectsData)


    def calculate_probability(self, d_lat, v, lane_width):
        self.v_max=v
        mu_lc=-(2/lane_width)**2 * self.v_max*(d_lat-lane_width/2)**2+self.v_max
        mu_lk=-(2/lane_width)**2 * self.v_max*d_lat**2

        # Variance
        sigma=1.1
        #P(A|C, M)
        # Normal Gaussian Probability Density Function
        p_lc_action = norm.pdf(v, mu_lc, sigma) # 속도에 대한 확률
        p_lk_action = norm.pdf(v, mu_lk, sigma) # 속도에 대한 확률

        #P(M|C)
        p_lk=norm.pdf(d_lat, 0, sigma) # lk 확률
        p_lc=norm.pdf(d_lat, lane_width/2,sigma) # lc 확률

        return p_lc_action, p_lk_action, p_lk, p_lc
            # 차량의 행동 예측 결과를 받아서 처리하고, Rviz에서 시각화를 위한 마커를 생성하여 발행
    def callback_result(self, data):

        Objects = MarkerArray() # 차량 위치를 표시할 마커들과 텍스트 마커들을 저장할 배열을 각각 초기화..
        Texts = MarkerArray()
        for i in range(len(self.vehicles)): # 차량 데이터를 순회.. 시뮬레이션에 포함된 모든 차량에 대해 반복 실행한다.

            veh_data = np.array(self.vehicles[i][self.time-10:self.time+1]) # 현재 시간을 기준으로 각 차량의 과거 10 타임스텝의 데이터를 배열로 추출
            #d_lat=veh_data[-1,3] # Latest d_lat
            lane_width=3.85535188 # Lane Width
            d_lat=abs(veh_data[-1,3])
            if d_lat<lane_width:
                d_lat=d_lat-lane_width/2
            elif lane_width < d_lat and d_lat < 2*lane_width:
                d_lat=d_lat-lane_width*1.5
            elif 2*lane_width < d_lat and d_lat < 3*lane_width:
                d_lat=d_lat-2.5*lane_width

            print("d_lat", d_lat)
            v=veh_data[-1,7]
            p_lc_action, p_lk_action, p_lk, p_lc=self.calculate_probability(d_lat, v, lane_width)
            print("P(M|C): ", p_lc)
            print("P(A|C, M)", p_lc_action)
            """
            To Do
            i번째 veh history data인 veh_data를 활용하여 LC intention에 대한 pred 수행

            """
            # P(A|C,M)P(M|C)/SUM(P(A|C,M)) = P(M|A,C) # Select Maneuver
            pred_lc= p_lc_action*p_lc/(p_lc_action+p_lk_action)
            pred_lk= p_lk_action*p_lk/(p_lk_action+p_lc_action)
            print("prediction of lk", pred_lk)
            print("prediction of LC", pred_lc)


            pred  = "LC" if np.argmax([pred_lk , pred_lc])==1 else "LK"
            # 실제 동작 정보,,, 차량의 실제 동작 값을 기반으로 실제 동작을 결정한다.
            gt = "LC" if self.vehicles[i][self.time][9] == 1 else "LK"
            rospy.loginfo("Vehicle ID: {}, GT: {}, Pred: {}, P_LC: {:.4f}, P_LK: {:.4f}".format(i, gt, pred, pred_lc, pred_lk))
            # 차량의 방향을 포현하기 위한 쿼터니언을 계산한다.
            q = tf.transformations.quaternion_from_euler(0, 0, self.vehicles[i][self.time][6])

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.CUBE

            marker.pose.position.x = self.vehicles[i][self.time][4]
            marker.pose.position.y = self.vehicles[i][self.time][5]
            marker.pose.position.z = 0.5

            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker.scale.x = self.vehicles[i][self.time][12]
            marker.scale.y = self.vehicles[i][self.time][13]
            marker.scale.z = 1
            marker.color.a = 1.0


            Objects.markers.append(marker)


            text = Marker()
            text.header.frame_id = "world"
            text.ns = "text"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING

            text.action = Marker.ADD

            text.color = ColorRGBA(1, 1, 1, 1)
            text.scale.z = 5
            text.text = str(i)+" / True : " + gt+" / Pred : "  + pred
            text.pose.position = Point(self.vehicles[i][self.time][4], self.vehicles[i][self.time][5], 3)

            Texts.markers.append(text)

        self.sur_pose_plot.publish(Objects)
        self.text_plot.publish(Texts)


        self.br.sendTransform((self.vehicles[0][self.time][4], self.vehicles[0][self.time][5], 0),
                                tf.transformations.quaternion_from_euler(0, 0,self.vehicles[0][self.time][6]),
                                rospy.Time.now(),
                                "base_link",
                                "world")

        # 로드된 맵 데이터를 처리하고 Rviz에서 시각화를 위해 발행
    def pub_map(self, is_delete = False):

        MapData = map_array_msg()
        for i in range(len(self.map_file)):
            MapSeg = map_msg()
            MapSeg.path_id = i

            temp = self.map_file[i]
            for j in range(len(temp)):

                point = point_msg()
                point.x = temp[j,0]
                point.y = temp[j,1]
                point.s = temp[j,2]
                point.d = self.D_list[i]
                MapSeg.center.append(point)

            MapData.data.append(MapSeg)
        self.map_pub.publish(MapData)


        Maps = MarkerArray()
        for i in range(len(self.map_file)):
            MapSeg = map_msg()
            MapSeg.path_id = i


            line_strip = Marker()
            line_strip.type = Marker.LINE_STRIP
            line_strip.id = i
            line_strip.scale.x = 2
            line_strip.scale.y = 0.1
            line_strip.scale.z = 0.1

            line_strip.color = ColorRGBA(1.0,1.0,1.0,0.5)
            line_strip.header = Header(frame_id='world')

            temp = self.map_file[i]
            for j in range(len(temp)):
                point = Point()
                point.x = temp[j,0]
                point.y = temp[j,1]
                point.z = 0

                line_strip.points.append(point)

                point = point_msg()
                point.x = temp[j,0]
                point.y = temp[j,1]
                point.s = temp[j,2]
                point.d = self.D_list[i]
                MapSeg.center.append(point)

            Maps.markers.append(line_strip)

        self.map_plot.publish(Maps)
    # 필요한 로스 토픽에 대한 구독자와 발행자를 설정함.
    def set_subscriber(self):
        rospy.Subscriber('/cmd_vel',Twist, self.callback_plot,queue_size=1)
        rospy.Subscriber('/result', dataset_array_msg, self.callback_result, queue_size=1)

    def set_publisher(self):

        self.sur_pose_plot = rospy.Publisher('/rviz/sur_obj_pose', MarkerArray, queue_size=1)
        self.map_plot = rospy.Publisher('/rviz/maps', MarkerArray, queue_size=1)
        self.text_plot = rospy.Publisher('/rviz/text', MarkerArray, queue_size=1)
        self.map_pub = rospy.Publisher('/map_data', map_array_msg, queue_size=1)
        self.history_pub = rospy.Publisher('/history', dataset_array_msg, queue_size=1)


if __name__ == '__main__':

    try:
        f = Environments()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node.')

