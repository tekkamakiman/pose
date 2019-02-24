
# coding: utf-8

import time
import os
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import chainer
from collections import OrderedDict
from pose_detector import draw_person_pose
import threading
import time
import webbrowser


def isKeypoint(point):

    for i in [2,3,4,5,6,7,8,9,10,11,12,13,]:
        if(point[i][2]==0):
            return False
    return True

def degree(point,a,b,c,part):
    #vectoraはbaベクトル
    #vectorbはbcベクトル
    #Trueは根本
    ax=point[a][0]-point[b][0]
    ay=point[a][1]-point[b][1]
    bx=point[c][0]-point[b][0]
    by=point[c][1]-point[b][1]
    vectora=np.array([ax,ay])
    vectorb=np.array([bx,by])
    return vectorcal(vectora,vectorb,part)

def vectorcal(vectora,vectorb,part):
    inner=vectora[0]*vectorb[0]+vectora[1]*vectorb[1]
    cross=vectora[0]*vectorb[1]-vectora[1]*vectorb[0]
    return np.rad2deg(np.arctan2(cross,inner))

def create(point):
    code=""
    arr=point
    #18から22までを追加
    arr=np.append(arr, np.array([[point[8][0], point[8][1]+1, 2]]) ,axis=0)
    arr=np.append(arr,np.array([[point[11][0],point[11][1]+1,2]]),axis=0)
    arr=np.append(arr,np.array([[point[5][0],point[5][1]+1,2]]),axis=0)
    arr=np.append(arr,np.array([[point[2][0],point[2][1]+1,2]]),axis=0)
    arr=np.append(arr,np.array([[(point[8][0]+point[11][0])/2,(point[8][1]+point[11][1])/2,2]]),axis=0)
    BODY=np.rad2deg(np.arctan((arr[22][1]-arr[1][1])/(arr[1][0]-arr[22][0])))
    if(BODY<=0):
        BODY=90-abs(BODY)
    else:
        BODY=BODY-90
    RUA=(-1)*degree(arr,21,2,3,"RUA")-BODY
    RLA=180-degree(arr,2,3,4,"RLA")
    LUA=(-1)*degree(arr,20,5,6,"LUA")-BODY
    LLA=180-degree(arr,5,6,7,"LLA")
    RUL=(-1)*degree(arr,18,8,9,"RUL")-BODY
    RLL=180-degree(arr,8,9,10,"RLL")
    LUL=(-1)*degree(arr,19,11,12,"LUL")-BODY
    LLL=180-degree(arr,11,12,13,"LLL")
    theta = OrderedDict()
    theta['BODY']=BODY
    theta["LUA"]=LUA
    theta["LLA"]=LLA
    theta["RUA"]=RUA
    theta["RLA"]=RLA
    theta["LUL"]=threshold(LUL,11)
    theta["LLL"]=threshold(LLL,10)
    theta["RUL"]=threshold(RUL,11)
    theta["RLL"]=threshold(RLL,10)
#     theta={"LUA":LUA,"LLA":LLA,"RUA":RUA,"RLA":RLA,"LUL":LUL,"LLL":LLL,"RUL":RUL,"RLL":RLL}
    for i in theta:
        if(theta[i]>180):
            theta[i]=theta[i]-360
            #codeの追加
        code=code+"R "+str(i)+" "+str(round((-1)*theta[i]))+"%0D%0A"
    return code

def paint(a):
    code=""
    pose_arr=a
    code=create(pose_arr)
    return code

def distance(arr,a,b):
    c=np.sqrt(np.square(arr[a][0]-arr[b][0])+np.square(arr[a][1]-arr[b][1]))
    return c

def basis(par):
    global armave
    global legave
    global basisT
    img = par
    #時間かかる
    pose_arr = model(img)
    pose_arr=pose_arr[0]
    pose_arr=pose_arr[0]
    pose_arr=np.append(pose_arr,np.array([[(pose_arr[8][0]+pose_arr[11][0])/2,(pose_arr[8][1]+pose_arr[11][1])/2,2]]),axis=0)

    if(isKeypoint(pose_arr)):
        Rzenwan=distance(pose_arr,3,4)
        Rjouwan=distance(pose_arr,2,3)
        Lzenwan=distance(pose_arr,6,7)
        Ljouwan=distance(pose_arr,5,6)
        Rdaitai=distance(pose_arr,8,9)
        Rkatai=distance(pose_arr,9,10)
        Ldaitai=distance(pose_arr,11,12)
        Lkatai=distance(pose_arr,12,13)
        leg={"Rdaitai":Rdaitai,"Rkatai":Rkatai,"Ldaitai":Ldaitai,"Lkatai":Lkatai}
        arm={"Rzenwan":Rzenwan,"Rjouwan":Rjouwan,"Lzenwan":Lzenwan,"Ljouwan":Ljouwan}
        for i in leg:
            legave[i]=leg[i]
        for j in arm:
            armave[j]=arm[j]
    basisT=True


def isSide(pose_arr,armave,legave):
    RJcos=np.rad2deg(np.cos(np.sqrt(np.square(pose_arr[2][0]-pose_arr[3][0])+np.square(pose_arr[2][1]-pose_arr[3][1]))/armave['Rjouwan']))
    LJcos=np.rad2deg(np.cos(np.sqrt(np.square(pose_arr[5][0]-pose_arr[6][0])+np.square(pose_arr[5][1]-pose_arr[6][1]))/armave['Ljouwan']))
#     RDcos=np.rad2deg(np.cos(np.sqrt(np.square(pose_arr[8][0]-pose_arr[9][0])+np.square(pose_arr[8][1]-pose_arr[9][1]))/legave['Rdaitai']))
#     LDcos=np.rad2deg(np.cos(np.sqrt(np.square(pose_arr[11][0]-pose_arr[12][0])+np.square(pose_arr[11][1]-pose_arr[12][1]))/legave['Ldaitai']))
    RDcos=0
    LDcos=0
    for i in [RJcos,LJcos,RDcos,LDcos]:
        if(i>=45):
            return True
    return False

def n2p(dic,point,arr,a,b):
    t=np.square(dic[point])-(np.square(arr[a][0]-arr[b][0])+np.square(arr[a][1]-arr[b][1]))
    if(np.square(dic[point])-(np.square(arr[a][0]-arr[b][0])+np.square(arr[a][1]-arr[b][1]))<=0):
        return 0
    return np.sqrt(t)

def threshold(a,deg=20):
    if(abs(a)<deg):
        return 0
    return a

def zaxis(basica,basicl,pose_arr):
    pose_arr=pose_arr
    pose_arr=np.append(pose_arr,np.array([[(pose_arr[8][0]+pose_arr[11][0])/2,(pose_arr[8][1]+pose_arr[11][1])/2,2]]),axis=0)
    z={}
    Rzenwan=distance(pose_arr,3,4)
    Rjouwan=distance(pose_arr,2,3)
    Lzenwan=distance(pose_arr,6,7)
    Ljouwan=distance(pose_arr,5,6)
    basicarm={"Rzenwan":Rzenwan,"Rjouwan":Rjouwan,"Lzenwan":Lzenwan,"Ljouwan":Ljouwan}
    z["zRzenwan"]=n2p(basica,"Rzenwan",pose_arr,3,4)
    z["zRjouwan"]=n2p(basica,"Rjouwan",pose_arr,2,3)
    z["zLzenwan"]=n2p(basica,"Lzenwan",pose_arr,6,7)
    z["zLjouwan"]=n2p(basica,"Ljouwan",pose_arr,5,6)
    z["zRdaitai"]=n2p(basicl,"Rdaitai",pose_arr,8,9)
    z["zRkatai"]=n2p(basicl,"Rkatai",pose_arr,9,10)
    z["zLdaitai"]=n2p(basicl,"Ldaitai",pose_arr,11,12)
    z["zLkatai"]=n2p(basicl,"Lkatai",pose_arr,12,13)
    #zdeg={}
    zdeg["RUA"]=arctandegree(pose_arr,3,2,z,"zRjouwan")
    zdeg["RLA"]=threshold(arctandegree(pose_arr,4,3,z,"zRzenwan")-zdeg["RUA"],30)
    zdeg["LUA"]=arctandegree(pose_arr,6,5,z,"zRzenwan")
    zdeg["LLA"]=threshold(arctandegree(pose_arr,7,6,z,"zRzenwan")-zdeg["LUA"],30)
    zdeg["RUL"]=arctandegree(pose_arr,9,8,z,"zRdaitai")
    zdeg["RLL"]=(-1)*threshold(arctandegree(pose_arr,10,9,z,"zRkatai")-zdeg["RUL"],20)
    zdeg["LUL"]=arctandegree(pose_arr,12,11,z,"zLdaitai")
    zdeg["LLL"]=(-1)*threshold(arctandegree(pose_arr,13,12,z,"zLkatai")-zdeg["LUL"],20)
    return zdeg

def arctandegree(arr,a,b,z,part):
    y=arr[a][1]-arr[b][1]
    z=z[part]
    return np.rad2deg(np.arctan2(z,y))

def CreateSide(zdeg):
    code="SD"+"%0D%0A"
    for g in zdeg:
        code=code+"R "+str(g)+" "+str(round(zdeg[g]))+"%0D%0A"
    return code


# モジュール検索パスの設定
REPO_ROOT = ''

# PoseDetectorクラスのインポート
from pose_detector import PoseDetector

armave={'Rzenwan': 63, 'Lzenwan': 64, 'Ljouwan': 67, 'Rjouwan': 66}
legave={'Ldaitai': 96, 'Lkatai': 86, 'Rdaitai': 95, 'Rkatai': 90}

# モデルのロード
arch_name = 'posenet'
image_path = os.path.join(REPO_ROOT, 'data', 'person.png')
weight_path = os.path.join(REPO_ROOT, 'models', 'coco_posenet.npz')
model = PoseDetector(arch_name, weight_path)


#1はUSBカメラ、0は内蔵カメラ
cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture(1)
q=0
timer=0
w=0
code=""
isBasis=True
zdeg={}
counting=1
basisT=False
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #cv2.StartWindowThread()
    # Display the resulting frame
    cv2.imshow('frame', frame)
    #Escで抜ける
    key = cv2.waitKey(1) & 0xFF

    #if(isBasis):
    if(key == ord('f')):
        th=threading.Thread(target=basis,name='th',args=(np.array(frame, dtype=np.float32),))
        th.setDaemon(True)
        th.start()

    #緊急用
    if(key == ord('q')):
        break

    if(key == ord('s')):
        print("start")
        img=np.array(frame, dtype=np.float32)
        cv2.imwrite("photo1.png",img)
        pose_arr = model(img)
        pose_arr=pose_arr[0]
        result_img = draw_person_pose(img, pose_arr)
        plt.figure(figsize=(6, 6))
        plt.imshow(255-result_img[:, :, ::-1])
        pose_arr=pose_arr[0]
        if(isKeypoint(pose_arr)):
            while(basisT==False):
                counting=counting+1
            if(isSide(pose_arr,armave,legave)):
                zdeg=zaxis(armave,legave,pose_arr)
                code=CreateSide(zdeg)
            else:
                code=paint(pose_arr)
        else:
            code="SAY 'キーポイントが足りません' 5"
        break

cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
url = "http://127.0.0.1:8000/?code="+code
webbrowser.open(url)
