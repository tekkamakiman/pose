
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
import webbrowser
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

def second_HMS(time):
    hour=int(time/3600)
    time=time-3600*hour
    minute=int(time/60)
    time=time-60*minute
    string=str(hour)+"時間"+str(minute)+"分"+str(time)+"秒"
    return string
#------------------ここからHumanPictogrammingの関数-------------------------------
def isKeypoint(point):
    for i in [2,3,4,5,6,7,8,9,10,11,12,13,]:
        if(point[i][2]==0):
            print("キーポイントが足りません")
            return False
    return True

def degree(point,a,b,c):
    #vectoraはbaベクトル
    #vectorbはbcベクトル
    #Trueは根本
    ax=point[a][0]-point[b][0]
    ay=point[a][1]-point[b][1]
    bx=point[c][0]-point[b][0]
    by=point[c][1]-point[b][1]
    vectora=np.array([ax,ay])
    vectorb=np.array([bx,by])
    return vectorcal(vectora,vectorb)

def vectorcal(vectora,vectorb):
    inner=vectora[0]*vectorb[0]+vectora[1]*vectorb[1]
    cross=vectora[0]*vectorb[1]-vectora[1]*vectorb[0]
    return np.rad2deg(np.arctan2(cross,inner))

def create(point,isImage,part=""):
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
    RUA=(-1)*degree(arr,21,2,3)-BODY
    RLA=180-degree(arr,2,3,4)
    LUA=(-1)*degree(arr,20,5,6)-BODY
    LLA=180-degree(arr,5,6,7)
    RUL=(-1)*degree(arr,18,8,9)-BODY
    RLL=180-degree(arr,8,9,10)
    LUL=(-1)*degree(arr,19,11,12)-BODY
    LLL=180-degree(arr,11,12,13)
    theta = OrderedDict()
    theta['BODY']=BODY
    theta["LUA"]=LUA
    theta["LLA"]=LLA
    theta["RUA"]=RUA
    theta["RLA"]=RLA
    theta["LUL"]=LUL
    theta["LLL"]=LLL
    theta["RUL"]=RUL
    theta["RLL"]=RLL
#     theta={"LUA":LUA,"LLA":LLA,"RUA":RUA,"RLA":RLA,"LUL":LUL,"LLL":LLL,"RUL":RUL,"RLL":RLL}
    if(isImage):
        for i in theta:
            code=code+"R "+str(i)+" "+str(round((-1)*theta[i]))+"%0D%0A"
        return code
    else:
        return round((-1)*theta[part])

def paint(a):
    code=""
    pose_arr=a
    code=create(pose_arr,True)
    return code

def distance(arr,a,b):
    c=np.sqrt(np.square(arr[a][0]-arr[b][0])+np.square(arr[a][1]-arr[b][1]))
    return c

def basis(par):
    global armave
    global legave
    img = par
    #時間かかる
    pose_arr = model(img)
    pose_arr=pose_arr[0]
    pose_arr=pose_arr[0]
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

def isSide(pose_arr,armave,legave):
    RJcos=np.rad2deg(np.cos(np.sqrt(np.square(pose_arr[2][0]-pose_arr[3][0])+np.square(pose_arr[2][1]-pose_arr[3][1]))/armave['Rjouwan']))
    LJcos=np.rad2deg(np.cos(np.sqrt(np.square(pose_arr[5][0]-pose_arr[6][0])+np.square(pose_arr[5][1]-pose_arr[6][1]))/armave['Ljouwan']))
    RDcos=np.rad2deg(np.cos(np.sqrt(np.square(pose_arr[8][0]-pose_arr[9][0])+np.square(pose_arr[8][1]-pose_arr[9][1]))/legave['Rdaitai']))
    LDcos=np.rad2deg(np.cos(np.sqrt(np.square(pose_arr[11][0]-pose_arr[12][0])+np.square(pose_arr[11][1]-pose_arr[12][1]))/legave['Ldaitai']))

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
    if(abs(a)<=deg):
        return 0
    return a

def zaxis(basica,basicl,pose_arr):
    pose_arr=pose_arr
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
    print("RLA"+str(arctandegree(pose_arr,4,3,z,"zRzenwan")-zdeg["RUA"]))

    zdeg["LUA"]=arctandegree(pose_arr,6,5,z,"zRzenwan")
    zdeg["LLA"]=threshold(arctandegree(pose_arr,7,6,z,"zRzenwan")-zdeg["LUA"],30)
    print("RLA"+str(arctandegree(pose_arr,7,6,z,"zRzenwan")-zdeg["LUA"]))


    zdeg["RUL"]=arctandegree(pose_arr,9,8,z,"zRdaitai")
    zdeg["RLL"]=(-1)*threshold(arctandegree(pose_arr,10,9,z,"zRkatai")-zdeg["RUL"],20)
    print(arctandegree(pose_arr,10,9,z,"zRkatai")-zdeg["RUL"])
    zdeg["LUL"]=arctandegree(pose_arr,12,11,z,"zLdaitai")
    zdeg["LLL"]=(-1)*threshold(arctandegree(pose_arr,13,12,z,"zLkatai")-zdeg["LUL"],20)
    print(arctandegree(pose_arr,13,12,z,"zLkatai")-zdeg["LUL"])
    return zdeg

def arctandegree(arr,a,b,z,part):
    y=arr[a][1]-arr[b][1]
    z=z[part]
    return np.rad2deg(np.arctan2(z,y))

def CreateSide(zdeg):
    code="SIDE"+"%0D%0A"
    for g in zdeg:
        code=code+"R "+str(g)+" "+str(round(zdeg[g]))+"%0D%0A"
    return code

#------------------ここまでHumanPictogrammingの関数-------------------------------

#------------------ここからAnimationの関数-------------------------------



def frame2second(fps,frame):
    return frame/fps

def round1ex(number):
    return int(number+0.5)

def makegrapharray(array):
    img=array

def img2pose(img):
    pose_arr = model(img)
    pose_arr=pose_arr[0]
    return pose_arr[0]


def Video_path(video_file):
    delta=3
    Dsum=0
    code=""
    j=0
    i=0
    t=0.02
    part=["RUA","RLA","LUA","LLA","RUL","RLL","LUL","LLL"]
    pretime=0
    count=0
    start=0
    timesum=0
    timelist=[0]
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)+0.5)
    if(1/fps>0.02):
        t=1/fps
    listnum=0
    takeframe=fps*t

    # 総フレーム数とFPSを確認
    print("FRAME_COUNT: ",  frame_count)
    print("FPS: ", fps )
    #takeframe: t秒に必要なフレーム
    print(t,"[s]ごと")
    print(t,"[s]ごとに",takeframe,"frams")
    print("LENGTH:"+str((1/fps)*frame_count)+"[s]")

    print("--------------")

    while(round1ex(takeframe*(count+1))<=frame_count):
        count=count+1
        timelist.append(round1ex(takeframe*count))
    #RUA,RLA,LUA,LLA,RUL,RLL,LUL,LLL
    print("実行時間予想:",second_HMS(17*len(timelist)))
    theta_axis=np.zeros((8,len(timelist)))
    t_axis=np.zeros(len(timelist))
    while(cap.isOpened()):
        flag, frame = cap.read()  # Capture frame-by-frame
        if(flag!=True):  # Is a frame left?
            break
        if(i==timelist[listnum]):
            timesum+=frame2second(fps,i)-pretime
            print(frame2second(fps,i))
            t_axis[listnum]=frame2second(fps,i)
            #print('Save', image_dir+image_file % str(i).zfill(6))
            #ここを関数を追加する
            start=time.time()
            pose_array=img2pose(np.array(frame, dtype=np.float32))
            #def create(point,part):
            theta_axis[0][listnum]=create(pose_array,False,'RUA')
            theta_axis[1][listnum]=create(pose_array,False,'RLA')
            theta_axis[2][listnum]=create(pose_array,False,'LUA')
            theta_axis[3][listnum]=create(pose_array,False,'LLA')
            theta_axis[4][listnum]=create(pose_array,False,'RUL')
            theta_axis[5][listnum]=create(pose_array,False,'RLL')
            theta_axis[6][listnum]=create(pose_array,False,'LUL')
            theta_axis[7][listnum]=create(pose_array,False,'LLL')

            listnum=listnum+1
            if(listnum==len(timelist)):
                break
        i += 1


    cap.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    hoge=0
    for i in range(len(t_axis)):
        if(i>0 and t==0.02):
            hoge=hoge+t_axis[i]-t_axis[i-1]
    t=round(hoge/len(t_axis),2)
    print("動作単位時間"+str(t))
    for q in range(8):
        for j in range(len(theta_axis[q])):
            if(abs(theta_axis[q][j])<10):
                Dsum+=Dsum+abs(theta_axis[q][j])
            if(360-abs(theta_axis[q][j])<10):
                Dsum+=Dsum+(360-abs(theta_axis[q][j]))
        if(Dsum/len(theta_axis[q])<10):
            for k in range(len(theta_axis[q])):
                theta_axis[q][k]=0
        theta_axis[q]=replace(theta_axis[q])
        code=code+UpdateGraph(np.linspace(0, len(theta_axis[q])*t,len(theta_axis[q])),theta_axis[q],part[q],delta,t)
    return code



def Video_camera(num=0):
    delta=3
    Dsum=0
    j=0
    i=0
    t=0.02
    part=["RUA","RLA","LUA","LLA","RUL","RLL","LUL","LLL"]
    pretime=0
    count=0
    start=0
    timesum=0
    timelist=[0]

    cap = cv2.VideoCapture(num)
    print(int(cap.get(cv2.CAP_PROP_FPS)+0.5))
    frames=np.zeros((720,1280,3))
    i=0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here

        # Display the resulting frame
        cv2.imshow('frame',frame)
        frames=np.append(frames,frame,axis=2)
        i+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    frames=np.delete(a,0,axis=2)
    frames=np.delete(a,0,axis=2)
    frames=np.delete(a,0,axis=2)

    cap = cv2.VideoCapture(num)
    fps = int(cap.get(cv2.CAP_PROP_FPS)+0.5)
    if(1/fps>0.02):
        t=1/fps
    listnum=0
    takeframe=fps*t





    # 総フレーム数とFPSを確認
    print("FRAME_COUNT: ",  frame_count)
    print("FPS: ", fps )
    #takeframe: t秒に必要なフレーム
    print(t,"[s]ごと")
    print(t,"[s]ごとに",takeframe,"frams")

    print("--------------")

    while(round1ex(takeframe*(count+1))<=frame_count):
        count=count+1
        timelist.append(round1ex(takeframe*count))
    #RUA,RLA,LUA,LLA,RUL,RLL,LUL,LLL
    print("実行時間予想[s]:",16*len(timelist)+"s")
    theta_axis=np.zeros((8,len(timelist)))
    t_axis=np.zeros(len(timelist))
    while(True):
        if(flag!=True):  # Is a frame left?
            break
        if(i==timelist[listnum]):
            timesum+=frame2second(fps,i)-pretime
            print(frame2second(fps,i))
            t_axis[listnum]=frame2second(fps,i)
            #print('Save', image_dir+image_file % str(i).zfill(6))
            #ここを関数を追加する
            start=time.time()
            frame=frames[:,:,listnum:listnum+3]
            pose_array=img2pose(np.array(frame, dtype=np.float32))
            #def create(point,part):
            theta_axis[0][listnum]=create(pose_array,False,'RUA')
            theta_axis[1][listnum]=create(pose_array,False,'RLA')
            theta_axis[2][listnum]=create(pose_array,False,'LUA')
            theta_axis[3][listnum]=create(pose_array,False,'LLA')
            theta_axis[4][listnum]=create(pose_array,False,'RUL')
            theta_axis[5][listnum]=create(pose_array,False,'RLL')
            theta_axis[6][listnum]=create(pose_array,False,'LUL')
            theta_axis[7][listnum]=create(pose_array,False,'LLL')
            #plt.show()
            #print("----------")
            #print("時間",time.time()-start)
            listnum=listnum+1
            if(listnum==len(timelist)):
                break
        i += 1


    cap.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    hoge=0
    for i in range(len(t_axis)):
        if(i>0 and t==0.02):
            hoge=hoge+t_axis[i]-t_axis[i-1]
    t=hoge/len(t_axis)
    for q in range(8):
        for j in range(len(theta_axis[q])):
            if(abs(theta_axis[q][j])<10):
                Dsum+=Dsum+abs(theta_axis[q][j])
            if(360-abs(theta_axis[q][j])<10):
                Dsum+=Dsum+(360-abs(theta_axis[q][j]))
        if(Dsum/len(theta_axis[q])<10):
            for k in range(len(theta_axis[q])):
                theta_axis[q][k]=0
        theta_axis[q]=replace(theta_axis[q])
        code=code+UpdateGraph(np.linspace(0, len(theta_axis[q])*t,len(theta_axis[q])),theta_axis[q],part[q],delta,t)
    return code

#------------------ここまでAnimationの関数-------------------------------
print("Finish!")



import matplotlib.pyplot as plt
import numpy as np
import random

print("start")

def replace(array):
    for i in  range(len(array)):
        if(i>0):
            if(abs(array[i]-array[i-1])>330):
                array[i]=array[i]+min(abs(array[i]),abs(360-array[i]),abs(array[i]-360))
    return array

#点と点の差を求める
def difference(x,y,a,b,c): #x,yは全部じゃなくて1点
    if(a!=0):
        dtheta=y-(a*x**2+b*x+c)
        return abs(dtheta)
    else:
        dtheta=y-(b*x+c)
        return abs(dtheta)

def differave(x,y,a,b,c,minimum,maximum):
    ranges=0
    if(maximum-minimum>2):
        ranges=maximum-minimum
    approx=0
    for i in range(ranges):
        #def difference(x,y,a,b,c):
        approx+=difference(x[i+minimum],y[i+minimum],a,b,c)
    return approx/ranges



def calcfunc(x,t,a,b,c):
    if(a==0):
        return x[t]*b+c
    else:
        return a*x[t]**2+b*x[t]+c

#近似式をだす.係数を返す
def DegreeOfApproximate(x,y,minimum,maximum):
    ranges=maximum-minimum
    a1, b1 = np.polyfit(x[minimum:maximum],y[minimum:maximum],1)
    a2,b2,c2=np.polyfit(x[minimum:maximum],y[minimum:maximum],2)
    return comparefunc(x,y,minimum,maximum,0,a1,b1,a2,b2,c2,False)

def comparefunc(x,y,minimum,maximum,a1,b1,c1,a2,b2,c2,compare=True):
    ranges=0
    #古いほうがどの程度新しいのより大きいのを許容するか
    if(maximum==0 and minimum!=0):
            maximum=len(x)
    if(maximum-minimum>2):
        ranges=maximum-minimum
    approxA,approxB=0,0
    for i in range(ranges):
        #def difference(x,y,a,b,c):
        approxA+=difference(x[i+minimum],y[i+minimum],a1,b1,c1)
        approxB+=difference(x[i+minimum],y[i+minimum],a2,b2,c2)
    approxA=approxA/ranges
    approxB=approxB/ranges
    #print(min(approxA,approxB))
    if(approxA>approxB): #1次式A > 2次式B
    #older is bigger
        return a2,b2,c2
    #newer is bigger
    else:
        return a1,b1,c1

#式をprint文で返す
def printfunc(a,b,c):
    if(a==0):
        print(str(b)+"*a+"+str(c))
    else:
        print(str(a)+"*a**2+"+str(b)+"*a+"+str(c))

def straight(y):
    isStraight=False
    start=0
    end=0
    counter=0
    count=3
    st=np.zeros((1,2))

    for t in range(len(y)):
        if(isStraight==False):
            if(abs(y[t]-y[t-1])<=3):
                counter=0
                start=t-1
                number=y[t-1]
                isStraight=True
        if(isStraight):
            counter=counter+1
            if(abs(number-y[t])>3):
                end=t-1
                isStraight=False
                if(counter>count):
                    st=np.append(st,[[start,end]],axis=0)

    st=np.delete(st,0,0)
    i=0
    while(i<len(st)):
        sums=sum(y[int(st[i][0]):int(st[i][1])])
        length=len(y[int(st[i][0]):int(st[i][1])])
        if(length==0):
            st=np.delete(st,i,0)
            i=0
        elif(abs(abs(sums/length)-abs(y[int(st[i][0])]))>3):
            st=np.delete(st,i,0)
            i=0
        i+=1
    #print(st)
    return st


def printcode(a,b,c,minimum,maximum,part,y,t):
    code=""
    ranges=maximum-minimum
    if(a==0 and b==0 and c==0):
        code="R"+" "+part+" "+"0"+" "+str(ranges*t)+" "+str(minimum*t)+"%0D%0A"
    else:
        if(a==0):
            diff=b*ranges*t
            code="R"+" "+part+" "+str(diff)+" "+str(ranges*t)+" "+str(minimum*t)+"%0D%0A"
        else:
            code=code+"SET :T"+part+" "+str(0)+"%0D%0A"
            code=code+"SET :OMEGA"+part+" "+str(round((2*a*t*minimum+b)*t,2))+"%0D%0A"
            code=code+"REPEAT"+" "+str(ranges)+"%0D%0A"
            code=code+"SET :OMEGA"+part+" [:OMEGA"+part+"+"+str(round(2*a*t*t,2))+"]"+"%0D%0A"
            code=code+"R"+" "+part+" "+":OMEGA"+part+" "+str(t)+" "+"["+str(minimum*t)+"+:T"+part+"*"+str(t)+"]"+"%0D%0A"
            code=code+"SET :T"+part+" [:T"+part+"+1]"+"%0D%0A"
            code=code+"END"+"%0D%0A"
    return code



def UpdateGraph(x,y,part,delta,t):
    ConfficientArray=np.zeros((1,5))  #係数を格納0~len(x)-1,0~2    [i][]=>o〜i(>2)のやつ  [0][]は除く
    confficient=np.zeros((1,5))
    a,b,c=0,0,0
    start,end=0,0
    number=0
    code=""
    flag=False
    counter=0
    count=0
    straightnum=straight(y)
    nextnum=0
    #comparefunc(x,y,minimum,maximum,a1,b1,c1,a2,b2,c2):
    #DegreeOfApproximate(x,y,minimum,maximum=0):
    #i-1~i
    #confficient[3][0],confficient[3][1],confficient[3][2]=DegreeOfApproximate(x,y,0,3)
    #confficient=[a,b,c,minimum,maximum]
#     for i in range(len(straightnum)):
#         ConfficientArray=np.append(ConfficientArray,[[0,0,0,straightnum[i][0],straightnum[i][1]]],axis=0)
    for i in range(len(x)):
        end=i
        if(count<len(straightnum)):
            if(i==int(straightnum[count][0])):
                confficient=[[a,b,c,start,end]]
                ConfficientArray=np.append(ConfficientArray,confficient,axis=0)
                i=int(straightnum[count][1])
                start=int(straightnum[count][1])
                end=int(straightnum[count][1])
                confficient=[[0,0,0,int(straightnum[count][0]),int(straightnum[count][1])]]
                ConfficientArray=np.append(ConfficientArray,confficient,axis=0)
                count=count+1
        if(end-start>2):
            #print(start,",",end)
            #print(i)
            #minimum~maximumの近似式の係数
            a,b,c=DegreeOfApproximate(x,y,start,end)
            if(count>=len(straightnum)):
                if(differave(x,y,a,b,c,start,end)>delta):
                #print(i,"転換")
                    confficient=[[a,b,c,start,end]]
                    ConfficientArray=np.append(ConfficientArray,confficient,axis=0)
                    start=i
            else:
                if(differave(x,y,a,b,c,start,end)>delta and straightnum[count][0]-end>3):
                #print(i,"転換")
                    confficient=[[a,b,c,start,end]]
                    ConfficientArray=np.append(ConfficientArray,confficient,axis=0)
                    start=i
                if(straightnum[count][0]==end):
                    a,b,c=DegreeOfApproximate(x,y,start,int(straightnum[count][0]))
                    confficient=[[a,b,c,start,straightnum[count][0]]]
                    ConfficientArray=np.append(ConfficientArray,confficient,axis=0)
                    start=i
    confficient=[[a,b,c,start,end-1]]
    ConfficientArray=np.delete(ConfficientArray,0,0)
    ConfficientArray=np.append(ConfficientArray,confficient,axis=0)
    s = sorted(ConfficientArray,key=lambda i:i[3])
    ConfficientArray= np.array(s)
    #print(ConfficientArray)
    #print("----------------------------------")
    for j in range(len(ConfficientArray)):
        print("")
        if(ConfficientArray[j][4]-ConfficientArray[j][3]>0):
            code=code+printcode(ConfficientArray[j][0],ConfficientArray[j][1],ConfficientArray[j][2],int(ConfficientArray[j][3]),int(ConfficientArray[j][4]),part,y,t)
            if(j!=len(ConfficientArray)-1):
                if(ConfficientArray[j][0]==0 and ConfficientArray[j][1]==0 and ConfficientArray[j][2]==0):
                    differ=(ConfficientArray[j+1][0]*((ConfficientArray[j+1][3])*t)**2+ConfficientArray[j+1][1]*((ConfficientArray[j+1][3])*t)+ConfficientArray[j+1][2])-y[int(ConfficientArray[j][3])]
                    code=code+"R"+" "+part+" "+str(round(differ,2))+" "+str((ConfficientArray[j+1][4]-ConfficientArray[j+1][3])*t)+" "+str(ConfficientArray[j+1][3]*t)+"%0D%0A"
                elif(ConfficientArray[j+1][0]==0 and ConfficientArray[j+1][1]==0 and ConfficientArray[j+1][2]==0):
                    differ=y[int(ConfficientArray[j+1][3])]-(ConfficientArray[j][0]*(ConfficientArray[j][4]*t)**2+ConfficientArray[j][1]*(ConfficientArray[j][4]*t)+ConfficientArray[j][2])
                    code=code+"R"+" "+part+" "+str(round(differ,2))+" "+str((ConfficientArray[j][4]-ConfficientArray[j][3])*t)+" "+str(ConfficientArray[j][3]*t)+"%0D%0A"
                else:
                    differ=(ConfficientArray[j+1][0]*((ConfficientArray[j+1][3])*t)**2+ConfficientArray[j+1][1]*((ConfficientArray[j+1][3])*t)+ConfficientArray[j+1][2])-(ConfficientArray[j][0]*((ConfficientArray[j][4])*t)**2+ConfficientArray[j][1]*((ConfficientArray[j][4])*t)+ConfficientArray[j][2])
                    code=code+"R "+part+" "+str(round(differ,2))+" "+str((ConfficientArray[j][4]-ConfficientArray[j][3])*t)+" "+str(ConfficientArray[j][3]*t)+"%0D%0A"
    return code



def Image_camera(num=0):
#1はUSBカメラ、0は内蔵カメラ
    cap = cv2.VideoCapture(num)
    code=""
    zdeg={}

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
            img=np.array(frame, dtype=np.float32)
            pose_arr = model(img)
            pose_arr=pose_arr[0]
            result_img = draw_person_pose(img, pose_arr)
            pose_arr=pose_arr[0]
            if(isKeypoint(pose_arr)):
                if(isSide(pose_arr,armave,legave)):
                    zdeg=zaxis(armave,legave,pose_arr)
                    code=CreateSide(zdeg)
                else:
                    code=paint(pose_arr)
            break

    cap.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return code

def OpenPictogramming(code):
    url = "http://127.0.0.1:8000/?code="+code
    webbrowser.open(url)

def Image_path(path):
#1はUSBカメラ、0は内蔵カメラ
    code=""
    zdeg={}
    frame=cv2.imread(path)

    img=np.array(frame, dtype=np.float32)
    pose_arr = model(img)
    pose_arr=pose_arr[0]
    result_img = draw_person_pose(img, pose_arr)
    pose_arr=pose_arr[0]
    if(isKeypoint(pose_arr)):
        if(isSide(pose_arr,armave,legave)):
            zdeg=zaxis(armave,legave,pose_arr)
            code=CreateSide(zdeg)
        else:
            code=paint(pose_arr)
    return code



#video or image ideopath
args = sys.argv
print(args)
if(len(args)==2): #pathが無い(camera使用)
    if(args[1]=='Usage'):
        print("python HumanPictogramming.py [Video or Image] [path]")
    elif(args[1]=='Image' or args[1]=='image'):
        print('image')
        model = PoseDetector(arch_name, weight_path)
        code=Image_camera()
        OpenPictogramming(code)
    elif(args[1]=='Video' or args[1]=='video'):
        print('video')
        model = PoseDetector(arch_name, weight_path)
        code=Video_camera()
        OpenPictogramming(code)
    else:
        print('check your args')
        print("python HumanPictogramming.py [Video or Image] [path]")
elif(len(args)==3): #pathあり
    if(args[1]=='Image' or args[1]=='image'):
        print('image')
        print(args[2])
        model = PoseDetector(arch_name, weight_path)
        code=Image_path(args[2])
        OpenPictogramming(code)
    elif(args[1]=='Video' or args[1]=='video'):
        print('video')
        model = PoseDetector(arch_name, weight_path)
        code=Video_path(args[2])
        OpenPictogramming(code)
    else:
        print('check your args')
        print("python HumanPictogramming.py [Video or Image] [path]")
else: #引数なしor引数過多
    print('check your args')
    print("python HumanPictogramming.py [Video or Image] [path]")
