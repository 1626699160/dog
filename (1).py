#! /usr/bin/env python3
import rospy
import cv2
import os
import queue
import threading
import time
import numpy as np
from geometry_msgs.msg import Twist
# from std_msgs.msg import String
import socket, struct, time


class VideoCapture:
    """Customized VideoCapture, always read latest frame """

    def __init__(self, camera_id):
        # "camera_id" is a int type id or string name
        self.cap = cv2.VideoCapture(camera_id)
        self.q = queue.Queue(maxsize=3)
        self.stop_threads = False  # to gracefully close sub-thread
        th = threading.Thread(target=self._reader)
        th.daemon = True  # 设置工作线程为后台运行
        th.start()

    # 实时读帧，只保存最后一帧
    def _reader(self):
        while not self.stop_threads:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def terminate(self):
        self.stop_threads = True
        self.cap.release()


class udpTest():
    def __init__(self):
        # 创建套接字
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 发送数据
        self.send_addr = ('192.168.1.120', 43893)
        self.send_data(1, 0, 0)
        time.sleep(5)
        self.send_data(2, 0, 0)
        time.sleep(5)

    def send_data(self, code, value, type):
        data = struct.pack("<3i", code, value, type)
        self.udp_socket.sendto(data, self.send_addr)

    def pufu(self):
        print("pufu")
        self.send_data(0x31010406, 0, 0)

    def louti(self):
        print("loti")
        self.send_data(0x31010407, 0, 0)

    def pingdi(self):
        print("pingdi")
        self.send_data(0x31010300, 0, 0)

    def zhengque(self):
        print("ZHENGQUE")
        self.send_data(0x31010D05, 0, 0)
        self.send_data(0x31010C02, 0, 0)
        rospy.sleep(1)
        self.send_data(0x31010130, 20000, 0)
        rospy.sleep(1)
        self.send_data(0x31010130, -20000, 0)
        rospy.sleep(1)
        self.send_data(0x31010130, 20000, 0)
        rospy.sleep(1)
        self.send_data(0x31010130, -20000, 0)
        rospy.sleep(1)
        self.send_data(0x31010D05, 0, 0)
        self.send_data(0x31010C03, 0, 0)

    def chuowu(self):
        print("CHUOWU")
        self.send_data(0x31010C02, 0, 0)
        self.send_data(0x31010D05, 0, 0)
        rospy.sleep(1)
        self.send_data(0x31010135, 20000, 0)
        rospy.sleep(1)
        self.send_data(0x31010135, -20000, 0)
        rospy.sleep(1)
        self.send_data(0x31010135, 20000, 0)
        rospy.sleep(1)
        self.send_data(0x31010135, -20000, 0)
        rospy.sleep(1)
        self.send_data(0x31010D05, 0, 0)
        self.send_data(0x31010C03, 0, 0)


video = "http://admin:admin@192.168.1.105:8081/"  # 此处@后的ipv4 地址需要修改为自己的地址

if __name__ == "__main__":
    u = udpTest()
    # video = "http://admin:admin@192.168.1.102:8081/"
    rospy.init_node("shudu")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lower_purple = np.array([-1, 140, 120])
    higher_purple = np.array([200, 250, 220])
    # lower_red=np.array([90,90,20])
    # higher_red=np.array([150,150,90])
    lower_red = np.array([10, 30, 20])
    higher_red = np.array([200, 150, 90])
    # lower_yellow=np.array([90,90,20])
    # higher_yellow=np.array([150,150,90])
    lower_yellow = np.array([0, 100, 200])
    higher_yellow = np.array([30, 200, 300])
    lower_1 = np.array([60, 200, 120])
    higher_1 = np.array([120, 280, 200])
    lower_2 = np.array([0, 150, 100])
    higher_2 = np.array([50, 240, 300])
    cap1 = cv2.VideoCapture(2)
    frame1 = VideoCapture(video)
    # frame1=cv2.VideoCapture(3)
    # frame1=VideoCapture(4)
    g = Twist()

    # u.a()
    # rospy.sleep(2)
    # g.angular.z=1
    # cap = cv2.VideoCapture(2)
    loutipanduan = 0
    loutijiance = 1
    langanjiance = 1
    langanpanduan = 0
    tong = 0
    tong1 = 0
    tong2 = 0
    tupian = 0
    shijian4 = 0
    lt1 = 1
    lg1 = 1
    dx = 0
    dx2 = 0
    dx3 = 300
    dx4 = 20
    jiaohuan1 = 0
    jh = 0
    jhb = 0

    dy0 = 1
    dy1 = 0
    dy2 = 0
    dy3 = 0
    dy4 = 0
    dy5 = 0
    dy6 = 0
    dy7 = 0
    dy8 = 0
    dy9 = 0
    dy10 = 0
    dy11 = 0
    dy12 = 0
    dy13 = 0
    dy14 = 0
    dy15 = 0
    dy16 = 0
    dy17 = 0
    dy18 = 0
    dy19 = 0
    dy20 = 0
    dy21 = 0
    dy22 = 0
    dy23 = 0
    dy24 = 0
    dy25 = 0
    dy26 = 0
    dy27 = 0
    dy28 = 0
    dy29 = 0
    dy30 = 0

    a1 = 0
    a2 = 2
    a3 = 3
    b1 = 4
    b2 = 5
    b3 = 6
    '''while 1:

        g.angular.z=-0.02
        pub.publish(g)'''
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while dy0:
        ex = 45
        ex1 = 10
        ey = 38
        jiaohuan1 = 0
        ret, cap = cap1.read()
        cap = cv2.resize(cap, (640, 360))
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)
        img_hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(img_hsv, lower_red, higher_red)  # 获得绿色部分掩膜
        mask_yellow = cv2.medianBlur(mask_yellow, 11)  # 中值滤波
        cnts3, hierarchy3 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if lt1 == 1:
            for cnt in cnts3:
                if loutijiance == 1:
                    (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
                    if w > 200:
                        cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                        cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                        if y > dx3:
                            loutipanduan = 1
                            jiaohuan1 = 1
                            print("1//////////1")

        mask_red = cv2.inRange(img_hsv, lower_yellow, higher_yellow)  # 获得绿色部分掩膜
        mask_red = cv2.medianBlur(mask_red, 7)  # 中值滤波
        cnts1, hierarchy1 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if lg1 == 1:
            for cnt in cnts1:
                # while dy2:
                if langanjiance == 1:
                    (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
                    if w > 500:
                        if h < 50:
                            cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                            cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                            if y < 100:
                                langanpanduan = 1
                                jiaohuan1 = 2
        if jh == 0:
            if jiaohuan1 == 2:
                jhb = 2
                jh = 1
            elif jiaohuan1 == 1:
                jhb = 1
                jh = 1
        print(jhb)
        if jhb == 1:
            if loutipanduan == 1:
                u.louti()
                rospy.sleep(1)
                lt1 = 0
                loutijiance = 0
                loutipanduan = 0
                jiaohuan1 = 1
                while ex:
                    g.linear.x = 0.1
                    pub.publish(g)
                    rospy.sleep(0.2)
                    ex -= 1
                # if langanpanduan == 1:

                u.pingdi()
                u.pufu()
                lg1 = 0
                rospy.sleep(1)
                while ex1:
                    g.linear.x = 0.1
                    pub.publish(g)
                    rospy.sleep(0.2)
                    ex1 -= 1
                langanjiance = 0
                langanpanduan = 0
                dx = 1
                dy0 = 0
        if jhb == 2:
            dx3 = 190
            if loutipanduan == 1:
                u.pufu()
                rospy.sleep(1)
                u.louti()
                rospy.sleep(1)
                while ey:
                    g.linear.x = 0.1
                    pub.publish(g)
                    rospy.sleep(0.2)
                    ey -= 1

                lt1 = 0
                loutijiance = 0
                loutipanduan = 0
                jiaohuan1 = 1
                u.pingdi()
                u.pufu()
                rospy.sleep(1)
                dx = 0
                dy0 = 0
                dy1 = 1
            if langanpanduan == 1:
                u.pufu()
                lg1 = 0
                rospy.sleep(1)
                langanjiance = 0
                langanpanduan = 0

        g.linear.x = 0.1
        pub.publish(g)
        rospy.sleep(0.2)
        g.linear.x = 0
        cv2.imshow('cap', mask_yellow)
        cv2.imshow('cap1', cap)

        if cv2.waitKey(90) & 0xFF == ord('q'):
            break
    while dx == 1:
        bbbbb = 10
        while bbbbb:
            g.linear.x = 0.1
            pub.publish(g)
            rospy.sleep(0.2)
            bbbbb -= 1
        dx = 0
        dy1 = 1
    while dy1:
        c = 0
        d = 0
        e = 0
        b = 0
        ret, cap = cap1.read()
        cap = cv2.resize(cap, (640, 360))
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)
        img_hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(img_hsv, lower_purple, higher_purple)  # 获得绿色部分掩膜
        mask_yellow = cv2.medianBlur(mask_yellow, 5)  # 中值滤波
        cnts1, hierarchy1 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt1 in cnts1:
            (x, y, w, h) = cv2.boundingRect(cnt1)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong = 1
                # b = (x+(w/2))-320
        mask_1 = cv2.inRange(img_hsv, lower_1, higher_1)  # 获得绿色部分掩膜
        mask_1 = cv2.medianBlur(mask_1, 5)  # 中值滤波
        cnts, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong = 2
                # b = (x+(w/2))-320
        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        dst = cv2.dilate(dst, None, iterations=0)
        dst = cv2.erode(dst, None, iterations=11)
        color = dst[340]
        color1 = dst[300]
        # 找到白色的像素点个数
        white_count = np.sum(color == 255)
        white_count1 = np.sum(color1 == 255)
        # 找到白色的像素点索引
        white_index = np.where(color == 255)
        white_index1 = np.where(color1 == 255)
        if white_count == 0:
            white_count = 1
        if white_index == 0:
            white_index = 1
        try:
            center_now = (white_index[0][white_count - 1] + white_index[0][0]) / 2
            a = center_now - 320
        except:
            continue
        if white_count1 == 0:
            white_count1 = 1
        if white_index1 == 0:
            white_index1 = 1
        try:
            center_now1 = (white_index1[0][white_count1 - 1] + white_index1[0][0]) / 2
            ac = center_now1 - 320
        except:
            continue
        '''print("////////////////aaaa")
        print(a)
        print("////////////////bbbb")
        print(b)'''
        if 60 < a < 80:
            print("////////////////3333")
            c = 1
        elif 80 < a:
            g.linear.y = -0.02
            # g.angular.z=-0
            pub.publish(g)
            # rospy.sleep(0.1)
            g.angular.z = 0
            g.linear.y = 0
            print("/////////////////222")
        elif 60 > a:
            g.linear.y = 0.02
            # g.angular.z=-0
            pub.publish(g)
            # g.angular.z=0.0
            rospy.sleep(0.1)
            g.linear.y = 0
            print("//////////////111")
        if c == 1:
            # b = (x+(w/2))-320
            b = ac - a
            if -10 < b < 10:
                d = 1
                print("////////////////44444")
            elif b > 10:
                g.angular.z = -0.02
                pub.publish(g)
                rospy.sleep(0.4)
                g.angular.z = 0
                print("1111/////////////////111")
            elif b < -10:
                g.angular.z = 0.02
                pub.publish(g)
                rospy.sleep(0.1)
                g.angular.z = 0
                print("22222//////////////2222")
        if d == 1:
            if e > 15000:

                if tong == 1:
                    dy1 = 0
                    dy2 = 1
                    g.linear.x = 0
                    rospy.sleep(0.1)
                    u.pufu()
                    rospy.sleep(1)
                if tong == 2:
                    dy1 = 0
                    dy4 = 1
                    g.linear.x = 0
                    rospy.sleep(0.1)
                    u.pufu()
                    rospy.sleep(1)
                print("////////////////5555555")
            else:
                g.linear.x = 0.1
                pub.publish(g)
                rospy.sleep(0.1)

                print("//////////////111")
        '''if e>15000:

                if tong==1: 
                        dy2=1
                        dy1=0
                        u.pufu()
                if tong ==2 :
                        dy3=1
                        dy1=0
                        u.pufu()
                print("////////////////5555555")'''
        g.linear.x = 0
        g.linear.y = 0
        g.angular.z = 0
        cv2.imshow('cap1', dst)
        if chr(cv2.waitKey(1) & 255) == 'q':  # 按 q 退出
            cap.terminate()
            break
    while dy2:
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        g.linear.x = 0.005
        pub.publish(g)
        # rospy.sleep(0.1)
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)
        img1 = cv2.imread('/home/ysc/learningx/tuku/110.jpg')
        img2 = cv2.imread('/home/ysc/learningx/tuku/120.jpg')
        img3 = cv2.imread('/home/ysc/learningx/tuku/119.jpg')
        img4 = cv2.imread('/home/ysc/learningx/tuku/huo.jpg')
        img5 = cv2.imread('/home/ysc/learningx/tuku/tou.jpg')
        img6 = cv2.imread('/home/ysc/learningx/tuku/sou.jpg')
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        h4, w4 = img4.shape[:2]
        h5, w5 = img5.shape[:2]
        h6, w6 = img6.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

        retva, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        retval, dst1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_OTSU)
        retva2, dst2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)
        retva3, dst3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_OTSU)
        retva4, dst4 = cv2.threshold(gray4, 0, 255, cv2.THRESH_OTSU)
        retva5, dst5 = cv2.threshold(gray5, 0, 255, cv2.THRESH_OTSU)
        retva6, dst6 = cv2.threshold(gray6, 0, 255, cv2.THRESH_OTSU)
        # 相似匹配
        # 列表中所有的6种比较方法   u.zz1()
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                   'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

        res1 = cv2.matchTemplate(dst, dst1, cv2.TM_CCORR_NORMED)
        res2 = cv2.matchTemplate(dst, dst2, cv2.TM_CCORR_NORMED)
        res3 = cv2.matchTemplate(dst, dst3, cv2.TM_CCORR_NORMED)
        res4 = cv2.matchTemplate(dst, dst4, cv2.TM_CCORR_NORMED)
        res5 = cv2.matchTemplate(dst, dst5, cv2.TM_CCORR_NORMED)
        res6 = cv2.matchTemplate(dst, dst6, cv2.TM_CCORR_NORMED)
        # print(res.shape)
        threshold = 0.92
        threshold1 = 0.92
        loc1 = np.where(res1 >= threshold)
        loc2 = np.where(res2 >= threshold)
        loc3 = np.where(res3 >= threshold)
        loc4 = np.where(res4 >= threshold1)
        loc5 = np.where(res5 >= threshold1)
        loc6 = np.where(res6 >= threshold1)

        # 获取最匹配的位置信息
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
        min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)
        min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)
        min_val6, max_val6, min_loc6, max_loc6 = cv2.minMaxLoc(res6)

        img100 = frame.copy()

        if len(loc1[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc1[::-1]):
                print("向左和向右转")
                a1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w1, pt[1] + h1), (0, 255, 0), 1)
        if len(loc2[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc2[::-1]):
                print("直行和向左转弯")
                a2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 1)
        if len(loc3[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc3[::-1]):
                print("直行和向右转弯")
                a3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w3, pt[1] + h3), (0, 255, 0), 1)
        if len(loc4[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc4[::-1]):
                print("向右转弯")
                b1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w4, pt[1] + h4), (0, 255, 0), 1)
        if len(loc5[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc5[::-1]):
                print("向左转弯")
                b2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w5, pt[1] + h5), (0, 255, 0), 1)
        if len(loc6[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc6[::-1]):
                b3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w6, pt[1] + h6), (0, 255, 0), 1)
        if a1 == b3:

            print("急救标志 电话110 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话110匹配错误'")
            tupian = 2
            dy2 = 0
            dy3 = 1
        elif a3 == b3:

            print("急救标志 电话119 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话119匹配错误'")
            tupian = 2
            dy2 = 0
            dy3 = 1
        elif a2 == b3:

            print("急救标志 电话120 正确")
            os.system("espeak -vzh '此区域发现伤员拨打120匹配正确'")
            tupian = 1
            dy2 = 0
            dy3 = 1
        elif a1 == b2:

            print("危险品标志 电话110 正确")
            os.system("espeak -vzh '此区域发现危险品拨打110匹配正确'")
            tupian = 1
            dy2 = 0
            dy3 = 1
        elif a3 == b2:
            print("危险品标志 电话119 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话119匹配错误'")
            tupian = 2
            dy2 = 0
            dy3 = 1
        elif a2 == b2:

            print("危险品标志 电话120 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话120匹配错误'")
            tupian = 2
            dy2 = 0
            dy3 = 1
        if a1 == b1:

            print("火警标志 电话110 错误")
            os.system("espeak -vzh '此区域发现火灾，报警电话110,匹配错误'")
            tupian = 2
            dy2 = 0
            dy3 = 1
        elif a3 == b1:

            print("火警标志 电话119 正确")
            os.system("espeak -vzh '此区域发现火灾，拨打119,匹配正确'")
            tupian = 1
            dy2 = 0
            dy3 = 1
        elif a2 == b1:

            print("火警标志 电话120 错误")
            os.system("espeak -vzh '此区域发现火灾报警电话120匹配错误'")
            tupian = 2
            dy2 = 0
            dy3 = 1
        # cv2.namedWindow("frame", 0)#创立一个创客
        # cv2.resizeWindow("frame",325 ,288)
        cv2.imshow('frame', ju_xing)  # 将图片显示在窗口
        if cv2.waitKey(90) & 0xFF == ord('q'):
            break
    while dy3:
        if tupian == 2:
            u.chuowu()
            dy3 = 0
            dy4 = 1
            tupian = 0
        if tupian == 1:
            u.zhengque()
            dy3 = 0
            dy4 = 1
            tupian = 0

    while dy4:
        print("dy4")
        shijian4 = 61
        while shijian4:
            g.linear.y = 0.5
            pub.publish(g)
            rospy.sleep(0.1)
            shijian4 -= 1
            dx2 = 1
        g.linear.y = 0
        g.linear.x = 0
        dy4 = 0
        dy5 = 1
        g.linear.y = 0
        pub.publish(g)
        rospy.sleep(2)
        u.pufu()
        rospy.sleep(1)
    while dy5:
        a1 = 0
        a2 = 2
        a3 = 3
        b1 = 4
        b2 = 5
        b3 = 6
        c = 0
        d = 0
        e = 0
        b = 0
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        ret, cap = cap1.read()
        cap = cv2.resize(cap, (640, 360))
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)

        img_hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(img_hsv, lower_purple, higher_purple)  # 获得绿色部分掩膜
        mask_yellow = cv2.medianBlur(mask_yellow, 5)  # 中值滤波
        cnts1, hierarchy1 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt1 in cnts1:
            (x, y, w, h) = cv2.boundingRect(cnt1)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong2 = 1
                b = (x + (w / 2)) - 320
        mask_1 = cv2.inRange(img_hsv, lower_1, higher_1)  # 获得绿色部分掩膜
        mask_1 = cv2.medianBlur(mask_1, 5)  # 中值滤波
        cnts, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong2 = 2
                b = (x + (w / 2)) - 320
        if -10 < b < 10:
            print("////////////////3333")
            d = 1
        elif 10 < b:
            g.linear.y = -0.15
            # g.angular.z=-0
            pub.publish(g)
            # rospy.sleep(0.1)
            g.angular.z = 0
            g.linear.y = 0
            print("/////////////////222")
        elif -10 > b:
            g.linear.y = 0.15
            # g.angular.z=-0
            pub.publish(g)
            # g.angular.z=0.0
            rospy.sleep(0.1)
            g.linear.y = 0
            print("//////////////111")
        if d == 1:
            if e > 10000:

                '''if tong==1:
                         dy15=0
                         dy16=1
                         u.pufu()
                if tong ==2 :
                         dy15=0 
                         dy17=1
                         u.pufu()'''
                print("////////////////5555555")
            else:
                g.linear.x = 0.5
                pub.publish(g)
                rospy.sleep(0.1)
                print("//////////////111")
        if e > 10000:

            if tong2 == 1:
                dy5 = 0
                dy6 = 1
                g.linear.x = 0.0
                pub.publish(g)
                rospy.sleep(1)
                u.pufu()
                rospy.sleep(1)
            if tong2 == 2:
                dy5 = 0
                dy8 = 1
                g.linear.x = 0.0
                pub.publish(g)
                rospy.sleep(1)
                u.pufu()
                rospy.sleep(1)
            print("////////////////5555555")

        g.linear.x = 0
        g.linear.y = 0
        g.angular.z = 0
        cv2.imshow('cap1', cap)
        cv2.imshow('frame', frame)
        if chr(cv2.waitKey(1) & 255) == 'q':  # 按 q 退出
            cap.terminate()
            break
    while dy6:
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        g.linear.x = 0.005
        pub.publish(g)
        # rospy.sleep(0.1)
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)
        img1 = cv2.imread('/home/ysc/learningx/tuku/110.jpg')
        img2 = cv2.imread('/home/ysc/learningx/tuku/120.jpg')
        img3 = cv2.imread('/home/ysc/learningx/tuku/119.jpg')
        img4 = cv2.imread('/home/ysc/learningx/tuku/huo.jpg')
        img5 = cv2.imread('/home/ysc/learningx/tuku/tou.jpg')
        img6 = cv2.imread('/home/ysc/learningx/tuku/sou.jpg')
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        h4, w4 = img4.shape[:2]
        h5, w5 = img5.shape[:2]
        h6, w6 = img6.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

        retva, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        retval, dst1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_OTSU)
        retva2, dst2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)
        retva3, dst3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_OTSU)
        retva4, dst4 = cv2.threshold(gray4, 0, 255, cv2.THRESH_OTSU)
        retva5, dst5 = cv2.threshold(gray5, 0, 255, cv2.THRESH_OTSU)
        retva6, dst6 = cv2.threshold(gray6, 0, 255, cv2.THRESH_OTSU)
        # 相似匹配
        # 列表中所有的6种比较方法   u.zz1()
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                   'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

        res1 = cv2.matchTemplate(dst, dst1, cv2.TM_CCORR_NORMED)
        res2 = cv2.matchTemplate(dst, dst2, cv2.TM_CCORR_NORMED)
        res3 = cv2.matchTemplate(dst, dst3, cv2.TM_CCORR_NORMED)
        res4 = cv2.matchTemplate(dst, dst4, cv2.TM_CCORR_NORMED)
        res5 = cv2.matchTemplate(dst, dst5, cv2.TM_CCORR_NORMED)
        res6 = cv2.matchTemplate(dst, dst6, cv2.TM_CCORR_NORMED)
        # print(res.shape)
        threshold = 0.92
        threshold1 = 0.92

        loc1 = np.where(res1 >= threshold)
        loc2 = np.where(res2 >= threshold)
        loc3 = np.where(res3 >= threshold)
        loc4 = np.where(res4 >= threshold1)
        loc5 = np.where(res5 >= threshold1)
        loc6 = np.where(res6 >= threshold1)

        # 获取最匹配的位置信息
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
        min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)
        min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)
        min_val6, max_val6, min_loc6, max_loc6 = cv2.minMaxLoc(res6)

        img100 = frame.copy()
        if len(loc1[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc1[::-1]):
                print("向左和向右转")
                a1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w1, pt[1] + h1), (0, 255, 0), 1)
        if len(loc2[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc2[::-1]):
                print("直行和向左转弯")
                a2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 1)
        if len(loc3[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc3[::-1]):
                print("直行和向右转弯")
                a3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w3, pt[1] + h3), (0, 255, 0), 1)
        if len(loc4[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc4[::-1]):
                print("向右转弯")
                b1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w4, pt[1] + h4), (0, 255, 0), 1)
        if len(loc5[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc5[::-1]):
                print("向左转弯")
                b2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w5, pt[1] + h5), (0, 255, 0), 1)
        if len(loc6[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc6[::-1]):
                b3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w6, pt[1] + h6), (0, 255, 0), 1)
        if a1 == b3:

            print("急救标志 电话110 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话110匹配错误'")
            tupian = 2
            dy6 = 0
            dy7 = 1
        elif a3 == b3:

            print("急救标志 电话119 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话119匹配错误'")
            tupian = 2
            dy6 = 0
            dy7 = 1
        elif a2 == b3:

            print("急救标志 电话120 正确")
            os.system("espeak -vzh '此区域发现伤员拨打120匹配正确'")
            tupian = 1
            dy6 = 0
            dy7 = 1
        elif a1 == b2:

            print("危险品标志 电话110 正确")
            os.system("espeak -vzh '此区域发现危险品拨打110匹配正确'")
            tupian = 1
            dy6 = 0
            dy7 = 1
        elif a3 == b2:
            print("危险品标志 电话119 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话119匹配错误'")
            tupian = 2
            dy6 = 0
            dy7 = 1
        elif a2 == b2:

            print("危险品标志 电话120 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话120匹配错误'")
            tupian = 2
            dy6 = 0
            dy7 = 1
        if a1 == b1:

            print("火警标志 电话110 错误")
            os.system("espeak -vzh '此区域发现火灾，报警电话110,匹配错误'")
            tupian = 2
            dy6 = 0
            dy7 = 1
        elif a3 == b1:

            print("火警标志 电话119 正确")
            os.system("espeak -vzh '此区域发现火灾，拨打119,匹配正确'")
            tupian = 1
            dy6 = 0
            dy7 = 1
        elif a2 == b1:

            print("火警标志 电话120 错误")
            os.system("espeak -vzh '此区域发现火灾报警电话120匹配错误'")
            tupian = 2
            dy6 = 0
            dy7 = 1
        # cv2.namedWindow("frame", 0)#创立一个创客
        # cv2.resizeWindow("frame",325 ,288)
        cv2.imshow('frame', ju_xing)  # 将图片显示在窗口
        if cv2.waitKey(90) & 0xFF == ord('q'):
            break

    while dy7:
        g.linear.x = 0
        pub.publish(g)
        if tupian == 2:
            u.chuowu()
            dy7 = 0
            dy8 = 1
            tupian = 0
        if tupian == 1:
            u.zhengque()
            dy7 = 0
            dy8 = 1
            tupian = 0
    while dy8:
        print("dy4")
        shijian4 = 28
        while shijian4:
            g.linear.y = 1
            pub.publish(g)
            rospy.sleep(0.1)
            shijian4 -= 1
            dx2 = 1

        dy8 = 0
        dy9 = 1

        g.linear.y = 0
        pub.publish(g)
        rospy.sleep(2)
        u.pufu()
        rospy.sleep(1)

    while dy9:
        a1 = 0
        a2 = 2
        a3 = 3
        b1 = 4
        b2 = 5
        b3 = 6
        c = 0
        d = 0
        e = 0
        b = 0
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        ret, cap = cap1.read()
        cap = cv2.resize(cap, (640, 360))
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)

        img_hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(img_hsv, lower_purple, higher_purple)  # 获得绿色部分掩膜
        mask_yellow = cv2.medianBlur(mask_yellow, 5)  # 中值滤波
        cnts1, hierarchy1 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt1 in cnts1:
            (x, y, w, h) = cv2.boundingRect(cnt1)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong2 = 1
                b = (x + (w / 2)) - 320
        mask_1 = cv2.inRange(img_hsv, lower_1, higher_1)  # 获得绿色部分掩膜
        mask_1 = cv2.medianBlur(mask_1, 5)  # 中值滤波
        cnts, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong2 = 2
                b = (x + (w / 2)) - 320
        if -10 < b < 10:
            print("////////////////3333")
            d = 1
        elif 10 < b:
            g.linear.y = -0.15
            # g.angular.z=-0
            pub.publish(g)
            # rospy.sleep(0.1)
            g.angular.z = 0
            g.linear.y = 0
            print("/////////////////222")
        elif -10 > b:
            g.linear.y = 0.15
            # g.angular.z=-0
            pub.publish(g)
            # g.angular.z=0.0
            rospy.sleep(0.1)
            g.linear.y = 0
            print("//////////////111")
        if d == 1:
            if e > 10000:

                '''if tong==1:
                         dy15=0
                         dy16=1
                         u.pufu()
                if tong ==2 :
                         dy15=0 
                         dy17=1
                         u.pufu()'''
                print("////////////////5555555")
            else:
                g.linear.x = 0.5
                pub.publish(g)
                rospy.sleep(0.1)
                print("//////////////111")
        if e > 10000:

            if tong2 == 1:
                dy9 = 0
                dy10 = 1
                u.pufu()
            if tong2 == 2:
                dy9 = 0
                dy12 = 1
                u.pufu()
            print("////////////////5555555")

        g.linear.x = 0
        g.linear.y = 0
        g.angular.z = 0
        cv2.imshow('cap1', cap)
        cv2.imshow('frame', frame)
        if chr(cv2.waitKey(1) & 255) == 'q':  # 按 q 退出
            cap.terminate()
            break
        if chr(cv2.waitKey(1) & 255) == 'q':  # 按 q 退出
            cap.terminate()
            break
    while dy10:
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        g.linear.x = 0.005
        pub.publish(g)
        # rospy.sleep(0.1)
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)
        img1 = cv2.imread('/home/ysc/learningx/tuku/110.jpg')
        img2 = cv2.imread('/home/ysc/learningx/tuku/120.jpg')
        img3 = cv2.imread('/home/ysc/learningx/tuku/119.jpg')
        img4 = cv2.imread('/home/ysc/learningx/tuku/huo.jpg')
        img5 = cv2.imread('/home/ysc/learningx/tuku/tou.jpg')
        img6 = cv2.imread('/home/ysc/learningx/tuku/sou.jpg')
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        h4, w4 = img4.shape[:2]
        h5, w5 = img5.shape[:2]
        h6, w6 = img6.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

        retva, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        retval, dst1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_OTSU)
        retva2, dst2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)
        retva3, dst3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_OTSU)
        retva4, dst4 = cv2.threshold(gray4, 0, 255, cv2.THRESH_OTSU)
        retva5, dst5 = cv2.threshold(gray5, 0, 255, cv2.THRESH_OTSU)
        retva6, dst6 = cv2.threshold(gray6, 0, 255, cv2.THRESH_OTSU)
        # 相似匹配
        # 列表中所有的6种比较方法   u.zz1()
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                   'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

        res1 = cv2.matchTemplate(dst, dst1, cv2.TM_CCORR_NORMED)
        res2 = cv2.matchTemplate(dst, dst2, cv2.TM_CCORR_NORMED)
        res3 = cv2.matchTemplate(dst, dst3, cv2.TM_CCORR_NORMED)
        res4 = cv2.matchTemplate(dst, dst4, cv2.TM_CCORR_NORMED)
        res5 = cv2.matchTemplate(dst, dst5, cv2.TM_CCORR_NORMED)
        res6 = cv2.matchTemplate(dst, dst6, cv2.TM_CCORR_NORMED)
        # print(res.shape)
        threshold = 0.92
        threshold1 = 0.92
        loc1 = np.where(res1 >= threshold)
        loc2 = np.where(res2 >= threshold)
        loc3 = np.where(res3 >= threshold)
        loc4 = np.where(res4 >= threshold1)
        loc5 = np.where(res5 >= threshold1)
        loc6 = np.where(res6 >= threshold1)

        # 获取最匹配的位置信息
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
        min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)
        min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)
        min_val6, max_val6, min_loc6, max_loc6 = cv2.minMaxLoc(res6)

        img100 = frame.copy()
        if len(loc1[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc1[::-1]):
                print("向左和向右转")
                a1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w1, pt[1] + h1), (0, 255, 0), 1)
        if len(loc2[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc2[::-1]):
                print("直行和向左转弯")
                a2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 1)
        if len(loc3[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc3[::-1]):
                print("直行和向右转弯")
                a3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w3, pt[1] + h3), (0, 255, 0), 1)
        if len(loc4[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc4[::-1]):
                print("向右转弯")
                b1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w4, pt[1] + h4), (0, 255, 0), 1)
        if len(loc5[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc5[::-1]):
                print("向左转弯")
                b2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w5, pt[1] + h5), (0, 255, 0), 1)
        if len(loc6[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc6[::-1]):
                b3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w6, pt[1] + h6), (0, 255, 0), 1)
        if a1 == b3:

            print("急救标志 电话110 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话110匹配错误'")
            tupian = 2
            dy10 = 0
            dy11 = 1
        elif a3 == b3:

            print("急救标志 电话119 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话119匹配错误'")
            tupian = 2
            dy10 = 0
            dy11 = 1
        elif a2 == b3:

            print("急救标志 电话120 正确")
            os.system("espeak -vzh '此区域发现伤员拨打120匹配正确'")
            tupian = 1
            dy10 = 0
            dy11 = 1
        elif a1 == b2:

            print("危险品标志 电话110 正确")
            os.system("espeak -vzh '此区域发现危险品拨打110匹配正确'")
            tupian = 1
            dy10 = 0
            dy11 = 1
        elif a3 == b2:
            print("危险品标志 电话119 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话119匹配错误'")
            tupian = 2
            dy10 = 0
            dy11 = 1
        elif a2 == b2:

            print("危险品标志 电话120 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话120匹配错误'")
            tupian = 2
            dy10 = 0
            dy11 = 1
        if a1 == b1:

            print("火警标志 电话110 错误")
            os.system("espeak -vzh '此区域发现火灾，报警电话110,匹配错误'")
            tupian = 2
            dy10 = 0
            dy11 = 1
        elif a3 == b1:

            print("火警标志 电话119 正确")
            os.system("espeak -vzh '此区域发现火灾，拨打119,匹配正确'")
            tupian = 1
            dy10 = 0
            dy11 = 1
        elif a2 == b1:

            print("火警标志 电话120 错误")
            os.system("espeak -vzh '此区域发现火灾报警电话120匹配错误'")
            tupian = 2
            dy10 = 0
            dy11 = 1
        # cv2.namedWindow("frame", 0)#创立一个创客
        # cv2.resizeWindow("frame",325 ,288)
        cv2.imshow('frame', ju_xing)  # 将图片显示在窗口
        if cv2.waitKey(90) & 0xFF == ord('q'):
            break

    while dy11:
        if tupian == 2:
            u.chuowu()
            dy11 = 0
            dy12 = 1
            tupian = 0
        if tupian == 1:
            u.zhengque()
            dy11 = 0
            dy12 = 1
            tupian = 0
    while dy12:
        print("dy4")
        shijian4 = 25
        while shijian4:
            g.linear.y = 1
            pub.publish(g)
            rospy.sleep(0.1)
            shijian4 -= 1
            dx2 = 1
        g.linear.y = 0
        g.linear.x = 0
        dy12 = 0
        dy13 = 1
    while dy13:
        print("dy13")
        shijian4 = 60
        while shijian4:
            g.linear.x = 0.5
            pub.publish(g)
            rospy.sleep(0.1)
            g.linear.x = 0
            shijian4 -= 1

        shijian1 = 61
        while shijian1:
            g.angular.z = 0.5
            pub.publish(g)
            rospy.sleep(0.1)
            g.angular.z = 0
            shijian1 -= 1

        dy13 = 0
        dy14 = 1
    while dy14:

        print("dy13")
        shijian4 = 60
        while shijian4:
            g.linear.y = 0.1
            pub.publish(g)
            rospy.sleep(0.1)
            g.linear.y = 0
            shijian4 -= 1
        dy14 = 0
        dy15 = 1
        pub.publish(g)
        rospy.sleep(0.1)
        u.pufu()
        rospy.sleep(1)
    while dy15:
        a1 = 0
        a2 = 2
        a3 = 3
        b1 = 4
        b2 = 5
        b3 = 6
        c = 0
        d = 0
        e = 0
        b = 0
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        ret, cap = cap1.read()
        cap = cv2.resize(cap, (640, 360))
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)

        img_hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(img_hsv, lower_purple, higher_purple)  # 获得绿色部分掩膜
        mask_yellow = cv2.medianBlur(mask_yellow, 5)  # 中值滤波
        cnts1, hierarchy1 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt1 in cnts1:
            (x, y, w, h) = cv2.boundingRect(cnt1)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong2 = 1
                b = (x + (w / 2)) - 320
        mask_1 = cv2.inRange(img_hsv, lower_1, higher_1)  # 获得绿色部分掩膜
        mask_1 = cv2.medianBlur(mask_1, 5)  # 中值滤波
        cnts, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong2 = 2
                b = (x + (w / 2)) - 320
        if -10 < b < 10:
            print("////////////////3333")
            d = 1
        elif 10 < b:
            g.linear.y = -0.15
            # g.angular.z=-0
            pub.publish(g)
            # rospy.sleep(0.1)
            g.angular.z = 0
            g.linear.y = 0
            print("/////////////////222")
        elif -10 > b:
            g.linear.y = 0.15
            # g.angular.z=-0
            pub.publish(g)
            # g.angular.z=0.0
            rospy.sleep(0.1)
            g.linear.y = 0
            print("//////////////111")
        if d == 1:
            if e > 10000:

                '''if tong==1:
                         dy15=0
                         dy16=1
                         u.pufu()
                if tong ==2 :
                         dy15=0 
                         dy17=1
                         u.pufu()'''
                print("////////////////5555555")
            else:
                g.linear.x = 0.5
                pub.publish(g)
                rospy.sleep(0.1)
                print("//////////////111")
        if e > 10000:

            if tong2 == 1:
                dy15 = 0
                dy16 = 1

                g.linear.x = 0.0
                pub.publish(g)
                rospy.sleep(1)
                u.pufu()
                rospy.sleep(1)
            if tong2 == 2:
                dy15 = 0
                dy18 = 1
                g.linear.x = 0.0
                pub.publish(g)
                rospy.sleep(1)
                u.pufu()
                rospy.sleep(1)
            print("////////////////5555555")

        g.linear.x = 0
        g.linear.y = 0
        g.angular.z = 0
        cv2.imshow('cap1', cap)
        cv2.imshow('frame', frame)
        if chr(cv2.waitKey(1) & 255) == 'q':  # 按 q 退出
            cap.terminate()
            break

    while dy16:

        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        g.linear.x = 0.005
        pub.publish(g)
        # rospy.sleep(0.1)
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)
        img1 = cv2.imread('/home/ysc/learningx/tuku/110.jpg')
        img2 = cv2.imread('/home/ysc/learningx/tuku/120.jpg')
        img3 = cv2.imread('/home/ysc/learningx/tuku/119.jpg')
        img4 = cv2.imread('/home/ysc/learningx/tuku/huo.jpg')
        img5 = cv2.imread('/home/ysc/learningx/tuku/tou.jpg')
        img6 = cv2.imread('/home/ysc/learningx/tuku/sou.jpg')
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        h4, w4 = img4.shape[:2]
        h5, w5 = img5.shape[:2]
        h6, w6 = img6.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

        retva, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        retval, dst1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_OTSU)
        retva2, dst2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)
        retva3, dst3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_OTSU)
        retva4, dst4 = cv2.threshold(gray4, 0, 255, cv2.THRESH_OTSU)
        retva5, dst5 = cv2.threshold(gray5, 0, 255, cv2.THRESH_OTSU)
        retva6, dst6 = cv2.threshold(gray6, 0, 255, cv2.THRESH_OTSU)
        # 相似匹配
        # 列表中所有的6种比较方法   u.zz1()
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                   'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

        res1 = cv2.matchTemplate(dst, dst1, cv2.TM_CCORR_NORMED)
        res2 = cv2.matchTemplate(dst, dst2, cv2.TM_CCORR_NORMED)
        res3 = cv2.matchTemplate(dst, dst3, cv2.TM_CCORR_NORMED)
        res4 = cv2.matchTemplate(dst, dst4, cv2.TM_CCORR_NORMED)
        res5 = cv2.matchTemplate(dst, dst5, cv2.TM_CCORR_NORMED)
        res6 = cv2.matchTemplate(dst, dst6, cv2.TM_CCORR_NORMED)
        # print(res.shape)
        threshold = 0.92
        threshold1 = 0.92
        loc1 = np.where(res1 >= threshold)
        loc2 = np.where(res2 >= threshold)
        loc3 = np.where(res3 >= threshold)
        loc4 = np.where(res4 >= threshold1)
        loc5 = np.where(res5 >= threshold1)
        loc6 = np.where(res6 >= threshold1)

        # 获取最匹配的位置信息
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
        min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)
        min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)
        min_val6, max_val6, min_loc6, max_loc6 = cv2.minMaxLoc(res6)

        img100 = frame.copy()
        if len(loc1[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc1[::-1]):
                print("向左和向右转")
                a1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w1, pt[1] + h1), (0, 255, 0), 1)
        if len(loc2[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc2[::-1]):
                print("直行和向左转弯")
                a2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 1)
        if len(loc3[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc3[::-1]):
                print("直行和向右转弯")
                a3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w3, pt[1] + h3), (0, 255, 0), 1)
        if len(loc4[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc4[::-1]):
                print("向右转弯")
                b1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w4, pt[1] + h4), (0, 255, 0), 1)
        if len(loc5[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc5[::-1]):
                print("向左转弯")
                b2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w5, pt[1] + h5), (0, 255, 0), 1)
        if len(loc6[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc6[::-1]):
                b3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w6, pt[1] + h6), (0, 255, 0), 1)
        if a1 == b3:

            print("急救标志 电话110 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话110匹配错误'")
            tupian = 2
            dy16 = 0
            dy17 = 1
        elif a3 == b3:

            print("急救标志 电话119 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话119匹配错误'")
            tupian = 2
            dy16 = 0
            dy17 = 1
        elif a2 == b3:

            print("急救标志 电话120 正确")
            os.system("espeak -vzh '此区域发现伤员拨打120匹配正确'")
            tupian = 1
            dy16 = 0
            dy17 = 1
        elif a1 == b2:

            print("危险品标志 电话110 正确")
            os.system("espeak -vzh '此区域发现危险品拨打110匹配正确'")
            tupian = 1
            dy16 = 0
            dy17 = 1
        elif a3 == b2:
            print("危险品标志 电话119 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话119匹配错误'")
            tupian = 2
            dy16 = 0
            dy17 = 1
        elif a2 == b2:

            print("危险品标志 电话120 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话120匹配错误'")
            tupian = 2
            dy16 = 0
            dy17 = 1
        if a1 == b1:

            print("火警标志 电话110 错误")
            os.system("espeak -vzh '此区域发现火灾，报警电话110,匹配错误'")
            tupian = 2
            dy16 = 0
            dy17 = 1
        elif a3 == b1:

            print("火警标志 电话119 正确")
            os.system("espeak -vzh '此区域发现火灾，拨打119,匹配正确'")
            tupian = 1
            dy16 = 0
            dy17 = 1
        elif a2 == b1:

            print("火警标志 电话120 错误")
            os.system("espeak -vzh '此区域发现火灾报警电话120匹配错误'")
            tupian = 2
            dy16 = 0
            dy17 = 1
        # cv2.namedWindow("frame", 0)#创立一个创客
        # cv2.resizeWindow("frame",325 ,288)
        cv2.imshow('frame', ju_xing)  # 将图片显示在窗口
        if cv2.waitKey(90) & 0xFF == ord('q'):
            break
    while dy17:
        if tupian == 2:
            u.chuowu()
            dy17 = 0
            dy18 = 1
            tupian = 0
        if tupian == 1:
            u.zhengque()
            dy17 = 0
            dy18 = 1
            tupian = 0
    while dy18:
        print("dy4")
        shijian4 = 25
        while shijian4:
            g.linear.y = 1
            pub.publish(g)
            rospy.sleep(0.1)
            shijian4 -= 1
            dx2 = 1

        dy18 = 0
        dy19 = 1

        g.linear.y = 0
        pub.publish(g)
        rospy.sleep(2)
        u.pufu()
        rospy.sleep(1)
    while dy19:
        a1 = 0
        a2 = 2
        a3 = 3
        b1 = 4
        b2 = 5
        b3 = 6
        c = 0
        d = 0
        e = 0
        b = 0
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        ret, cap = cap1.read()
        cap = cv2.resize(cap, (640, 360))
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)

        img_hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(img_hsv, lower_purple, higher_purple)  # 获得绿色部分掩膜
        mask_yellow = cv2.medianBlur(mask_yellow, 5)  # 中值滤波
        cnts1, hierarchy1 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt1 in cnts1:
            (x, y, w, h) = cv2.boundingRect(cnt1)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong2 = 1
                b = (x + (w / 2)) - 320
        mask_1 = cv2.inRange(img_hsv, lower_1, higher_1)  # 获得绿色部分掩膜
        mask_1 = cv2.medianBlur(mask_1, 5)  # 中值滤波
        cnts, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong2 = 2
                b = (x + (w / 2)) - 320
        if -10 < b < 10:
            print("////////////////3333")
            d = 1
        elif 10 < b:
            g.linear.y = -0.15
            # g.angular.z=-0
            pub.publish(g)
            # rospy.sleep(0.1)
            g.angular.z = 0
            g.linear.y = 0
            print("/////////////////222")
        elif -10 > b:
            g.linear.y = 0.15
            # g.angular.z=-0
            pub.publish(g)
            # g.angular.z=0.0
            rospy.sleep(0.1)
            g.linear.y = 0
            print("//////////////111")
        if d == 1:
            if e > 10000:

                '''if tong==1:
                         dy15=0
                         dy16=1
                         u.pufu()
                if tong ==2 :
                         dy15=0 
                         dy17=1
                         u.pufu()'''
                print("////////////////5555555")
            else:
                g.linear.x = 0.5
                pub.publish(g)
                rospy.sleep(0.1)
                print("//////////////111")
        if e > 10000:

            if tong2 == 1:
                dy19 = 0
                dy20 = 1

                g.linear.x = 0.0
                pub.publish(g)
                rospy.sleep(1)
                u.pufu()
                rospy.sleep(1)
            if tong2 == 2:
                dy19 = 0
                dy22 = 1
                g.linear.x = 0.0
                pub.publish(g)
                rospy.sleep(1)
                u.pufu()
                rospy.sleep(1)
            print("////////////////5555555")

        g.linear.x = 0
        g.linear.y = 0
        g.angular.z = 0
        cv2.imshow('cap1', cap)
        cv2.imshow('frame', frame)
        if chr(cv2.waitKey(1) & 255) == 'q':  # 按 q 退出
            cap.terminate()
            break
    while dy20:
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        g.linear.x = 0.005
        pub.publish(g)
        # rospy.sleep(0.1)
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)
        img1 = cv2.imread('/home/ysc/learningx/tuku/110.jpg')
        img2 = cv2.imread('/home/ysc/learningx/tuku/120.jpg')
        img3 = cv2.imread('/home/ysc/learningx/tuku/119.jpg')
        img4 = cv2.imread('/home/ysc/learningx/tuku/huo.jpg')
        img5 = cv2.imread('/home/ysc/learningx/tuku/tou.jpg')
        img6 = cv2.imread('/home/ysc/learningx/tuku/sou.jpg')
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        h4, w4 = img4.shape[:2]
        h5, w5 = img5.shape[:2]
        h6, w6 = img6.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

        retva, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        retval, dst1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_OTSU)
        retva2, dst2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)
        retva3, dst3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_OTSU)
        retva4, dst4 = cv2.threshold(gray4, 0, 255, cv2.THRESH_OTSU)
        retva5, dst5 = cv2.threshold(gray5, 0, 255, cv2.THRESH_OTSU)
        retva6, dst6 = cv2.threshold(gray6, 0, 255, cv2.THRESH_OTSU)
        # 相似匹配
        # 列表中所有的6种比较方法   u.zz1()
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                   'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

        res1 = cv2.matchTemplate(dst, dst1, cv2.TM_CCORR_NORMED)
        res2 = cv2.matchTemplate(dst, dst2, cv2.TM_CCORR_NORMED)
        res3 = cv2.matchTemplate(dst, dst3, cv2.TM_CCORR_NORMED)
        res4 = cv2.matchTemplate(dst, dst4, cv2.TM_CCORR_NORMED)
        res5 = cv2.matchTemplate(dst, dst5, cv2.TM_CCORR_NORMED)
        res6 = cv2.matchTemplate(dst, dst6, cv2.TM_CCORR_NORMED)
        # print(res.shape)
        threshold = 0.92
        threshold1 = 0.92
        loc1 = np.where(res1 >= threshold)
        loc2 = np.where(res2 >= threshold)
        loc3 = np.where(res3 >= threshold)
        loc4 = np.where(res4 >= threshold1)
        loc5 = np.where(res5 >= threshold1)
        loc6 = np.where(res6 >= threshold1)

        # 获取最匹配的位置信息
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
        min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)
        min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)
        min_val6, max_val6, min_loc6, max_loc6 = cv2.minMaxLoc(res6)

        img100 = frame.copy()
        if len(loc1[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc1[::-1]):
                print("向左和向右转")
                a1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w1, pt[1] + h1), (0, 255, 0), 1)
        if len(loc2[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc2[::-1]):
                print("直行和向左转弯")
                a2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 1)
        if len(loc3[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc3[::-1]):
                print("直行和向右转弯")
                a3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w3, pt[1] + h3), (0, 255, 0), 1)
        if len(loc4[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc4[::-1]):
                print("向右转弯")
                b1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w4, pt[1] + h4), (0, 255, 0), 1)
        if len(loc5[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc5[::-1]):
                print("向左转弯")
                b2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w5, pt[1] + h5), (0, 255, 0), 1)
        if len(loc6[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc6[::-1]):
                b3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w6, pt[1] + h6), (0, 255, 0), 1)
        if a1 == b3:

            print("急救标志 电话110 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话110匹配错误'")
            tupian = 2
            dy20 = 0
            dy21 = 1
        elif a3 == b3:

            print("急救标志 电话119 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话119匹配错误'")
            tupian = 2
            dy20 = 0
            dy21 = 1
        elif a2 == b3:

            print("急救标志 电话120 正确")
            os.system("espeak -vzh '此区域发现伤员拨打120匹配正确'")
            tupian = 1
            dy20 = 0
            dy21 = 1
        elif a1 == b2:

            print("危险品标志 电话110 正确")
            os.system("espeak -vzh '此区域发现危险品拨打110匹配正确'")
            tupian = 1
            dy20 = 0
            dy21 = 1
        elif a3 == b2:
            print("危险品标志 电话119 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话119匹配错误'")
            tupian = 2
            dy20 = 0
            dy21 = 1
        elif a2 == b2:

            print("危险品标志 电话120 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话120匹配错误'")
            tupian = 2
            dy20 = 0
            dy21 = 1
        if a1 == b1:

            print("火警标志 电话110 错误")
            os.system("espeak -vzh '此区域发现火灾，报警电话110,匹配错误'")
            tupian = 2
            dy20 = 0
            dy21 = 1
        elif a3 == b1:

            print("火警标志 电话119 正确")
            os.system("espeak -vzh '此区域发现火灾，拨打119,匹配正确'")
            tupian = 1
            dy20 = 0
            dy21 = 1
        elif a2 == b1:

            print("火警标志 电话120 错误")
            os.system("espeak -vzh '此区域发现火灾报警电话120匹配错误'")
            tupian = 2
            dy20 = 0
            dy21 = 1
        # cv2.namedWindow("frame", 0)#创立一个创客
        # cv2.resizeWindow("frame",325 ,288)
        cv2.imshow('frame', ju_xing)  # 将图片显示在窗口
        if cv2.waitKey(90) & 0xFF == ord('q'):
            break
    while dy21:
        if tupian == 2:
            u.chuowu()
            dy21 = 0
            dy22 = 1
            tupian = 0
        if tupian == 1:
            u.zhengque()
            dy21 = 0
            dy22 = 1
            tupian = 0
    while dy22:
        print("dy4")
        shijian4 = 70
        while shijian4:
            g.linear.y = 0.1
            pub.publish(g)
            rospy.sleep(0.1)
            shijian4 -= 1
            dx2 = 1

        dy22 = 0
        dy23 = 1

        g.linear.y = 0
        pub.publish(g)
        rospy.sleep(2)
        u.pufu()
        rospy.sleep(1)
    while dy23:
        a1 = 0
        a2 = 2
        a3 = 3
        b1 = 4
        b2 = 5
        b3 = 6
        c = 0
        d = 0
        e = 0
        b = 0
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        ret, cap = cap1.read()
        cap = cv2.resize(cap, (640, 360))
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)

        img_hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(img_hsv, lower_purple, higher_purple)  # 获得绿色部分掩膜
        mask_yellow = cv2.medianBlur(mask_yellow, 5)  # 中值滤波
        cnts1, hierarchy1 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt1 in cnts1:
            (x, y, w, h) = cv2.boundingRect(cnt1)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong2 = 1
                b = (x + (w / 2)) - 320
        mask_1 = cv2.inRange(img_hsv, lower_1, higher_1)  # 获得绿色部分掩膜
        mask_1 = cv2.medianBlur(mask_1, 5)  # 中值滤波
        cnts, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
            if w * h > 1500:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                tong2 = 2
                b = (x + (w / 2)) - 320
        if -10 < b < 10:
            print("////////////////3333")
            d = 1
        elif 10 < b:
            g.linear.y = -0.15
            # g.angular.z=-0
            pub.publish(g)
            # rospy.sleep(0.1)
            g.angular.z = 0
            g.linear.y = 0
            print("/////////////////222")
        elif -10 > b:
            g.linear.y = 0.15
            # g.angular.z=-0
            pub.publish(g)
            # g.angular.z=0.0
            rospy.sleep(0.1)
            g.linear.y = 0
            print("//////////////111")
        if d == 1:
            if e > 10000:

                '''if tong==1:
                         dy15=0
                         dy16=1
                         u.pufu()
                if tong ==2 :
                         dy15=0 
                         dy17=1
                         u.pufu()'''
                print("////////////////5555555")
            else:
                g.linear.x = 0.5
                pub.publish(g)
                rospy.sleep(0.1)
                print("//////////////111")
        if e > 10000:

            if tong2 == 1:
                dy23 = 0
                dy24 = 1

                g.linear.x = 0.0
                pub.publish(g)
                rospy.sleep(1)
                u.pufu()
                rospy.sleep(1)
            if tong2 == 2:
                dy23 = 0
                dy26 = 1
                g.linear.x = 0.0
                pub.publish(g)
                rospy.sleep(1)
                u.pufu()
                rospy.sleep(1)
            print("////////////////5555555")

        g.linear.x = 0
        g.linear.y = 0
        g.angular.z = 0
        cv2.imshow('cap1', cap)
        cv2.imshow('frame', frame)
        if chr(cv2.waitKey(1) & 255) == 'q':  # 按 q 退出
            cap.terminate()
            break
    while dy24:
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        g.linear.x = 0.005
        pub.publish(g)
        # rospy.sleep(0.1)
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)
        img1 = cv2.imread('/home/ysc/learningx/tuku/110.jpg')
        img2 = cv2.imread('/home/ysc/learningx/tuku/120.jpg')
        img3 = cv2.imread('/home/ysc/learningx/tuku/119.jpg')
        img4 = cv2.imread('/home/ysc/learningx/tuku/huo.jpg')
        img5 = cv2.imread('/home/ysc/learningx/tuku/tou.jpg')
        img6 = cv2.imread('/home/ysc/learningx/tuku/sou.jpg')
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        h4, w4 = img4.shape[:2]
        h5, w5 = img5.shape[:2]
        h6, w6 = img6.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 读取为灰度图像
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

        retva, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        retval, dst1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_OTSU)
        retva2, dst2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)
        retva3, dst3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_OTSU)
        retva4, dst4 = cv2.threshold(gray4, 0, 255, cv2.THRESH_OTSU)
        retva5, dst5 = cv2.threshold(gray5, 0, 255, cv2.THRESH_OTSU)
        retva6, dst6 = cv2.threshold(gray6, 0, 255, cv2.THRESH_OTSU)
        # 相似匹配
        # 列表中所有的6种比较方法   u.zz1()
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                   'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

        res1 = cv2.matchTemplate(dst, dst1, cv2.TM_CCORR_NORMED)
        res2 = cv2.matchTemplate(dst, dst2, cv2.TM_CCORR_NORMED)
        res3 = cv2.matchTemplate(dst, dst3, cv2.TM_CCORR_NORMED)
        res4 = cv2.matchTemplate(dst, dst4, cv2.TM_CCORR_NORMED)
        res5 = cv2.matchTemplate(dst, dst5, cv2.TM_CCORR_NORMED)
        res6 = cv2.matchTemplate(dst, dst6, cv2.TM_CCORR_NORMED)
        # print(res.shape)
        threshold = 0.92
        threshold1 = 0.92
        loc1 = np.where(res1 >= threshold)
        loc2 = np.where(res2 >= threshold)
        loc3 = np.where(res3 >= threshold)
        loc4 = np.where(res4 >= threshold1)
        loc5 = np.where(res5 >= threshold1)
        loc6 = np.where(res6 >= threshold1)

        # 获取最匹配的位置信息
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
        min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)
        min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)
        min_val6, max_val6, min_loc6, max_loc6 = cv2.minMaxLoc(res6)

        img100 = frame.copy()
        if len(loc1[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc1[::-1]):
                print("向左和向右转")
                a1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w1, pt[1] + h1), (0, 255, 0), 1)
        if len(loc2[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc2[::-1]):
                print("直行和向左转弯")
                a2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 1)
        if len(loc3[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc3[::-1]):
                print("直行和向右转弯")
                a3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w3, pt[1] + h3), (0, 255, 0), 1)
        if len(loc4[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc4[::-1]):
                print("向右转弯")
                b1 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w4, pt[1] + h4), (0, 255, 0), 1)
        if len(loc5[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc5[::-1]):
                print("向左转弯")
                b2 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w5, pt[1] + h5), (0, 255, 0), 1)
        if len(loc6[0]) == 0:
            ju_xing = cv2.rectangle(img100, (0, 0), (0, 0), (0, 255, 0), 1)
        else:
            for pt in zip(*loc6[::-1]):
                b3 = 1
                ju_xing = cv2.rectangle(img100, pt, (pt[0] + w6, pt[1] + h6), (0, 255, 0), 1)
        if a1 == b3:

            print("急救标志 电话110 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话110匹配错误'")
            tupian = 2
            dy24 = 0
            dy25 = 1
        elif a3 == b3:

            print("急救标志 电话119 错误")
            os.system("espeak -vzh '此区域发现伤员报警电话119匹配错误'")
            tupian = 2
            dy24 = 0
            dy25 = 1
        elif a2 == b3:

            print("急救标志 电话120 正确")
            os.system("espeak -vzh '此区域发现伤员拨打120匹配正确'")
            tupian = 1
            dy24 = 0
            dy25 = 1
        elif a1 == b2:

            print("危险品标志 电话110 正确")
            os.system("espeak -vzh '此区域发现危险品拨打110匹配正确'")
            tupian = 1
            dy24 = 0
            dy25 = 1
        elif a3 == b2:
            print("危险品标志 电话119 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话119匹配错误'")
            tupian = 2
            dy24 = 0
            dy25 = 1
        elif a2 == b2:

            print("危险品标志 电话120 错误")
            os.system("espeak -vzh '此区域发现危险品报警电话120匹配错误'")
            tupian = 2
            dy24 = 0
            dy25 = 1
        if a1 == b1:

            print("火警标志 电话110 错误")
            os.system("espeak -vzh '此区域发现火灾，报警电话110,匹配错误'")
            tupian = 2
            dy24 = 0
            dy25 = 1
        elif a3 == b1:

            print("火警标志 电话119 正确")
            os.system("espeak -vzh '此区域发现火灾，拨打119,匹配正确'")
            tupian = 1
            dy24 = 0
            dy25 = 1
        elif a2 == b1:

            print("火警标志 电话120 错误")
            os.system("espeak -vzh '此区域发现火灾报警电话120匹配错误'")
            tupian = 2
            dy24 = 0
            dy25 = 1
        # cv2.namedWindow("frame", 0)#创立一个创客
        # cv2.resizeWindow("frame",325 ,288)
        cv2.imshow('frame', ju_xing)  # 将图片显示在窗口
        if cv2.waitKey(90) & 0xFF == ord('q'):
            break
    while dy25:
        if tupian == 2:
            u.chuowu()
            dy25 = 0
            dy26 = 1
            tupian = 0
        if tupian == 1:
            u.zhengque()
            dy25 = 0
            dy26 = 1
            tupian = 0

    while dy26:
        shijian1 = 60
        while shijian1:
            g.angular.z = 0.5
            pub.publish(g)
            rospy.sleep(0.1)
            g.angular.z = 0
            shijian1 -= 1

        dy26 = 0
        dy27 = 1
    while dy27:
        a1 = 0
        a2 = 2
        a3 = 3
        b1 = 4
        b2 = 5
        b3 = 6
        c = 0
        d = 0
        e = 0
        b = 0
        frame = frame1.read()
        frame = cv2.resize(frame, (640, 360))
        ret, cap = cap1.read()
        cap = cv2.resize(cap, (640, 360))
        cv2.namedWindow("dst1", 0)
        cv2.resizeWindow("dst1", 640, 360)

        img_hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(img_hsv, lower_2, higher_2)  # 获得绿色部分掩膜
        mask_yellow = cv2.medianBlur(mask_yellow, 5)  # 中值滤波
        cnts1, hierarchy1 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt1 in cnts1:
            (x, y, w, h) = cv2.boundingRect(cnt1)  # 该函数返回矩阵四个点
            if w * h > 50:
                cv2.rectangle(cap, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
                cv2.putText(cap, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
                print(w * h)
                e = w * h
                b = (x + (w / 2)) - 320
        if 70 < b < 90:
            print("////////////////3333")
            d = 1
        elif 90 < b:
            g.linear.y = -0.1
            # g.angular.z=-0
            pub.publish(g)
            # rospy.sleep(0.1)
            g.angular.z = 0
            g.linear.y = 0
            print("/////////////////222")
        elif 70 > b:
            g.linear.y = 0.1
            # g.angular.z=-0
            pub.publish(g)
            # g.angular.z=0.0
            rospy.sleep(0.1)
            g.linear.y = 0
            print("//////////////111")
        if d == 1:
            if e > 10000:

                '''if tong==1:
                         dy15=0
                         dy16=1
                         u.pufu()
                if tong ==2 :
                         dy15=0 
                         dy17=1
                         u.pufu()'''
                print("////////////////5555555")
            else:
                g.linear.x = 0.1
                pub.publish(g)
                rospy.sleep(0.1)
                print("//////////////111")
        if e > 7000:
            dy27 = 0
            dy28 = 1

            g.linear.x = 0.0
            pub.publish(g)
            rospy.sleep(2)
            print("////////////////5555555")

        g.linear.x = 0
        g.linear.y = 0
        g.angular.z = 0
        cv2.imshow('cap1', cap)
        if chr(cv2.waitKey(1) & 255) == 'q':  # 按 q 退出
            cap.terminate()
            break
    while dy28:
        shijian1 = 9
        while shijian1:
            g.linear.x = 8
            pub.publish(g)
            rospy.sleep(0.1)
            g.linear.x = 0
            shijian1 -= 1

        dy28 = 0
        dy29 = 1

    while dy29:
        print("wanchengla")
        break

cv2.waitKey(0)
cv2.destroyAllwindows()
rospy.spin()

