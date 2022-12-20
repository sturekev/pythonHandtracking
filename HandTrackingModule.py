import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectioCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectioCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(imgRGB)

        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img
                # for id, lm in enumerate(handlms.landmark):
                #     # print(id,lm)
                #     h, w , c = img.shape
                #     cx , cy = int(lm.x *w) , int(lm.y *h)
                #     # print(id ,cx, cy)
                #     if id == 0:
                #         cv2.circle(img, (cx,cy), 25, (255,0,225), cv2.FILLED)


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        
        img = detector.findHands(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (18,70), cv2.FONT_HERSHEY_PLAIN,3,(255,9,255), 3 )


        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()