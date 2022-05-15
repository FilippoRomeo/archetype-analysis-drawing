import cv2
import math
import numpy as np
import pandas as pd

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
points = np.array([[0.3791454386035252, 0.5089704263689607], [0.4983802415059109, 0.4865878212776629], [0.4191061040406586, 0.4890729258496474], [0.48898375092596835, 0.6904554156787046], [0.41117320428962, 0.6855686449973655], [0.48969027909831686, 0.8806483247709954], [0.4096722346480175, 0.8725103831012889], [0.45146556567120294, 0.216198952126905], [0.6304876750748412, 0.1994776546413951], [0.6406976694235704, 0.1861724655606558], [0.6199918357274865, 0.18561325370105788], [0.6525936779272056, 0.201758477474465], [0.6013198509477334, 0.20041966221830415], [0.6683290543094758, 0.29699362669473495], [0.5645238852104717, 0.3113999818240313], [0.6545654774178274, 0.49620430200480303], [0.5898070573107588, 0.49659117464889346], [0.6592482998457356, 0.6834740545963035], [0.5840631897032319, 0.6828527784533074], [0.6408640096147972, 0.8299668209407426], [0.5829181988101784, 0.8173392725052692], [0.6197806290284397, 0.30050890733295843], [0.8252923243905792, 0.23409826375167195], [0.835683753646597, 0.2185883280832016], [0.8131540844750428, 0.21904862499113367], [0.8506741192799976, 0.2279991219170517], [0.7959142481709739, 0.22725381616179272], [0.8733570624656342, 0.3256920048853457], [0.7652207837892534, 0.3239122878098148], [0.893097550288673, 0.44273291363944955], [0.7346131146711571, 0.4430594635999311], [0.902709244982588, 0.5343829401117663], [0.8520378940615836, 0.543215423861057], [0.7842126810888624, 0.5430821914771806], [0.8496391467917583, 0.7170072127563635], [0.7934480818135997, 0.7157067918591926], [0.8415470663986131, 0.8790693270711738], [0.7969306654944098, 0.8786970205344115], [0.8191112469834433, 0.32444646417244244], [0.4544294400182521, 0.10802826838116084], [0.4652589441860643, 0.09470838455219986], [0.44184697991125976, 0.09401847354478254], [0.4784184639521475, 0.1113126386155105], [0.42270482157448985, 0.10977393520172159], [0.5101597581790689, 0.21719483055184013], [0.39370939342390643, 0.21645334444157344], [0.3703281257159549, 0.34746637604116004]], np.float64)

def calculateDistance(x1,y1,x2,y2):  
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    return dist  

def defineCornerValues(arr,value):
    for i in arr:
            if i[0] == value:
                a = np.where(ids==i)
                return a[0][0]

# Load Cap
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#Reading the first frame
(grabbed, img) = cap.read()

pana = []
tilta = []
pan  = np.pi/2
tilt = np.pi/2

while cap.isOpened(): 
    grabbed, img = cap.read()

    cv2.namedWindow('img')
   
    # Get Aruco marker
    
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    corners2 = np.array([c[0] for c in corners])

    # if corners:
    if len(corners)>=4:

        posArrTopl = defineCornerValues(ids, 0)
        posArrTopr = defineCornerValues(ids, 1)
        posArrBotl = defineCornerValues(ids, 2)
        posArrBotr = defineCornerValues(ids, 3)

        topl = corners[posArrTopl]
        topr = corners[posArrTopr]
        botl = corners[posArrBotl]
        botr = corners[posArrBotr]

        data = pd.DataFrame({"x": corners2[:,:,0].flatten(), "y": corners2[:,:,1].flatten()}, index = pd.MultiIndex.from_product(
                        [ids.flatten(), ["c{0}".format(i)
                        for i in np.arange(4)+1]], 
                        names = ["marker", ""]))
             

        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 100  

        Dist = []

        toplxy = ((topl[0][2][0]).astype(int), (topl[0][2][1]).astype(int))
        toprxy = ((topr[0][3][0]).astype(int), (topr[0][3][1]).astype(int))
        botlxy = ((botl[0][0][0]).astype(int), (botl[0][0][1]).astype(int))
        botrxy = ((botr[0][0][0]).astype(int), (botr[0][0][1]).astype(int))

        # red
        cv2.line(img, toplxy, toprxy, (255, 0, 0), 2)
        # yellow
        cv2.line(img, toprxy, botlxy, (255, 253, 0), 2)
        # pink
        cv2.line(img, botlxy, botrxy, (255, 2, 255), 2)
        # green
        cv2.line(img, botrxy, toplxy, (0, 255, 0), 2)

        wX = int((toplxy[0] + toprxy[0])//2)
        wY = int((toplxy[1] + toprxy[1])//2)
        hX = int((toplxy[0] + botrxy[0])//2)
        hY = int((toplxy[1] + botrxy[1])//2)

        
        # width center point
        cv2.circle(img, (wX, wY), 4, (255, 255, 0), -1)
        # heigh center point
        cv2.circle(img, (hX, hY), 4, (255, 255, 0), -1)
        # center
        cv2.circle(img, (wX, hY), 4, (255, 255, 0), -1)

        # print(math.hypot(toplxy[0]-toprxy[0],  toplxy[1])-toprxy[1])

        w = math.dist(toprxy,toplxy)
        # toplxy[0] + toprxy[0]
        # 
        # h =  botrxy[0] - toplxy[0]
        # math.dist(toprxy,toplxy)
        # math.hypot(toplxy[0]-toprxy[0],  toplxy[1] -toprxy[1])
        # math.sqrt((toprxy[0] - toplxy[0])**2 + (toprxy[1] - toplxy[1])**2)
        # print("W: ", w)
        h = math.dist(toplxy,botlxy) 
        # toplxy[0] + botrxy[0]
        # 
        # math.sqrt((botrxy[0] - toprxy[0])**2 + (botrxy[1] - toprxy[1])**2)
        

        cv2.circle(img,(int(w),int(h)),5,(255, 45, 250),3)

        fov = ((np.pi/2)/(w/h))

        
     
        for x, y in points:
           
            screenX = (x * w) + toplxy[0] 
            # #+ rect[0] #- differencesqrx
            screenY = (y * h) + toplxy[1] 
            # #+ rect[1] #- differencesqry

            cv2.circle(img,(int(screenX),int(screenY)),5,(255, 0, 250),-1)

            canvasXpoint = (x * w)
            canvasYpoint = (y * h)

            dx =  canvasXpoint - wX 

            dy =  canvasYpoint - hY

            pan = (np.arctan((100/dx))) * 180/np.pi +90
            tilt = (np.arctan(100/dy)) * 180/np.pi +90

            lx = (2 * wX / w - 1) * np.tan(fov / 2)
            ly = (-2 * hY / h + 1) * np.tan(fov / 2) 
            lz = 100
            tx = np.cos(pan) * np.cos(tilt) * lx - np.cos(tilt) * np.sin(pan) * ly - np.sin(tilt) * lz
            ty = np.sin(pan) * lx + np.cos(pan) * ly
            tz = np.cos(pan) * np.sin(tilt) * lx - np.sin(pan) * np.sin(tilt) * ly + np.cos(tilt) * lz
            tilt = abs(np.arctan2(tz, tx) )*180 /np.pi
            pan  = np.arcsin(ty / np.sqrt(tx**2 + ty**2 + tz**2))*180 /np.pi
            pana.append(pan)
            tilta.append(tilt)
            
        break
  
    cv2.imshow('img',img)
    if cv2.waitKey(1)==ord('q'):
        break

print(pana)
print(tilta)
cap.release()
cv2.destroyAllWindows()