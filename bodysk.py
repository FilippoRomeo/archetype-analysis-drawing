import jetson.inference
import jetson.utils
import random

import numpy as np
import cv2

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.",
    formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() + jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

img_size = (1920,1080)
skl = np.ones(img_size) 

class npoints: 

    def __init__(self, id, x,y):
        self.id = id
        self.x = x
        self.y = y
normalizedPoints = []
normalizepxb = []

# if no argument passed do live stream
# ---> if not len(sys.argv) >1:
# process frames until the user exits

while True:

    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=opt.overlay)
    #net.Overlay(poses,skl)
    
    jetson.inference.poseNet

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        x = 0
        y = 0
        x2 = 0
        y2 = 0
        
        # print(pose)
        # print(pose.Keypoints)
        # print('Links', pose.Links)
   
        dictLinks = pose.Links
        dictPoint = pose.Keypoints
   
        print("(((((((((((((())))))))))))))")
        # print(dictLinks)
        # print(dictPoint)
        idlistp = []

        for obj in pose.Links:
            
            a=obj[0]
            b=obj[1]
            if a not in idlistp:
                idlistp.append(a)
            if b not in idlistp:
                idlistp.append(b)    
        # print(idlistp)

        for obj in pose.Keypoints:

            id =getattr(obj,'ID')
            x = getattr(obj,'x')
            y = getattr(obj,'y')
            
            if id not in  idlistp:
                cv2.circle(skl, (round(x),round(y)),radius=1,color=(0,0,0), thickness= 5)
            
            row_i = random.randint(0, img.shape[0]) 
            col_j = random.randint(0, img.shape[1]) 
        
            num_rows, num_cols = img.shape[:2]
            x = x/(num_cols - 1.)
            y = y/(num_rows - 1.)

            a = npoints(id,x,y)
            if a not in normalizedPoints:   
                normalizedPoints.append(a)
        if normalizedPoints not in normalizepxb:
            normalizepxb.append(normalizedPoints)

        for tup in pose.Links:
            
            # print(tup)
            # print(tup[0])
            # print(tup[1])
            a = dictPoint[tup[0]]
            ase = dictPoint[tup[1]]
            x = getattr(a,"x")
            y = getattr(a,"y")
            x2 = getattr(ase,"x")
            y2 = getattr(ase,"y")
            if getattr(a,"ID") in range(5):
                cv2.circle(skl, (round(x),round(y)),radius=1,color=(0,0,0), thickness= 5)
            else:
                cv2.line(skl, (round(x),  round(y)),(round(x2), round(y2)), (0,0,0),10)

            # cv2.circle(skl, (round(x),round(y)),radius=1,color=(0,0,0), thickness= 5)
            # cv2.circle(skl, (round(x2),round(y2)),radius=1,color=(0,0,0), thickness= 5)
       
        
    cv2.imshow("foo",skl)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    xyarr=[]
    peparr=[]
    for e in normalizepxb:

        for s in e:
            x = getattr(s,"x")
            y = getattr(s,"y")
            a = [x,y]
            xyarr.append(a)
            # print("_______________")
            # print(getattr(s,"id"))
            # print(x)
            # print(y)
        peparr.append(xyarr)
    print(peparr)
    print("_________________________________________________")
    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
