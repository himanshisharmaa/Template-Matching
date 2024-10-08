import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", type=str,
                required=True,help="Path to input Image where we'll apply template matching")
ap.add_argument("-t","--template",type=str,
                required=True,help="Path to template image")
ap.add_argument("-b","--threshold",type=float,
                default=0.8, help="threshold for multi-template matching")
args=vars(ap.parse_args())

#load the input image and template image from disk,then
# grab the template image spatial dimensions
print("[INFO] loading images...")
image=cv2.imread(args["image"])
template=cv2.imread(args['template'])
#We grab the template‘s spatial dimensions 
# so we can use them to derive 
# the bounding box coordinates of matched 
# objects easily.
(tH,tW)=template.shape[:2]

# display the image and template to our screen
cv2.imshow("Image",image)
cv2.imshow("Template",template)

# convert both the image and template to grayscale
imageGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
templateGray=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)


#perform template matching
print("[INFO] performing template matching...")
result=cv2.matchTemplate(imageGray,templateGray,cv2.TM_CCOEFF_NORMED)


#find all locations in the result map where the matched
# value is greater than the threshold,then clone our original
#image so we can draw on it
(yCoords,xCoords)=np.where(result>=args["threshold"])

clone=image.copy()
print(f"[INFO] {len(yCoords)} matched locations 'before' NMS")


# loop over our starting (x,y)-coordinates
for (x,y) in zip(xCoords,yCoords):
    # draw the bounding box on the image
    cv2.rectangle(clone,(x,y),(x+tW,y+tH),(255,0,0),3)

cv2.imshow("Before NMS",clone)
cv2.waitKey(0)
cv2.imwrite("Outputs/before_nms.jpg",clone)

# initialize our list of rectangles
rects=[]

#loop over the starting (x,y)-coordinates again
for (x,y) in zip(xCoords,yCoords):
    # update the list of rectangles
    rects.append((x,y,x+tW,y+tH))

    #apply non-maximum suppression to the rectangles
pick=non_max_suppression(np.array(rects))
print(f"[INFO] {len(pick)} matched locations 'after' NMS")

#loop over the final bounding boxes 
for (startX,startY,endX,endY) in pick:
    # draw the bounding box on the image
    cv2.rectangle(image,(startX,startY),(endX,endY),
                    (255,0,0),2)

cv2.imshow("After NMS",image)
cv2.imwrite("Outputs/after_nms.jpg",image)
cv2.waitKey(0)