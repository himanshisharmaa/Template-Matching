import argparse
import cv2

# construct the argument parser
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",type=str,required=True,
help="Path to the input image")
ap.add_argument("-t","--template",type=str,required=True,
        help="path to template image")
args=vars(ap.parse_args())

# load the input image and template image from disk,
# then display
print("[INFO] loading images...")
image=cv2.imread(args['image'])
template=cv2.imread(args['template'])
cv2.imshow("Image",image)
cv2.imshow("Template",template)

# convert both the image and template to grayscale
imageGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
templateGray=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)


# perform template matching
print("[INFO] performing template matching...")
results=cv2.matchTemplate(imageGray,templateGray,
                          cv2.TM_CCOEFF_NORMED)
print(f"Results: {results}")
(minVal,maxVal,minLoc,maxLoc)=cv2.minMaxLoc(results)

# determine the strting and ending (x,y)-coordinates of 
# the bounding box
(startX,startY)=maxLoc
endX=startX+template.shape[1]
endY=startY+template.shape[0]

# draw the bounding box on the image
cv2.rectangle(image,(startX,startY),(endX,endY),(255,0,0),3)

# show the output image
cv2.imshow("Output",image)
cv2.imwrite("Outputs/single_template_matching.jpg",image)
cv2.waitKey(0)
