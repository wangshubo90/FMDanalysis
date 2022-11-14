import cv2
import os

def video_to_image(vidcap, xl, xr, yup, ydown, fps = None, freq=100, dstdir="output"):
    
    if fps is None:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
    dt = 1 / fps
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {n_frames}")
    print(f"del_t = {dt}")
    print(f"FPS: {fps}")
    success, image = vidcap.read()
    count=0
    while count < n_frames:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        if not image is None and count%freq==0:
            image = image[yup:ydown, xl:xr]
            os.makedirs(os.path.join(dstdir, "orgCropped"), mode=0o777, exist_ok=True)
            curtime = count*dt
            imagefname = "{:d}m{:d}s.png".format(int(curtime//60), int(curtime%60))
            print(os.path.join(dstdir, "orgCropped",imagefname))
            cv2.imwrite( os.path.join(dstdir, "orgCropped" , imagefname), image)     # save frame as JPEG file
            count = count + 1
        else:
            count = count + 1

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    global box
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position: '+str(x)+', '+str(y))
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        box.append((x, y))
        cv2.drawMarker(img, (int(x),int(y)), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=1)
        # displaying the coordinates
        # on the image window
        cv2.imshow('image', img)
    elif event == cv2.EVENT_LBUTTONUP:
        print('End Mouse Position: '+str(x)+', '+str(y))
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        box.append((x, y))
        cv2.drawMarker(img, (int(x),int(y)), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=1)
        cv2.imshow('image', img)        
        
        x , y = zip(*box)
        xl = min(x)
        xr = max(x)
        yup = min(y)
        ydown = max(y)
        video_to_image(vidcap, xl, xr, yup, ydown)

def main(fvideo):
    global img, vidcap
    vidcap = cv2.VideoCapture(fvideo)
    success, img = vidcap.read()
    cv2.imshow('image', img)
    # setting mouse handler for the image
    # and calling the click_event() function

    cv2.setMouseCallback('image', click_event)
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    # close the window
    cv2.destroyAllWindows()

# driver function
if __name__=="__main__":
    box = []
    main(r"data/2.avi")