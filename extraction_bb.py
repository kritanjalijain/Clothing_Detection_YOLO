import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(device)

#YOLO PARAMS
yolo_modanet_params = {   "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
"weights_path" : "yolo/weights/yolov3-modanet_last.weights",
"class_path":"yolo/modanetcfg/modanet.names",
"conf_thres" : 0.5,
"nms_thres" :0.4,
"img_size" : 416,
"device" : device}


#DATASET
dataset = 'modanet' 
yolo_params = yolo_modanet_params


#Classes
def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")
    return names

classes = load_classes(yolo_params["class_path"])

#Colors
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])
#np.random.shuffle(colors)

model = 'yolo' 
detectron = YOLOv3Predictor(params=yolo_params)




folder = "tests"
images=[]
detections = []

    #path = input('img path: ')
    #if not os.path.exists(path):
    #    print('Img does not exists..')
    #    break#continue
for filename in os.listdir(folder):
    path = os.path.join(folder,filename)
    #print(path)
    img = cv2.imread(path)
    if img is not None:
        images.append(img)
        #print('image appended')
    detections = detectron.get_detections(img)
    #print(detections)
    #print(type(detections))
    #print(type(images))
    count = 1

    if len(detections) != 0 :
        detections.sort(reverse=False ,key = lambda x:x[4])
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))            
                color = colors[int(cls_pred)]
                
                color = tuple(c*255 for c in color)
                color = (.7*color[2],.7*color[1],.7*color[0])       
                    
                font = cv2.FONT_HERSHEY_SIMPLEX   
            
            
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)
                
                
                
                #print(img)
                #print(y1, y2, x1, x2)
                img_crop = img[y1:y2, x1:x2]
                #print(img_crop)
                img_id = path.split('/')[-1].split('.')[0]
                print(img_id)
                parent_dir = 'output/cropped'
                
                if classes[int(cls_pred)] in ['boots' , 'footwear']:
                    
                    directory = 'footwear'#str(classes[int(cls_pred)])
                    shoe_dir_path = os.path.join(parent_dir, directory)
                    
                    try: 
                        os.mkdir(shoe_dir_path) 
                    except OSError as error: 
                        print(error)
                    crop_path = shoe_dir_path + "/" + str(img_id) + str(classes[int(cls_pred)])+ str(count)+'.png'
                    count = count+1 

                elif classes[int(cls_pred)] in ['pants', 'shorts','skirt' ]:
                    
                    directory = 'bottomwear'#str(classes[int(cls_pred)])
                    bottom_dir_path = os.path.join(parent_dir, directory)
                    try: 
                        os.mkdir(bottom_dir_path) 
                    except OSError as error: 
                        print(error)
                    crop_path = bottom_dir_path + "/" + str(img_id) + str(classes[int(cls_pred)]) + '.png' 
                     

                elif classes[int(cls_pred)] in ['top', 'outer']:
                     
                    directory = 'topwear'#str(classes[int(cls_pred)])
                    top_dir_path = os.path.join(parent_dir, directory)
                    try: 
                        os.mkdir(top_dir_path) 
                    except OSError as error: 
                        print(error)
                    crop_path = top_dir_path + "/" + str(img_id) + str(classes[int(cls_pred)])+ '.png' 
                     
                
                else:
                    
                    directory = str(classes[int(cls_pred)])
                    new_dir_path = os.path.join(parent_dir, directory) 
                    try: 
                        os.mkdir(new_dir_path) 
                    except OSError as error: 
                        print(error)
                    crop_path = new_dir_path + "/" + str(img_id) + '.png' 
                if((x1 > 0) & (x2 > 0) & (y1 > 0) & (y2 > 0)):
                    cv2.imwrite(crop_path,img_crop)
                cv2.rectangle(img.copy(),(x1,y1) , (x2,y2) , color,3)
                y1 = 0 if y1<0 else y1
                y1_rect = y1-25
                y1_text = y1-5

                if y1_rect<0:
                    y1_rect = y1+27
                    y1_text = y1+20
                    #break
                

        print('Output saved')        
        print('End inner loop')
        #break
    #print("end of if loop")
print("End of while loop")

    
    
