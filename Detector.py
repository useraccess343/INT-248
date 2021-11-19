import cv2 , time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.sedd(123)

class Detector:
    def __init__(self):
        pass
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

            #color list
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))

            print(len(self.classesList), len(self.colorList))

            def downloadModel(self, modelURL):
                fileName= os.path.basename(modelURL)
                self.modelName=fileName[:fileName.index('.')]
                
                print(fileName)
                print(self.modelName)

                self.cacheDir = "./pretrained_models"
                os.makedirs(self.cacheDir, exist_ok=True)

                get_file(fname=fileName,
                origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)
                

            def loadModel(self):
                print("Loading Model" + self.modelName)
                tf.keras.backend.clear_session()
                self.model= tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.ModelName, "saved_model"))

                print("Model" + self.modelName + "loaded Successfully")

                def createBoundingBox(self, image, threshold= 0.5):
                    inputTensor = cv2.cvtcolor(image.copy(), cv2.Color_BGR2RGB)
                    inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
                    inputTensor = inputTensor[tf.newaxis,...]
                    detections = self.model(inputTensor)
                    bboxs = detections['detection_boxes'][0].numpy()
                    classIndexes = detections['detections_classes'][0].numpy().astype(np.int32)
                    classScores = detections['detections_classes'][0].numpy()
                    

                    imH , imW, imC = image.shape

                    bboxIdx= tf.image.non_max_supression(bbox, classScores, max_output_size=50,
                    iou_threshold=threshold, score_threshold=threshold)
                    print(bboxIdx)



                    if len(bboxIdx) !=0:
                        for i in bboxIdx:
                            bbox = tuple(bbox[i].tolist())
                            classConfidence = round(100*classScores[i])
                            classIndex = classIndexes[i]


                            classLabelText = self.classesList[classIndex],upper()
                            classColor = self.colorList[classIndex]
                            displayText = '{}: {}%'.format(classLabelText, classConfidence)

                            ymin , xmin , ymax , xmax = bbox
                            
                             ymin , xmin , ymax , xmax  = (xmin * imW, xmax * imw, ymin * imH, ymax * imH)

                              ymin , xmin , ymax , xmax =  int(ymin) , int(xmin) , int(ymax) , int(xmax)

                              cv2.rectangel(image, (xmin, ymin), (xmax,ymax), color= classColor, thickness=1)

                              lineWidth = min(int((xmax - xmin)*0.2), int((ymax - ymin)*0.2))

                              cv2.line(image,(xmin,ymin), (xmin  + lineWidth,ymin), classColor , thickness= 5)
                              cv2.line(image,(xmin,ymin), (xmin ,ymin + lineWidth), classColor , thickness= 5)

                              cv2.line(image,(xmax,ymin), (xmax  - lineWidth,ymin), classColor , thickness= 5)
                              cv2.line(image,(xmax,ymin), (xmax ,ymin + lineWidth), classColor , thickness= 5)

                              cv2.line(image,(xmin,ymax), (xmin  + lineWidth,ymax), classColor , thickness= 5)
                              cv2.line(image,(xmin,ymax), (xmin ,ymax - lineWidth), classColor , thickness= 5)
                              
                              cv2.line(image,(xmin,ymax), (xmax  - lineWidth,ymax), classColor , thickness= 5)
                              cv2.line(image,(xmin,ymax), (xmin ,ymax - lineWidth), classColor , thickness= 5)
                              return image 






                def predictVideo(self, videoPath, threshold= 0.5):
                    cap = cv2.VideoCapture(videoPath)

                    if(cap.isOpened() == False):
                        print("Error opening File ....")
                        return 
                        (success, image)= cap.read()

                        startTime= 0

                        while success:
                            currentTime = time.time()
                            fps= 1/(currentTime - startTime)
                            startTime= currentTime
                            
                            bboxImage = self.createBoundingBox(image, threshold)

                            cv2.puttext(bboxImage, "FPS" + str(int(fps)), (20.70), cv2.FONT_HERSHY_PLAIN, 2, (0,255,0),2)
                            cv2.imshow("Result", bboxImage)

                            key = cv2.waitkey(1) & 0xFF
                            if key == ord("t"):
                                break
                            (success, image) = cap.read()
                            cv2.destroyAllWindows()







