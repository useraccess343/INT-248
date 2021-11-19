from Detector import*

#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz"
videoPath="video.mp4"
threshold = 0.5



classFile = "coco.names"
detector = Detector()
detector.readClasses(classfile) 
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictvideo(videoPath, threshold)