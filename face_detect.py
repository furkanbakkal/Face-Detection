import cv2 
from imutils.video import VideoStream
import imutils
import time

cascade = "haarcascade_frontalface_default.xml" #kullanacağımız data dosyası
detector = cv2.CascadeClassifier(cascade)

vs = VideoStream(usePiCamera=True).start() #Pi Cam başlatma
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500) #500x500 lük bir izleme ekranı
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
#görüntüyü net işleyebilmek için şablona gri filtre uyguladık
    
    faces = detector.detectMultiScale(gray,1.3,3)
#.detectMultiSclae(şablon,ölçek,hassasiyet)
    for x,y,w,h in faces :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(20,200,100),2)
#cv2.rectangle(şablon, yüz koordinatları , çerçeve koordinatları,BGR cinsinden rengi,çerçeve kalınlığı)

    cv2.imshow("Yüz Tespit Sistemi",frame) 
#cv2.imshow(pencere ismi,şablon)

#eğer şablonları gray seçerseniz önizlemeyi siyah-beyaz görebilirsiniz.

    p= cv2.waitKey(1)

    if p== ord("q"): #q ya basılırsa pencereyi kapat kodu durdur.
        break


cv2.destroyAllWindows()
vs.stop()
