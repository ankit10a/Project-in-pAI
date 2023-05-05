import cv2
import pickle
import keras
import tensorflow as tf

cap = cv2.VideoCapture(0)

# Load the model with the specified options
def load_model():
    try:
        load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        model = tf.keras.models.load_model('maskModel.keras', options=load_options)
        print('model',model)
        return model
    except Exception as e:
        print("error->",e)

loaded_model = load_model()

def detect_mask(img):
    y_predict = loaded_model.predict(img.reshape(1,224,224,3))
    return y_predict[0][0]

def draw_label(img,text,post,bg_color):
    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.8,cv2.FILLED)
    end_x = post[0] + text_size[0][0] +2
    end_y = post[1] + text_size[0][1] -2
    cv2.rectangle(img,post,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,post,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

while True:

    ret, frame = cap.read()

    #resize the images
    img_resize = cv2.resize(frame, (224, 224))
    # detection method Called
    y_pred = (detect_mask(img_resize))

    print("result==>",y_pred,round(y_pred * 1000000))
    result = round(y_pred * 1000000)
    draw_label(frame,"Face mask Detection",(20,20),(255,0,0))
    if result <= 99:
        print("mask detected")
        draw_label(frame, "Mask detection", (30, 70), (0, 255, 0))
    else:
        print("no mask detected")
        draw_label(frame, "NO Mask", (30, 70), (0, 0, 255))

    # image show

    cv2.imshow('window', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()


# har = cv2.CascadeClassifier()