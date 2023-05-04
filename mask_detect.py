import cv2
import pickle
import keras
import tensorflow as tf


cap = cv2.VideoCapture(0)




# Load the model with the specified options
# model = tf.keras.models.load_model(model_path, options=load_options)
def load_model():
    try:
        # mask_pkl = open('mask.pkl','rb')
        # print('mask_pkl',mask_pkl)
        # loaded_model = pickle.load(mask_pkl)
        # print('loaded_model----->', loaded_model)
        # Set the experimental_io_device option
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



# load_model = keras.models('maskModel.keras')


# def detect_mask(img):
#
#     y_predi = loaded_model.predict(img.reshape(1,224,224,3))
#
#     return y_predi[0][0]

while True:

    ret, frame = cap.read()

    # detection method Called
    img_resize = cv2.resize(frame, (224, 224))
    y_pred = detect_mask(img_resize)

    print("result==>",y_pred)

    # if y_pred == 0:
    #     draw_label(frame,"Mask detection", (30,30),(0,255,0))
    # else:
    #     draw_label(frame,"NO Mask",(30,30),(0,0,255))

    # print(y_pred)

    # image show

    cv2.imshow('window', frame)

    if cv2.waitKey(1) &cv2.waitKey(0) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()