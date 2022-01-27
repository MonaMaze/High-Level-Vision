# Build Tkinter APP With the ability to be Complete Image Processing Control Panel with ability to 
from tkinter import *
from tkinter import ttk
from ttkbootstrap import Style
from tkinter import filedialog
from PIL import ImageTk , Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical, plot_model

def save_face():
    txt_val = str(txt.get())
    i = 1
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('img', frame)
        if cv2.waitKey(5) & 0xFF == ord('s'):
            os.makedirs('faces/'+txt_val, exist_ok=True)
            cv2.imwrite(os.path.join('faces/'+txt_val, txt_val+str(i)+'.jpg'), frame)
            if i == 400:
                break
            i += 1
    cap.release()
    cv2.destroyAllWindows()
    
def bld_model():
    #global labels
    i = 0
    labels = {}
    y = []
    X = []
    folder = 'faces'
    for subfolder in os.listdir(folder):
        for filename in os.listdir(os.path.join(folder,subfolder)):
            img = cv2.imread(os.path.join(folder,subfolder,filename))
            img=cv2.resize(img,(224,224))
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255
            if img is not None:
                X.append(img)
        y += [i]*400
        labels[i]=subfolder
        i += 1
    y_ = np.array(y).T
            
    X,y_ = shuffle(X,y_)
    X_train, X_test, y_train, y_test = train_test_split(X, y_, train_size = 0.7, random_state = 101)
    Y_train = to_categorical(y_train, i)
    Y_test = to_categorical(y_test, i)
    X_train=np.asarray(X_train)
    X_test=np.asarray(X_test)
    Y_train=np.asarray(Y_train)
    Y_test=np.asarray(Y_test)
    
    # Build CNN
    # First layer
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224,224,3)))
    #model.add(BatchNormalization())
    convLayer01 = Activation('relu')
    model.add(convLayer01)

    # Convolution Layer 2
    model.add(Conv2D(32, (3, 3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    convLayer02 = MaxPooling2D(pool_size=(2,2))
    model.add(convLayer02)

    # Convolution Layer 3
    model.add(Conv2D(64,(3, 3)))
    #model.add(BatchNormalization(axis=-1))
    convLayer03 = Activation('relu')
    model.add(convLayer03)

    # Convolution Layer 4
    model.add(Conv2D(64, (3, 3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    convLayer04 = MaxPooling2D(pool_size=(2,2))
    model.add(convLayer04)
    model.add(Flatten())

    # Fully Connected Layer 5
    model.add(Dense(512))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 6                       
    model.add(Dropout(0.2))
    model.add(Dense(i))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
    test_gen = ImageDataGenerator()
    train_generator = gen.flow(X_train, Y_train, batch_size=40)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=40)
    history = model.fit_generator(train_generator, steps_per_epoch=280*i//40, epochs=5, verbose=1, 
                    validation_data=test_generator, validation_steps=120*i//40)
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    model.save("model.h5")
    print("Saved model to disk")

def int_model():
    global model
    model = load_model('model.h5')
    print("Model loaded")
    
def rec_face():
    global model
    i = 0
    labels = {}
    folder = 'faces'
    for subfolder in os.listdir(folder):
        labels[i]=subfolder
        i += 1
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        img=cv2.resize(frame,(224,224))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255
        
        img=img.reshape(1,224,224,3)
        pred_f = model.predict(img)
        pred_n = np.argmax(np.round(pred_f), axis=1)
        
        txt = 'NO ONE'
        for x in range(len(labels)):
            if pred_n == x:
                txt = str(labels[x])

        cv2.putText(frame, txt, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        cv2.imshow('img', frame)
        
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# initialize the window toolkit along with the two image panels
style = Style(theme='minty')
root = style.master
root.title("Face Recognition")
root.geometry("+1+50")
root.wait_visibility()
#root.after_idle(load_image)

frame = Frame(root, relief=RAISED, borderwidth=0)
frame.pack(fill=BOTH, expand=True)

# Text ox for writing the name
txt = ttk.Entry(frame)
txt.insert(INSERT, "Write your name")
txt.pack(side='top', fill='both', padx=10, pady=10)

# Button to Upload faces
btn1 = ttk.Button(frame, text = '     Save faces     ', style='primary.TButton', command=save_face)
btn1.pack(side='left', fill='both', padx=30, pady=10)

# Button to Build the model
btn3 = ttk.Button(frame, text = '  Build the model  ', style='danger.TButton', command=bld_model)
btn3.pack(side='left', fill='both', padx=30, pady=10)

# Button to Initialize the model
btn4 = ttk.Button(frame, text = 'Initialize the model', style='success.TButton', command=int_model)
btn4.pack(side='left', fill='both', padx=30, pady=10)

# Button for face recognition 
btn2 = ttk.Button(frame, text = 'Recognize faces', style='secondary.TButton', command=rec_face)
btn2.pack(side='right', fill='both', padx=30, pady=10)

# App Utilization Information Label 
bot = ttk.LabelFrame(root, text='Information')
bot.pack(side='bottom', padx=10, pady=10)
txt1 = "- After clicking Save faces, press 's' to save the scene. The model needs 400 photos, then the Camera will be closed."
txt2 = "- After clicking Recognize faces, press 'q' if you finished and want to close the camera."
txt3 = "- Ater adding new face/s, you have to choose 'Build the model' to add the new face/s to the model. This step takes long time."
txt4 = "- Before 'Recognize faces', you need to choose 'Initialize the model' first once"
lbl4 = ttk.Label(bot, text = txt4)
lbl4.pack(side='bottom', anchor='w')
lbl3 = ttk.Label(bot, text = txt3)
lbl3.pack(side='bottom', anchor='w')
lbl2 = ttk.Label(bot, text = txt2)
lbl2.pack(side='bottom', anchor='w')
lbl1 = ttk.Label(bot, text = txt1)
lbl1.pack(side='bottom', anchor='w')

# kick off the GUI
root.mainloop()
