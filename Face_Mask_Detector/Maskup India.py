#!/usr/bin/env python
# coding: utf-8

# In[1]:

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from imutils.video import VideoStream
import imutils

from tkinter import filedialog
from tkinter import Button
import tkinter as tk
from cv2 import destroyAllWindows
from inspect import EndOfBlock
from tkinter import *
from tkinter import ttk
from turtle import clear
from PIL import Image,ImageTk


# In[2]:


prototxtPath=os.path.sep.join([r'C:\Users\Rajdeep Sharma\Desktop\Face Mask Detector\face_detector','deploy.prototxt.txt'])
weightsPath=os.path.sep.join([r'C:\Users\Rajdeep Sharma\Desktop\Face Mask Detector\face_detector','res10_300x300_ssd_iter_140000.caffemodel'])


# In[3]:


faceNet=cv2.dnn.readNet(prototxtPath, weightsPath)


# In[4]:


maskNet=load_model(r'C:\Users\Rajdeep Sharma\Downloads\zenas multiple\face mask\mask_detector.model')


# In[5]:

def detect_and_predict_mask(frame,faceNet,maskNet):
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104,177,123))
    faceNet.setInput(blob)
    detections=faceNet.forward()
    
    faces=[]
    locs=[]
    preds=[]
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2] 
        if confidence>0.5:

            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
        
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX),min(h-1,endY))
        

            face=frame[startY:endY,startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
        
            faces.append(face)
            locs.append((startX,startY,endX,endY))
        
        if len(faces)>0:
            faces=np.array(faces,dtype='float32')
            preds=maskNet.predict(faces,batch_size=12)
        
        return (locs,preds)
        


# In[ ]:





# In[ ]:





# In[6]:

def detection_part() :
    top = Toplevel()
    top.title("Video Detection")
    top.geometry("750x400+300+100")

    f1=LabelFrame(top,bg="red")
    f1.pack()
    L1=Label(f1,bg="red")
    L1.pack()
    btn=Button(top,text="Close Window",command=top.destroy).pack()
    vs=VideoStream(src=0).start()

    while True:
        frame=vs.read()
        frame=imutils.resize(frame,width=400)
        (locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)

        for (box,pred) in zip(locs,preds):
            (startX,startY,endX,endY)=box
            (mask,withoutMask)=pred
            label='Mask' if mask>withoutMask else 'No Mask'
            color=(0,255,0) if label=='Mask' else (0,0,255)
            cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.90,color,2)
            cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)

        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame=ImageTk.PhotoImage(Image.fromarray(frame))
        L1['image']=frame
        key=cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        top.update()

    cv2.destroyAllWindows()
    vs.stop()


# In[ ]:





# In[ ]:





# In[7]:


def image_part() :

    filepath = filedialog.askopenfilename()
    img_array = cv2.imread(filepath)
    (locs,preds)=detect_and_predict_mask(img_array,faceNet,maskNet)

    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY)=box
        (mask,withoutMask)=pred
        label='Mask' if mask>withoutMask else 'No Mask'
        color=(0,255,0) if label=='Mask' else (0,0,255)
        cv2.putText(img_array,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.90,color,2)
        cv2.rectangle(img_array,(startX,startY),(endX,endY),color,2)
    cv2.imshow('Image Detection',img_array)
    
cv2.destroyAllWindows()

    


# In[ ]:





# In[ ]:





# In[24]:


import tkinter as tk
from inspect import EndOfBlock
from tkinter import *
from tkinter import ttk
from turtle import clear
from PIL import Image,ImageTk
#from cv2 import destroyAllWindows


def show_frame(frame):
    frame.tkraise()
    
# from GradientFrame import GradientFrame


# root.mainloop()
class Example(tk.Frame):
    def __init__(frame1, parent,col1,col2):
        tk.Frame.__init__(frame1, parent)
        f1 = GradientFrame(frame1,col1, col2,borderwidth=1, relief="sunken")
#         f2 = GradientFrame(parent, col1, col2, borderwidth=1, relief="sunken")
        f1.pack(side="top", fill="both", expand=True)
#         f2.pack(side="bottom", fill="both", expand=True)

class GradientFrame(tk.Canvas):
    '''A gradient frame which uses a canvas to draw the background'''
    def __init__(frame1, parent, color1, color2, **kwargs):
        tk.Canvas.__init__(frame1, parent, **kwargs)
        frame1._color1 = color1
        frame1._color2 = color2
        frame1.bind("<Configure>", frame1._draw_gradient)

    def _draw_gradient(frame1, event=None):
        '''Draw the gradient'''
        frame1.delete("gradient")
        width = frame1.winfo_width()
        height = frame1.winfo_height()
        limit = width
        (r1,g1,b1) = frame1.winfo_rgb(frame1._color1)
        (r2,g2,b2) = frame1.winfo_rgb(frame1._color2)
        r_ratio = float(r2-r1) / limit
        g_ratio = float(g2-g1) / limit
        b_ratio = float(b2-b1) / limit
        
        
        
        for i in range(limit):
            nr = int(r1 + (r_ratio * i))
            ng = int(g1 + (g_ratio * i))
            nb = int(b1 + (b_ratio * i))
            color = "#%4.4x%4.4x%4.4x" % (nr,ng,nb)
            frame1.create_line(i,0,i,1000, tags=("gradient",), fill=color)
        frame1.lower("gradient")
    
window = tk.Tk()
window.state('zoomed')
window.title("Maskup India")
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)

frame1 = tk.Frame(window)
frame2 = tk.Frame(window)
frame3 = tk.Frame(window)
frame4 = tk.Frame(window)
frame5 = tk.Frame(window)
frame6 = tk.Frame(window)
frame7 = tk.Frame(window)
frame8 = tk.Frame(window)
frame9 = tk.Frame(window)
Example(frame1,"#ff5f6d","#ffc371").pack(fill="both", expand=True)
Example(frame2,"#4ca1af","#c4e0e5").pack(fill="both", expand=True)
Example(frame3,"#eacda3","#d6ae7b").pack(fill="both", expand=True)
Example(frame4,"#56ab2f","#a8e063").pack(fill="both", expand=True)
Example(frame5,"#ed4264","#ffedbc").pack(fill="both", expand=True)
Example(frame6,"#ba5370","#f4e2d8").pack(fill="both", expand=True)
Example(frame7,"#ffd89b","#19547b").pack(fill="both", expand=True)
Example(frame8,"#ffd89b","#19547b").pack(fill="both", expand=True)
Example(frame9,"#ffd89b","#19547b").pack(fill="both", expand=True)
for frame in (frame1, frame2, frame3,frame4,frame5,frame6,frame7,frame8,frame9):
    frame.grid(row=0,column=0,sticky='nsew')


menubar = Menu(window)
menubar.add_command(label="Home ", command=lambda:show_frame(frame1))  
menubar.add_command(label="How it works  ", command = lambda:show_frame(frame2)) 
menubar.add_command(label="Detection  ", command = lambda:show_frame(frame3))
menubar.add_command(label="Guildelines  ", command = lambda:show_frame(frame4))     
menubar.add_command(label="About Us  ", command = lambda:show_frame(frame5)) 
menubar.add_command(label="Contact Us  ", command = lambda:show_frame(frame6))   
  
# display the menu  
window.config(menu=menubar)    

# 1st photo
img_1 = Image.open(r"C:\Python Programs\Tkinker programs\Face_Mask_Detector\detection_img\1.jpg")
img_1 = img_1.resize((600,300),Image.ANTIALIAS)
frame1.photoimg_1 = ImageTk.PhotoImage(img_1)
frame1.btn_1 = Button(frame1,image = frame1.photoimg_1)
frame1.btn_1.place(x= 50, y= 90,width = 600, height = 300)

# 2nd photo
img_2 = Image.open(r"C:\Python Programs\Tkinker programs\Face_Mask_Detector\detection_img\2.jpg")
img_2 = img_2.resize((600,300),Image.ANTIALIAS)
frame1.photoimg_2 = ImageTk.PhotoImage(img_2)
frame1.btn_2 = Button(frame1,image = frame1.photoimg_2)
frame1.btn_2.place(x= 700, y= 90,width = 600, height = 300)
 
# image
Covid19_img = PhotoImage(file ='C:\Python Programs\Tkinker programs\Face_Mask_Detector\detection_img\covid-19-14.png')
Omricon_img = PhotoImage(file ='C:\Python Programs\Tkinker programs\Face_Mask_Detector\detection_img\omicron-3.png')
DeltaVirus_img = PhotoImage(file ='C:\Python Programs\Tkinker programs\Face_Mask_Detector\detection_img\deltav-2.png')

# button

Covid19_button = Button(frame1,image = Covid19_img,border=0,command = lambda:show_frame(frame7))
Covid19_button.place(x = 300 ,y= 500) 
Omricon_button = Button(frame1,image = Omricon_img,border = 0,command = lambda:show_frame(frame8))
Omricon_button.place(x = 600 ,y= 500) 
DeltaVirus_button = Button(frame1,image = DeltaVirus_img,border = 0,command = lambda:show_frame(frame9))
DeltaVirus_button.place(x = 900 ,y= 500) 



lb1 = Label(frame7,text="What is Covid-19 ?",font = ("times 35",10))
lb1.place(x=25,y = 50)
lb2= Label(frame7,text="Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe.",font = ("times 35",10)) 
lb2.place(x=25,y = 150-70)
lb3= Label(frame7,text="These particles range from larger respiratory droplets to smaller aerosols.",font = ("times 35",10)) 
lb3.place(x=25,y = 170-65)

lb4 = Label(frame7,text="What are its symptoms?",font = ("times 35",10))
lb4.place(x=25,y = 210-50)
lb5= Label(frame7,text=" Most common symptoms:",font = ("times 35",10)) 
lb5.place(x=28,y = 240-50)
lb6 = Label(frame7,text=" • fever\n",font = ("times 35",10))
lb6.place(x=32,y = 270-50)
lb7 = Label(frame7,text=" • cough\n",font = ("times 35",10))
lb7.place(x=32,y = 290-50)
lb8 = Label(frame7,text=" • tiredness\n",font = ("times 35",10))
lb8.place(x=32,y = 310-50)
lb9 = Label(frame7,text=" • loss of taste or smell.",font = ("times 35",10))
lb9.place(x=32,y = 330-50)
lb10= Label(frame7,text=" Less common symptoms:",font = ("times 35",10)) 
lb10.place(x=28,y = 360-50)
lb11 = Label(frame7,text=" • headache\n",font = ("times 35",10))
lb11.place(x=32,y = 390-50)
lb12 = Label(frame7,text=" • diarrhoea\n",font = ("times 35",10))
lb12.place(x=32,y = 410-50)
lb13 = Label(frame7,text=" • sore throat\n ",font = ("times 35",10))
lb13.place(x=32,y = 430-50)
lb14 = Label(frame7,text=" • aches and pains\n",font = ("times 35",10))
lb14.place(x=32,y = 450-50)
lb15 = Label(frame7,text=" • red or irritated eyes.\n ",font = ("times 35",10))
lb15.place(x=32,y = 470-50)
lb16 = Label(frame7,text=" • a rash on skin, or discolouration of fingers or toes",font = ("times 35",10))
lb16.place(x=32,y = 490-50)

#Omricon button
frame8_title=  tk.Label(frame8, text='\n\n\n\n\n\n\n\n\n\nOmricon Page', font='times 35')


lb1 = Label(frame8,text="What is Omicron ?",font = ("times 35",10))
lb1.place(x=25,y = 50)
lb2= Label(frame8,text="The Omicron variant (B.1.1.529) is a variant of SARS-CoV-2 (the virus that causes COVID-19) that was first reported to the World Health Organization (WHO) from South Africa on 24 November 2021.",font = ("times 35",10)) 
lb2.place(x=25,y = 150-70)
lb3= Label(frame8,text="Omicron might be less able to penetrate deep lung tissue. Omicron infections are 91 percent less fatal than the delta variant, with 51 percent less risk of hospitalization.",font = ("times 35",10)) 
lb3.place(x=25,y = 170-65)

lb4 = Label(frame8,text="What are its symptoms?",font = ("times 35",10))
lb4.place(x=25,y = 210-50)

lb6 = Label(frame8,text=" • cough\n",font = ("times 35",10))
lb6.place(x=32,y = 240-50)
lb7 = Label(frame8,text=" • diarrhea\n",font = ("times 35",10))
lb7.place(x=32,y = 260-50)
lb8 = Label(frame8,text=" • headache\n",font = ("times 35",10))
lb8.place(x=32,y = 280-50)
lb9 = Label(frame8,text=" • skin rashes",font = ("times 35",10))
lb9.place(x=32,y = 300-50)
lb10 = Label(frame8,text=" • low-grade fever\n",font = ("times 35",10))
lb10.place(x=32,y = 320-50)
lb11 = Label(frame8,text=" • pain and itchiness in the throat\n",font = ("times 35",10))
lb11.place(x=32,y = 340-50)
lb12 = Label(frame8,text=" • general weakness and tiredness\n ",font = ("times 35",10))
lb12.place(x=32,y = 360-50)
lb13 = Label(frame8,text=" • loss of taste and smell (less common)\n",font = ("times 35",10))
lb13.place(x=32,y = 380-50)
lb14 = Label(frame8,text=" • severe body pain which interferes in any work ",font = ("times 35",10))
lb14.place(x=32,y = 400-50)

#Delta virus button
frame9_title=  tk.Label(frame9, text='\n\n\n\n\n\n\n\n\n\nDelta Virus Page', font='times 35')


lb1 = Label(frame9,text="What is Delta Virus ?",font = ("times 35",10))
lb1.place(x=25,y = 50)
lb2= Label(frame9,text="The Delta variant (B.1.617.2) is a variant of SARS-CoV-2, the virus that causes COVID-19. It was first detected in India in late 2020.Delta variant was named on 31 May 2021, had spread to over 179 countries by 22 November 2021.",font = ("times 35",10)) 
lb2.place(x=25,y = 150-70)
lb3= Label(frame9,text="The Delta variant of COVID-19 has been called a variant of concern by WHO because of its increased transmissibility and increased ability to cause a severe form of the disease.",font = ("times 35",10)) 
lb3.place(x=25,y = 170-65)

lb4 = Label(frame9,text="What are its symptoms?",font = ("times 35",10))
lb4.place(x=25,y = 210-50)

lb5 = Label(frame9,text=" • fever\n",font = ("times 35",10))
lb5.place(x=32,y = 240-50)
lb6 = Label(frame9,text=" • cough\n",font = ("times 35",10))
lb6.place(x=32,y = 260-50)
lb7 = Label(frame9,text=" • nausea\n",font = ("times 35",10))
lb7.place(x=32,y = 280-50)
lb8 = Label(frame9,text=" • vommiting",font = ("times 35",10))
lb8.place(x=32,y = 300-50)
lb9 = Label(frame9,text=" • headache\n",font = ("times 35",10))
lb9.place(x=32,y = 320-50)
lb10 = Label(frame9,text=" • sore throat\n",font = ("times 35",10))
lb10.place(x=32,y = 340-50)
lb11 = Label(frame9,text=" • runny nose\n ",font = ("times 35",10))
lb11.place(x=32,y = 360-50)
lb12 = Label(frame9,text=" • loss of taste and smell (less common)",font = ("times 35",10))
lb12.place(x=32,y = 380-50)




#==================How it works code
frame2_title=  tk.Label(frame2, text='\n\n\n\n\n\n\n\n\n\n  How it works Page', font='times 35')



#3rd photo
img_3 = Image.open(r"C:\Python Programs\Tkinker programs\Face_Mask_Detector\detection_img\4.jpg")
img_3 = img_3.resize((750,430),Image.ANTIALIAS)
frame2.photoimg_3 = ImageTk.PhotoImage(img_3)
frame2.btn_3 = Button(frame2,image = frame2.photoimg_3)
frame2.btn_3.place(x= 320, y= 120,width = 750, height = 430)



#==================Detection code
frame3_title=  tk.Label(frame3, text='\n\n\n\n\n\n\n\n\n\nStart Detection Page',font='times 35')

# image
ImageDetect_img = PhotoImage(file ='C:\Python Programs\Tkinker programs\Face_Mask_Detector\detection_img\Image Detection2.png')
VideoDetect_img = PhotoImage(file ='C:\Python Programs\Tkinker programs\Face_Mask_Detector\detection_img\Video Detection4.png')


# button

ImageDetection_button = Button(frame3,image = ImageDetect_img,borderwidth=0,command = image_part)
ImageDetection_button.place(x = 350 ,y= 250) 
VideoDetection_button = Button(frame3,image = VideoDetect_img,borderwidth = 0,command= detection_part)
VideoDetection_button.place(x = 850 ,y= 250) 



#==================Guidelines code
frame4_title=  tk.Label(frame4, text='\n\n\n\n\n\n\n\n\n\nGuidelines Page',font='times 35')



lb1 = Label(frame4,text="The guidelines which can be used to protect ourself and others, is as follows:",font = ("times 35",10))
lb1.place(x=25,y = 50)
lb2= Label(frame4,text=" • Get vaccinated and stay up to date on your COVID-19 vaccines.",font = ("times 35",10)) 
lb2.place(x=25,y = 150-50)
lb3= Label(frame4,text=" • Wear a mask.",font = ("times 35",10)) 
lb3.place(x=25,y = 170-50)
lb4 = Label(frame4,text=" • Stay 6 feet away from others.",font = ("times 35",10))
lb4.place(x=25,y = 190-50)
lb5= Label(frame4,text=" • Avoid poorly ventilated spaces and crowds.",font = ("times 35",10)) 
lb5.place(x=25,y = 210-50)
lb6 = Label(frame4,text=" • Go through test to prevent spread to others.",font = ("times 35",10))
lb6.place(x=25,y = 230-50)
lb7 = Label(frame4,text=" • Wash your hands often.",font = ("times 35",10))
lb7.place(x=25,y = 250-50)
lb8 = Label(frame4,text=" • Cover coughs and sneezes.",font = ("times 35",10))
lb8.place(x=25,y = 270-50)
lb9 = Label(frame4,text=" • Clean and disinfect the touched surfaces.",font = ("times 35",10))
lb9.place(x=25,y = 290-50)
lb10= Label(frame4,text=" • Monitor your health daily.",font = ("times 35",10)) 
lb10.place(x=25,y = 310-50)
lb11 = Label(frame4,text=" • Concern a doctor , when health is not good.",font = ("times 35",10))
lb11.place(x=25,y = 330-50)
lb12 = Label(frame4,text=" • Follow recommendations for quarantine,when needed.",font = ("times 35",10))
lb12.place(x=25,y = 350-50)
lb13 = Label(frame4,text=" • Follow recommendations for isolation, when needed.",font = ("times 35",10))
lb13.place(x=25,y = 370-50)
lb14 = Label(frame4,text=" • Take precautions when you travel.",font = ("times 35",10))
lb14.place(x=25,y = 390-50)


#==================About Us code
frame5_title=  tk.Label(frame5, text='\n\n\n\n\n\n\n\n\n\nAbout Us Page',font='times 35')



lb1 = Label(frame5,text="We the student of XYZ College,Area , Branch: Computer Science (2020 batch) had immense pleasure of getting this opportunity of making “Face Mask Detector, we came up with the idea of making face mask detector, as it is not",font = ("times 35",10))
lb1.place(x=25,y = 50)
lb2= Label(frame5,text="possible to keep an eye on each and every person whether he/she is wearing a mask or not.Therefore, making a face mask detector has become a crucial task for us to help global society.",font = ("times 35",10)) 
lb2.place(x=25,y = 125-50)
lb3= Label(frame5,text="Group members in our mini project are:",font = ("times 35",10)) 
lb3.place(x=25,y = 170-50)
lb4 = Label(frame5,text=" • Zenas Rumao : Interests:- Cyber Security, Animation, Operating System, AI/ML Languages Known:- C++ , C, Java ,Python ,HTML,MySql",font = ("times 35",10))
lb4.place(x=28,y = 200-50)
lb5= Label(frame5,text=" • Rajdeep Sharma : Interest:- Cyber Security, Cryptography,Web development, AI/ML  Languages Known:- C++ , C, Java ,Python ,HTML,MySql",font = ("times 35",10)) 
lb5.place(x=28,y = 220-50)
lb6 = Label(frame5,text=" • Suvitson Harrese : Interest:- App development, Game development ,AI/ML ,Editing Image Languages Known:- C++ , C, Java ,Python ,HTML,MySql",font = ("times 35",10))
lb6.place(x=28,y = 240-50)
lb7 = Label(frame5,text=" • Christy Valliamannil : Interest:- Web development , Game Development  Languages Known:- C++ , C, Java ,Python ,MySql",font = ("times 35",10))
lb7.place(x=28,y = 260-50)
lb7 = Label(frame5,text="Our Guide :-  ",font = ("times 35",10))
lb7.place(x=28,y = 290-50)
lb7 = Label(frame5,text=" Ms Snehal Nikalje: Interest:- Cyber Security, Deep Learning   Languages Known:- C++ , C, Java , Python , HTML, JavaScipt, MySql ",font = ("times 35",10))
lb7.place(x=28,y = 320-50)




#==================Contact Us code
frame6_title=  tk.Label(frame6, text='\n\n\n\n\n\n\n\n\n\nContact Us Page',font='times 35')


lb1 = Label(frame6,text="You all can connect us on:",font = ("times 35",10))
lb1.place(x=25,y = 50)
lb2= Label(frame6,text="• Twitter ID: 123456",font = ("times 35",10)) 
lb2.place(x=25,y = 130-50)
lb3= Label(frame6,text="• Discord ID: #123456",font = ("times 35",10)) 
lb3.place(x=25,y = 160-50)
lb4 = Label(frame6,text="• Phone No: 123456789",font = ("times 35",10))
lb4.place(x=25,y = 190-50)
lb5= Label(frame6,text="• Whatsapp No: 123456789",font = ("times 35",10)) 
lb5.place(x=25,y = 220-50)
lb6= Label(frame6,text="• Email ID: 123456@gmail.com",font = ("times 35",10)) 
lb6.place(x=25,y = 250-50)

show_frame(frame1)


window.mainloop()


# In[ ]:



# %%

# %%

# %%

# %%
