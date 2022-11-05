import logging
import argparse
from database import conn
import os
import streamlit as st
import requests
from io import StringIO 
from scipy.io import wavfile
import sys
import pathlib
import matplotlib.pyplot as plt
import test_both_ml_and_db
import Database_works as dbw
from connect_db_ml import connect_ml_and_db
noise_path="C:\\Users\\Tigist\\Downloads\\UrbanSound8K.tar\\UrbanSound8K\\UrbanSound8K\\audio\\"
from helpers import acc_per_class
#from helpers import compute_confusion_matrix
from helpers import last_fun
#from helpers import plot_spec
#from helpers import plot_wave
from Cancel import cancel
from PIL import Image
from Cancel import run
from Cancel import save_wav
from Cancel import lp_filter
from pydub import AudioSegment
import scipy.io.wavfile as wav


import base64








st.markdown(
    """
<style>


.reportview-container .markdown-text-container {
    color: #ebebeb  
    font-family: monospace;
}
.sidebar .sidebar-content {
    color: red
    background-image: linear-gradient(#ff0099,#ff0099);
    
.Widget>label {
    color:#ebebeb;
    font-family: monospace;
}
[class^="st-b"]  {
    color: black;
    font-family: monospace;
}
.st-bb {
    background-color: transparent;
    color: red
}
.st-at {
    background-color: #b452ff;
    color: red
}
footer {
    font-family: monospace;
    color: red
}
.reportview-container .main footer, .reportview-container .main footer a {
    color: red
}
header .decoration {
    background-image: none;
    color: red
}

</style>
""",
    unsafe_allow_html=True,
)



st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)










# # set background, use base64 to read local file
# def get_base64_of_bin_file(bin_file):
#     """
#     function to read png file 
#     ----------
#     bin_file: png -> the background image in local folder
#     """
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_png_as_page_bg(png_file):
#     """
#     function to display png as bg
#     ----------
#     png_file: png -> the background image in local folder
#     """
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = '''
#     <style>
#     st.App {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % bin_str
    
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return
# set_png_as_page_bg("download.png")





if "button_clicked" not in st.session_state:
    st.session_state.button_clicked=False
if "button_ok" not in st.session_state:
    st.session_state.button_clicked=False
if "button_ok" not in st.session_state:
    st.session_state.button_clicked=False
if "button_nf" not in st.session_state:
    st.session_state.button_clicked=False
def callback():
    st.session_state.button_clicked=True
    st.session_state.button_ok=True
    st.session_state.button_nf=False
def main():
    
    
    
    
    x=""
    menu=["Home","About"]
    choise=st.sidebar.selectbox("Menue",menu)
    st.title("Audio Noise Canceler App ")
    if choise=="Home":
        
        
        
        
        def sidebar_b(side_bg):

            side_bg_ext = 'png'

            st.markdown(
              f"""
              <style>
              div:last-child > div:first-child {{
                  background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
                  background-size: cover;
              }}
              </style>
              """,
              unsafe_allow_html=True,
              )


        side_bg = "pp.png"
        sidebar_b(side_bg)
        
        
        st.subheader("Home")

        path  = "C:\\Users\\Tigist\\Desktop"
        filelist = os.listdir(path)
        menu = []
        for file in filelist:
            if file.endswith('.mp3') or file.endswith('.m4a') or file.endswith('.wav'):
                menu.append(file)
        file = st.selectbox('Choose audio',menu)
        if  not  file.endswith('.wav'):
            wav_audio = AudioSegment.from_file(file)

            wav_audio.export(file[:-3] + "wav", format="wav")
        
        file = file[:-3] + "wav"
        
        
        if file is not None:
            
            last_fun(file)
    #To read file as bytes:
            
            col1, col2, col3 = st.columns([1,1,1])

            with col1:
                   if st.button('Play') or st.session_state.button_clicked:
                        audio_file = open(file,'rb') #enter the filename with filepath

                        audio_bytes = audio_file.read() #reading the file

                        st.audio(audio_bytes, format='audio/ogg') #displaying the audio    
                        
            with col2:
                if st.button('Spectrogram')or st.session_state.button_clicked: 
                    fig = plt.figure(figsize = (10, 5))
                    samplingFrequency, signalData = wavfile.read(file)
                    if signalData.ndim > 1:
                        plt.specgram(signalData[:,0],Fs=samplingFrequency,cmap="rainbow")
                    else:
                        plt.specgram(signalData,Fs=samplingFrequency,cmap="rainbow")
                    plt.xlabel('Time')

                    plt.ylabel('Frequency')
                    st.pyplot(fig)
            with col3:
                if st.button('Waveform')or st.session_state.button_clicked:
                    fig = plt.figure(figsize = (10, 5))
                    samplingFrequency, signalData = wavfile.read(file)
                    plt.plot(signalData)

                    plt.xlabel('Time')
           
                    plt.ylabel('Amplitude')
                    st.pyplot(fig)
         

       
        if st.button("Type of noise"):
            if file is not None:
                x=connect_ml_and_db(file)
                st.write(x)
            else:
                st.write("No input file detected")
        if st.button("Matching Noises",on_click=callback) or st.session_state.button_clicked:
            one = st.checkbox('type1')
            two = st.checkbox('type2')
            
            if file is not None:
            
                pathfile=test_both_ml_and_db.pth
                titlelist = []
                if two:
                    titlelist = dbw.identify2(conn, pathfile,-9)
    #                 tolerance = 0
                    if len(titlelist)>10 and len(titlelist)<400:
    #                     st.write("The number of matching noise is greater than 10 it is better to modify the tolerance. greater magnituede of tolerance gives better result")

                        tolerance = st.sidebar.slider("Select Matching tolerance level", -100.0, 10.0,-9.0)
                        st.sidebar.write('You selected:', tolerance)
                        titlelist = dbw.identify1(conn, pathfile, tolerance)
                    elif len(titlelist)>=400:
                        st.sidebar.write("no matching noise detected please decrease the magnitude of tolerance")

                        tolerance = st.sidebar.slider("Select Matching tolerance level", -100.0, 10.0,-9.0)
                        titlelist = dbw.identify2(conn, pathfile,tolerance)
                        st.sidebar.write('You selected:', tolerance)
                    else:
                        tolerance = st.sidebar.slider("Select Matching tolerance level", -100.0, 10.0,-9.0)
                        st.sidebar.write('You selected:', tolerance)
                        titlelist = dbw.identify2(conn, pathfile,tolerance)
                elif one:
                    titlelist = dbw.identify1(conn, pathfile,-9)
    #                 tolerance = 0
                    if len(titlelist)>10 and len(titlelist)<400:
    #                     st.write("The number of matching noise is greater than 10 it is better to modify the tolerance. greater magnituede of tolerance gives better result")

                        tolerance = st.sidebar.slider("Select Matching tolerance level", -400, 10,-9)
                        st.sidebar.write('You selected:', tolerance)
                        titlelist = dbw.identify1(conn, pathfile, tolerance)
                    elif len(titlelist)>=400:
                        st.sidebar.write("no matching noise detected please decrease the magnitude of tolerance")

                        tolerance = st.sidebar.slider("Select Matching tolerance level", -400, 10,-9)
                        titlelist = dbw.identify1(conn, pathfile,tolerance)
                        st.sidebar.write('You selected:', tolerance)
                    else:
                        tolerance = st.sidebar.slider("Select Matching tolerance level", -400, 10,-9)
                        st.sidebar.write('You selected:', tolerance)
                        titlelist = dbw.identify1(conn, pathfile,tolerance)
                else:
                    st.write("Please select either type1 or type2")
                if one or two:
                    st.sidebar.write("number of matched noises: ",len(titlelist))
    #                 if  st.button("OK",on_click = callback) or st.session_state.button_ok:
                    men=[]
                    option = ""

    #                     st.write("number of matched noises: ",len(titlelist))


                    for title in titlelist:

                        men.append(title+".wav")
    #                 option = st.selectbox('Choose noise to be cancelled',men,on_change = callback)

    #                 st.write('You selected:', option)

                    if 0 < len(men) <= 40:
    #                         st.text(men)
                        s1 = AudioSegment.from_file(noise_path+x+"\\wav\\"+men[0])
                        for m in range(1,len(men)):
                            pht=noise_path+x+"\\wav\\"+men[m]
                            s2 =  AudioSegment.from_file(pht)
                            s1 = s1.overlay(s2)
                        s1.export("noise.wav",format = "wav")
                    col1, col2, col3 = st.columns([1,1,1])
    #                     n_path=noise_path+x+"\\wav\\"+option
                    n_path = "noise.wav"


                    with col1:
                           if st.button('Play_noise',on_click = callback):
                                st.session_state.button_ok = True
                                audio_file = open(n_path,'rb') 

                                audio_bytes = audio_file.read() #reading the file

                                st.audio(audio_bytes, format='audio/ogg') #displaying the audio    

                    with col2:
                        if st.button('noise_Spectrogram',on_click = callback):
                            st.session_state.button_ok = True
                            fig = plt.figure(figsize = (10, 5))
                            samplingFrequency, signalData = wavfile.read(n_path)
                            if signalData.ndim > 1:
                                
                                plt.specgram(signalData[:,0],Fs=samplingFrequency,cmap="rainbow")
                            else:
                                plt.specgram(signalData,Fs=samplingFrequency,cmap="rainbow")
                            plt.xlabel('Time')
                            plt.ylabel('Frequency')


                            st.pyplot(fig)
                    with col3:
                        if st.button('noise_Waveform',on_click = callback):
                            st.session_state.button_ok = True
                            fig = plt.figure(figsize = (10, 5))
                            samplingFrequency, signalData = wavfile.read(n_path)
                            plt.plot(signalData)

                            plt.xlabel('Time')

                            plt.ylabel('Amplitude')

                            st.pyplot(fig)
                    level = st.sidebar.slider("Select the quality level", 1, 15,3)

                    st.sidebar.text('Selected: {}'.format(level))

                    if st.button("get noise reduced signal",on_click = callback) :
                        
                        
                        col1, col2, col3 = st.columns([1,1,1])
                        st.session_state.button_ok = True
                        
                        x=connect_ml_and_db(file)

    #                     ph=noise_path+x+"\\wav\\"+option
                        ph = "noise.wav"
                        ra,sign = wav.read(file)
                        pa=run(ph,file,ra,level)

    #                     if level>0:
    #                         for i in range(level):
    #                             pa=run(ph,pa,ra) 

                        song = AudioSegment.from_wav(pa) 
                        song.export("out.wav", format='wav')
                        rate,signal = wav.read("out.wav")
                        lp_filter(signal,rate)
                        song = AudioSegment.from_wav("out.wav")+ 10
                        song.export("out.wav", format='wav')
                        pp = "out.wav"
                        with col1:
                               
                                if st.button('Play audio') or st.session_state.button_clicked:
                                    st.session_state.button_ok = True
                                    audio_file = open("out.wav",'rb')

                                    audio_bytes = audio_file.read() #reading the file

                                    st.audio(audio_bytes, format='audio/ogg') #displaying the audio    

                        with col2:
                            if st.button('Spectrogram of audio') or st.session_state.button_clicked: 
                                st.session_state.button_ok = True
                                fig = plt.figure(figsize = (10, 5))
                                samplingFrequency, signalData = wavfile.read(pp)
                                if signalData.ndim > 1:
                                    plt.specgram(signalData[:,0],Fs=samplingFrequency,cmap="rainbow")
                                else:
                                    plt.specgram(signalData,Fs=samplingFrequency,cmap="rainbow")

                                plt.xlabel('Time')

                                plt.ylabel('Frequency')
                                st.pyplot(fig)
                        with col3:
                            st.session_state.button_ok = True
                            if st.button('Waveform of audio') or st.session_state.button_clicked:
                                
                                fig = plt.figure(figsize = (10, 5))
                                samplingFrequency, signalData = wavfile.read(pp)
                                plt.plot(signalData)

                                plt.xlabel('Time')

                                plt.ylabel('Amplitude')
                                st.pyplot(fig)
                        
                        
                        
    #                         st.text('Selected: {}'.format(level))
                       
                         

#                         audio_bytes = audio_file.read() 
#                         st.write("Here is noise reduced audio")

#                         st.audio(audio_bytes, format='audio/ogg')
    #                 if pathfile is None:
    #                     log.error('expected a pathfile for "identify" command')
            else:
                  st.write("No input file detected") 


                    


    else:
          
        st.subheader("Model Accuracy")
        st.write("for each class")
        labels = [97.395833,97.196262,97.076023,96.969697,92.746114,91.625616,91.578947,89.302326,89.024390,86.818182]
        Class = ["Air Conditioner","Jackhammer","Siren","Gun Shot","Engine Idling","Dog bark","Drilling","Street Music","Car Horn","Children Playing"]
        st.dataframe({
        'CLASS': Class,
        'ACCURACY': labels
        })
        
        st.write("General accuracy")
        accuracy = [92.4574,92.4574]
        lab = ["","tarining","test"]
        loss = [0.3245 ,0.3112]
        st.dataframe({
        
        'LOSS': loss,
        'ACCURACY': accuracy
         
        })
        
        st.write("This machine learning model is used to classify the noise that exist in audio file. It has 10 classes of noise that found mostly in our enviroment. After the machine learning model classifies the noise that exist in sound the matched noise will be selected from the corresponding database of class and when the user wants to cancel the noise from the input sound that matching noise will be loaded and the power spectral of the noise that loaded and the spectrum of input audio be calculated and the avrage spectrum of noise will be subtracted which will give us the noise reuced audio.")
    
    
    
if __name__=="__main__":
    main()


