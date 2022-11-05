import os
import eyed3
from pydub import AudioSegment
import logging

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.spatial import distance
import matplotlib
import matplotlib.pyplot as plt
import psycopg2
import credentials
from credentials import DB_USER, DB_PASSWORD,DB_NAME
from test_both_ml_and_db import pth
#import connect_db_ml
#class_type=(connect_db_ml.connect_ml_and_db(pth))
#DB_NAME=class_type


# connect to database
conn = psycopg2.connect(
    host="localhost",
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)



def create_table(conn):
   
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS fingerprint1")
    cur.execute("DROP TABLE IF EXISTS noise")

    cur.execute(
        """CREATE TABLE IF NOT EXISTS noise (
            song_id SERIAL PRIMARY KEY,
            title TEXT not null,
            fingerprinted INT default 0
            )""")
    cur.execute(
        """CREATE TABLE IF NOT EXISTS fingerprint1 (
            sig_id SERIAL PRIMARY KEY,
            song_id INT REFERENCES noise (song_id) ON DELETE CASCADE,
            center INT,
            signature1 NUMERIC,
            signature2 NUMERIC ARRAY
            
            )""")
    conn.commit()

def test_connect(conn):
    """ check connection to postgresql """
    if conn.closed == 0:
        print('Connected to MySQL database')
    else:
        print('Unable to connect')

        
        
        
        
#ANALYSIS

def spectrogram(pathfile):
    
    if not pathfile.endswith(".wav"):
        log.error("audio file must be in wav format")
    else:
        
        framerate, series = wavfile.read(pathfile)
       
        
        if series.ndim==1:
             
            series=np.asfortranarray(np.array([series,series]))
            series=np.transpose(series)
        
        
        series = np.mean(series, axis=1)
        
       
        f, t, spect = signal.spectrogram(
            series,
            fs=framerate,
            nperseg=100000,
            noverlap=0.02*100000,
            window="hamming"
        )
        
        
        return framerate, f, t, spect

def match(f1, f2,n):
   
    tolerance = 10**(n)
    dist = (f1-f2)**2

    return dist < tolerance




def match2(f1, f2,n):
  

 #     tolerance = 0.1
    x = min(len(f1),len(f2))
    tolerance = pow(10,n)

    
    pairs = zip(f1[:x-1], f2[:x-1])
    dists = [distance.euclidean(x, y) for x, y in pairs]
    if all([(d < tolerance) for d in dists]):
        return True
    return False

   





def fingerprint(f, spect):
#     print("dimension", spect.ndim)
    max_f = max(f)
    peaks = np.argmax(spect, axis=0)
    
   
    fingerprints = f[peaks] / max_f

    return np.array(fingerprints)




def fingerprint2(f, spect, framerate):
   
   
    m = 8
    min_f = int((2**-(m+1))*(framerate/2))
    fingerprints = []

    # iterate through all octaves
    for k in range(m):
        start = min_f*(2**k)*10
        end = min_f*(2**(k+1))*10
        sub_f = f[start:end]
        sub_spect = spect[start:end]
        
        if len(sub_f)>0:
            sub_fingerprint = fingerprint(sub_f, sub_spect)
            fingerprints.append(sub_fingerprint)
    fingerprints = np.array(fingerprints).T


    return fingerprints





# DATABASE



def add_song(tup, conn):
    """ add song to noise """
    if isinstance(tup, tuple) or isinstance(tup, list):
        cur = conn.cursor()
        query = "INSERT INTO noise (title) VALUES (%s)"
        cur.execute(query, tup)
        conn.commit()
    else:
        print("tup should be a tuple or list")



def add_fingerprint(filename, t, fingerprints1,fingerprints2, conn):
    
    query = 'INSERT INTO fingerprint1 (song_id, center, signature1, signature2) VALUES (%s,%s,%s,%s)'
    song_id = select_songid(filename, conn)
    for i in range(len(t)):
        val = (song_id, t[i], fingerprints1[i],list(fingerprints2[i]))
        cur = conn.cursor()
        cur.execute(query, val)
        conn.commit()


def update_fingerprinted(song_id, conn):
    
    cur = conn.cursor()
    query = 'UPDATE noise SET fingerprinted = 1 where song_id = %s'
    cur.execute(query, (song_id,))
    conn.commit()

def select_songid(filename, conn):
    
    cur = conn.cursor()
    
    query = 'SELECT song_id from noise WHERE title = %s'
    
   
    val = (filename[:-4])
   
    cur.execute(query, [val])

    records = cur.fetchall()
    
    cur.close()
    
    return records[0][0]



def select_max_song_id(conn):
    
    cur = conn.cursor()
    cur.execute('SELECT MAX(song_id) from noise')
    records = cur.fetchall()
    return records[0][0]

def select_title(song_id, conn):
    cur = conn.cursor()
    query = 'SELECT title from noise WHERE song_id = %s'
    cur.execute(query, (song_id,))
    records = cur.fetchall()
    return records[0][0]


def select_fingerprint1(conn, song_id):
    
    cur = conn.cursor()
    cur.execute('SELECT signature1 FROM fingerprint1 WHERE song_id=%s', (song_id,))
    records = cur.fetchall()
    records = [float(elem[0]) for elem in records]
    return records

def select_fingerprint2(conn, song_id):
    cur = conn.cursor()
    cur.execute('SELECT signature2 FROM fingerprint1 WHERE song_id=%s', (song_id,))
    records = cur.fetchall()
    records = [list(map(float, list(elem[0]))) for elem in records]
    return records



#CONVERSION


def convert(infile):
    try:
       
        filename = os.path.basename(infile)
        outfile = "./noises/wav/" + filename[:-3] +"wav"
        sound = AudioSegment.from_mp3(infile)
        sound.export(outfile, format="wav")
    except OSError:
        logging.error("expected an mp3 file in the directory")

def meta(infile):
    
    try:
        filen = os.path.basename(infile)
#         file = eyed3.load(infile)
        title = (filen[:-4])
        
        tup = (title,)

        return tup


    except OSError:
        logging.error("expected an mp3 file in the directory")



#ADD_TO_DATABASE





def add_single(conn, pathfile):
    
    if pathfile.endswith(".mp3"):
           
            tup = meta(pathfile)
            add_song(tup, conn)
            
            convert(pathfile)
            
            filename = os.path.basename(pathfile)
           
            pathwav = "./noises/wav/" + filename[:-3] + "wav"
            
            framerate, f, t, spect = spectrogram(pathwav)
            fingerprints1 = fingerprint(f, spect)
            fingerprints2 = fingerprint2(f, spect, framerate)
            
            song_id = select_songid(filename, conn)
            
            
            add_fingerprint(filename, t, fingerprints1, fingerprints2,conn)
          
            update_fingerprinted(song_id, conn)

def add_main():
    test_connect(conn)
    create_table(conn)
    
    pth="C:\\Users\\Tigist\\Downloads\\UrbanSound8K.tar\\UrbanSound8K\\UrbanSound8K\\audio\\street_music\\mp3\\"
    #pth="C:\\Users\\Tigist\\Desktop\\noises\\mp3\\"
    create_table(conn)
    file=os.listdir(pth)
    i = 0
    for m in file:
        pathfile=pth+m
        add_single(conn, pathfile)
        i += 1
        print("added",i)







#IDENTIFY MATCHING ONE


def identify1(conn, pathfile,tolerance):
    
    _, f, _, spect = spectrogram(pathfile)
    f_snippet = fingerprint(f, spect)
    
   
    match_count = []

    
    max_song_id = select_max_song_id(conn)
    for i in range(1, max_song_id+1):
        count = 0
        
        records = select_fingerprint1(conn, i)
        
        for f1 in records:
            for f2 in f_snippet:
                if match(f1, f2,tolerance):
                    count += 1
        match_count.append(count)
        
    max_count = max(match_count)
    
    l_songid = [i+1 for i, j in enumerate(match_count) if j == max_count]
    
    titlelist = []
    for song_id in l_songid:
        title = select_title(song_id, conn)
        titlelist.append(title)
        

    return titlelist



def identify2(conn, pathfile,n):

    framerate, f, _, spect = spectrogram(pathfile)
    f_snippet = fingerprint2(f, spect, framerate)
    match_count = []

    max_song_id = select_max_song_id(conn)
    for i in range(1, max_song_id+1):
        count = 0
        records = select_fingerprint2(conn, i)
        for f1 in records:
            f2 = f_snippet[0]
            if match2(f1, f2,n):
                count += 1
        match_count.append(count)

    max_count = max(match_count)
    l_songid = [i+1 for i, j in enumerate(match_count) if j == max_count]
    titlelist = []
    for song_id in l_songid:
        title = select_title(song_id, conn)
        titlelist.append(title)

    return titlelist




