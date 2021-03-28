
import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pylab

import sys
import warnings

import filetype

if not sys.warnoptions:
	warnings.simplefilter("ignore")



def save_spectrogram(sr, log_freq_spec, hop_length, songpath, y_axis='linear'):
	fig = plt.Figure()
	canvas = FigureCanvasTkAgg(fig)
	ax = fig.add_subplot(111)
	p = librosa.display.specshow(log_freq_spect, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
	fig.savefig(songpath+".png")



data_path = "dataset/"

labels = ['Progressive_Rock_Songs/', 'Not_Progressive_Rock/Other_Songs/', 'Not_Progressive_Rock/Top_Of_The_Pops/']

songs_paths = []
prog_songs_paths = []
nonprog_songs_paths = []
for label in labels:
	folder = os.path.join(data_path, label)
	for song in os.listdir(folder):
		song_path = os.path.join(folder, song)
		songs_paths.append(song_path)
		temp_path = song_path.split("/")
		if temp_path[1]=="Progressive_Rock_Songs":
			prog_songs_paths.append(song_path)
		if temp_path[1]=="Not_Progressive_Rock":
			nonprog_songs_paths.append(song_path)

print("Total songs", len(songs_paths))
print("Num of prog:" , len(prog_songs_paths))
print("Num of nonprog: ", len(nonprog_songs_paths))


FRAME_SIZE = 2048
HOP_SIZE = 512

stft_all = np.array([])
#STFT
i=0
for song in songs_paths:
	#load songs
	sname = song.split("/")
	song_name = sname[-1]
	ext = song_name.split(".")
	ftype = ext[-1]
	# import pdb;pdb.set_trace();
	if ftype=="mp3":
		audio, sr = librosa.load(song)
		kind = filetype.guess(song)
		#stft
		stft = librosa.stft(audio, n_fft = FRAME_SIZE, hop_length = HOP_SIZE)
		log_freq_spect = librosa.power_to_db(np.abs(stft)**2)
		#save_spectrogram(audio, log_freq_spect, HOP_SIZE, "Spectrograms/"+song, y_axis="log")
		pylab.figure(figsize=(3,3))
		pylab.axis('off') 
		pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
		librosa.display.specshow(log_freq_spect, sr=sr, hop_length=HOP_SIZE, x_axis="time", y_axis="log")
		if not os.path.exists('Spectrograms'):
			os.makedirs('Spectrograms')
		pylab.savefig("Spectrograms/"+song_name+'.jpg', bbox_inches=None, pad_inches=0)
		pylab.close()
		print("Plotted "+song_name+" - "+str(i)+" of "+str(len(songs_paths)))
		i=i+1






