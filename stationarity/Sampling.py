# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: thesis
#     language: python
#     name: thesis
# ---

# %%
import json
from scipy.io import wavfile as wav
from statsmodels.tsa.stattools import adfuller, kpss
import gpytorch
import torch
import GPy
import math
import numpy
import os
import warnings
import matrixprofile as mp
from tqdm import tqdm

# %pylab inline
figsize(15, 5)

# %% [markdown]
# # MIDI processor class
#
# Find segments where a new note has not been played in the last 300ms, while there are still active notes (key not released).

# %%
import pandas as pd
import mido

# 13230 data points equates to 300ms of audio when sampling at 44kHz
stationary_length = 13230

class MidiProcessor():
    def __init__(self, filename):
        self.stationary = []
        self.delta = 0
        self.tempo = 500000
        self.beats_per_seconds = 1000000 / self.tempo
        self.notes_playing = []
        self.total_ticks = 0
        self.datapoints = []
        self.find_stationary_segments(filename)
        
    def ticks_to_ms(self, ticks):
        beats = ticks / self.ticks_per_beat
        seconds = beats / self.beats_per_seconds
        msec = seconds * 1000.
        return msec
        
    def process_note_on(self, note_on):
        note = note_on.note
        velocity = note_on.velocity
        
        if velocity == 0:
            self.delta += note_on.time
            if note in self.notes_playing:
                self.notes_playing.remove(note)
        else:
            self.delta = 0
            self.new_segment = True
            self.notes_playing.append(note)
        
    def process_change(self, change):
        self.delta += change.time
        
    # 13230 data points corresponds to 300ms
    # So if we want to get to index 13230 with a value of 300 as input, we multiply 300 by 13230 / 300
    def time_to_index(self, time):
        factor = stationary_length / 300
        index = time * factor
        return index
        
    def add_current_position(self):
        elapsed_time = self.ticks_to_ms(self.total_ticks)
        index = round(self.time_to_index(elapsed_time))
        self.stationary.append(elapsed_time)
        self.datapoints.append(index)
        
    def check_length(self):
        if self.ticks_to_ms(self.delta) > 300:
            is_silence = not self.notes_playing
            if not is_silence and self.new_segment:
                self.add_current_position()
                self.new_segment = False
        
    def find_stationary_segments(self, filename):
        midi = mido.MidiFile(filename)
        meta_track = midi.tracks[0]
        sound_track = midi.tracks[1]
        
        self.ticks_per_beat = midi.ticks_per_beat
        
        for msg in sound_track:
            msg_type = msg.type
            self.total_ticks += msg.time
            
            is_note_on = msg_type == "note_on"
            if is_note_on:
                self.process_note_on(msg)
            else:
                self.process_change(msg)
                
            self.check_length()
            
        return self.datapoints


# %% [markdown]
# # Using the dataset metadata

# %%
with open("../data/maestro/maestro-v3.0.0.json", 'r') as file:
    data = json.load(file)
    
composers = data["canonical_composer"]
pieces = data["canonical_title"]
filenames = data["audio_filename"]

info = {}

for i in range(len(filenames)):
    i = str(i)
    composer = composers[i]
    piece = pieces[i]
    filename = filenames[i][5:-4]
    
    info[filename] = {"composer": composer, "piece": piece, "filenumber": i}


# %% [markdown]
# # Iteration over all MIDI files

# %%
import random

segments = {}

folder = "../data/maestro/"
for f in os.walk(folder):
    f_name = f[0]
    if f_name is not folder:
        performances = list(filter(lambda name : name.endswith('midi'), f[2]))
        for performance in tqdm(performances):
            full_filename = f_name + "/" + performance
            ss = MidiProcessor(full_filename).datapoints
            random.shuffle(ss)
            
            file_info = info[performance[:-5]]
            key = ":".join([file_info["composer"], file_info["piece"]])
            
            if key not in segments:
                segments[key] = []
            
            for segment in ss:
                value = ":".join([file_info["filenumber"], str(segment)])
                segments[key].append(value)

# %% [markdown]
# # Saving and retrieving segments

# %%
with open("segments.json", "w") as file:
    json.dump(segments, file)

# %%
with open("segments.json", "r") as file:
    segments = json.load(file)


# %% [markdown]
# # Sampling

# %%
class Sampler():
    def __init__(self, dataset):
        self.dataset = dataset
        self.samples = []
        
    def take_sample(self, key):
        sample = random.sample(self.dataset[key], 1)
        if sample in self.samples:
            self.take_sample(key)
        else:
            self.samples.append(sample)
        
    def take_n_samples(self, N):
        segment_amount = len(self.dataset.values())
        
        for key in self.dataset:
            segments_for_key = self.dataset[key]
            segments_for_key_amount = len(segments_for_key)
            
            percentage = segments_for_key_amount / segment_amount
            amount_to_take_for_key = round(percentage)
            
            for i in range(amount_to_take_for_key):
                self.take_sample(key)
                
sampler = Sampler(segments)
sampler.take_n_samples(100)

# %%
print(sampler.samples)


# %% [markdown]
# # Class for segment

# %%
class Segment():
    def __init__(self, identifier):
        self.filenumber, self.end = self.extract_from_identifier(identifier)    
        self.end = int(self.end)
        self.start = self.end - stationary_length
        
        self.filename = self.get_filename()
        self.data = self.get_data()
        self.show_segment()
        
    def extract_from_identifier(self, identifier):
        return identifier[0].split(":")
    
    def get_filename(self):
        return data["audio_filename"][self.filenumber]
    
    def get_data(self):
        root = "../data/maestro/"
        _, wav_data = wav.read(root + self.filename)
        wav_data = wav_data / amax(abs(wav_data))
        wav_data -= mean(wav_data)
        wav_data = wav_data[self.start:self.end, :]
        return wav_data
    
    def show_segment(self):
        plot(self.data[:, 0])
        show()
        plot(self.data[:, 1], color="purple")
        show()
        
seg = Segment(sampler.samples[1])

# %%
