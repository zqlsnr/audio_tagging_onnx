"""
#!/usr/bin/python3
#-*- coding: utf-8 -*-
@FileName: audio_tagging_inference.py
@Time: 2022/10/21 15:11        
@Author: zql
"""
import onnxruntime as ort
import numpy as np
import librosa
import csv

required_length = 1

sample_rate = 32000

model_name = 'MobileNetV1_1s.onnx'

# start an onnx inference session
ort_session = ort.InferenceSession(model_name)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
print("model load success...")

with open('class_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

audio_path = "example.wav"
(waveform, _) = librosa.load(audio_path, sr=sample_rate, mono=True)
senconds = int(len(waveform) / sample_rate)

for i in range(0, senconds):
    sencond_data = waveform[int(i) * sample_rate:int(i+1) * sample_rate]
    # make test audio to meet the required length
    required_samples = required_length*sample_rate
    sencond_data = sencond_data.repeat(int(np.ceil(required_samples/len(sencond_data))))[:required_samples] #  duplicate waveform and cut off extra length

    sencond_data = sencond_data[None, :]
    # inference
    outputs = ort_session.run([output_name],  {input_name: sencond_data.astype(np.float32)})

    sorted_indexes = np.argsort(outputs[0][0])[::-1]

    # Print audio tagging top probabilities
    for k in range(5):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], outputs[0][0][sorted_indexes[k]]))