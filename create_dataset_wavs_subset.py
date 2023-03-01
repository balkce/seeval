import glob
import os
import json
import random

wav_json_file_path = ""
subset_len = 0

if(len(sys.argv) > 2):
  subset_len = int(sys.argv[1])
  wav_json_file_path = sys.argv[2]
else:
  print("Need the sub-set length (such as 100) and the path to the valid.json manifest file of LibriSpeech (such as ./corpus/valid.json)")
  exit()

subset_len = 100 #needs to smaller than wavs_len for it to be a subset

wav_json_file = open(".corpus/valid.json")
wavs_info = json.load(wav_json_file)
wav_json_file.close()
wavs_keys = list(wavs_info.keys())
wavs_len = len(wavs_info)

#print(wavs_keys)
#print(wavs_len)

random_is = random.sample(range(0, wavs_len), subset_len)

wavs_keys_subset = [wavs_keys[i] for i in random_is]

f = open("dataset_wavs_subset.txt", "w")

for wav_key in wavs_keys_subset:
  print(wav_key)
  f.write(wav_key+"\n")

f.close()
