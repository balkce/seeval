import glob
import os
import sys

corpus_path = ""

if(len(sys.argv) > 1):
  corpus_path = sys.argv[1]
else:
  print("Need the path to the evaluation corpus (such as ./corpus/librispeech/4234)")
  exit()

paths = glob.glob(corpus_path+"/**/noisy.wav",recursive = True)

f = open("dataset_wavs.txt", "w")

for path in paths:
  path_dir = os.path.dirname(path)
  print(path_dir)
  f.write(path_dir+"/\n")

f.close()
