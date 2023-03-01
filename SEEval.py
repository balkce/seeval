# also requires: pip install denoiser
import torch
import torchaudio
import math
import mir_eval
import time
import os
import sys
import psutil
import shutil
import speechbrain as sb
from speechbrain.pretrained import SpectralMaskEnhancement
from speechbrain.pretrained import WaveformEnhancement
from denoiser import pretrained
from denoiser.dsp import convert_audio

class SEEval():
  dir_path=""
  results_dict = {
    "wav_id": "",
    "babble_snr": 0.0,
    "babble_num": 0,
    "noise_snr": 0.0,
    "reverb_scale": 0.0,
    "window_size": 0,
    "response_time": 0.0,
    "SIR": 0.0
  }
  curr_model = {
    "id": 0,
    "type": "",
    "model": 0
  }
  
  
  def __init__(self, path=""):
    self.dir_path = path
  
  def load_model(self, model_number):
    self.curr_model["id"] = model_number
    if model_number == 0:
      self.curr_model["type"] = "enhance"
      self.curr_model["model"] = SpectralMaskEnhancement.from_hparams(
                  source="speechbrain/metricgan-plus-voicebank",
                  savedir="pretrained_models/metricgan-plus-voicebank",
                  #run_opts={"device":"cuda"}
                )
    elif model_number == 1:
      self.curr_model["type"] = "enhance"
      self.curr_model["model"] = WaveformEnhancement.from_hparams(
                  source="speechbrain/mtl-mimic-voicebank",
                  savedir="pretrained_models/mtl-mimic-voicebank",
                  #run_opts={"device":"cuda"}
                )
    elif model_number == 2:
      self.curr_model["type"] = "notspeechbrain"
      self.curr_model["model"] = pretrained.dns64()
    else:
      print("Invalid model number (0...2).")

  def run_model(self, dir_path, window_size):
    dir_path_elems = dir_path.split("/")
    
    while("" in dir_path_elems):
      dir_path_elems.remove("")
    
    self.results_dict["wav_id"] = dir_path_elems[-1]
    self.results_dict["babble_snr"] = float(dir_path_elems[-2].split("_")[-1])
    self.results_dict["babble_num"] = int(dir_path_elems[-3].split("_")[-1])
    self.results_dict["noise_snr"] = float(dir_path_elems[-4].split("_")[-1])
    self.results_dict["reverb_scale"] = float(dir_path_elems[-5].split("_")[-1])
    self.results_dict["window_size"] = window_size
    
    wav_path = dir_path+"noisy.wav"
    info_path = dir_path+"info.txt"
    
    int_paths = open(info_path).readlines()
    
    if (os.path.exists("noisy.wav")):
      os.remove("noisy.wav")
    if (self.curr_model["type"] == "enhance"):
      noisy = self.curr_model["model"].load_audio(
          wav_path
      ).unsqueeze(0)
    else:
      noisy, sr = torchaudio.load(wav_path)
    
    result_len = noisy.size(1)
    if window_size == 0:
      one_window = True
      window_size = result_len
      window_num = 1
    else:
      one_window = False
      window_num = math.floor(result_len/window_size)
      if result_len % window_size > 0:
        window_num += 1
    
    #creating original signals
    references = torch.zeros(1,result_len)
    estimations = torch.zeros(1,result_len)
    clean = sb.dataio.dataio.read_audio(str(int_paths[0].rstrip()))
    if clean.size(0) < result_len: #shouldn't happen, but c'est la vie
      clean = torch.cat((clean.unsqueeze(0),torch.zeros(1,result_len-clean.size(0))),1).squeeze(0)
    references[0,:] = clean[:result_len].detach()
    
    win_result = torch.zeros(1,window_num*window_size)
    win_result_len = win_result.size(1)
    noisy_win = torch.zeros(1,window_size)
    exec_time_mean = 0.0
    exec_time_i = 0
    for i in range(0,window_num):
      if one_window:
        if self.curr_model["id"] == 1:
          tmp_result_len = 2**(result_len - 1).bit_length()
          noisy_win = torch.cat((noisy,torch.zeros(1,tmp_result_len-result_len)),1)
        else:
          noisy_win = noisy
      else:
        if i == window_num-1:
          noisy_win = torch.cat((noisy[:,(window_size*i):],torch.zeros(1,win_result_len-result_len)),1)
        else:
          noisy_win = noisy[:,(window_size*i):(window_size*(i+1))]
      
      if (self.curr_model["type"] == "enhance"):
        start_time = time.time()
        model_result = self.curr_model["model"].enhance_batch(noisy_win, lengths=torch.tensor([1.]))
        exec_time = time.time() - start_time
        if one_window and self.curr_model["id"] == 1:
          model_result = model_result[:,:result_len]
        
        win_result[:,(window_size*i):(window_size*(i+1))] = model_result
      else:
        start_time = time.time()
        #wav = convert_audio(noisy_win, 16000, model.sample_rate, model.chin)
        with torch.no_grad():
          model_result = self.curr_model["model"](noisy_win)[0]
        exec_time = time.time() - start_time
        win_result[:,(window_size*i):(window_size*(i+1))] = model_result
      exec_time_mean += exec_time
      exec_time_i += 1
    
    exec_time_mean /= exec_time_i
    
    estimations[0,:] = win_result[:,:result_len].squeeze(0).detach()
    
    #sb.dataio.dataio.write_audio("result.wav",estimations[0,:],16000)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(references.numpy(), estimations.numpy())
    self.results_dict["SIR"] = sdr[0].item()
    
    self.results_dict["response_time"] = exec_time_mean
    return self.results_dict

def extract_id_from_path(dir_path):
  dir_path_elems = dir_path.split("/")
  while("" in dir_path_elems):
    dir_path_elems.remove("")
  return dir_path_elems[-1]


do_full_run = False

if(len(sys.argv) > 1):
  if sys.argv[1] == "do_full_run":
    do_full_run = True

eval_window_size = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 0]

paths_file = open("dataset_wavs.txt")
paths = paths_file.readlines()
paths_file.close()
paths_num = len(paths)

if do_full_run:
  wavs_subset = []
else:
  f = open("dataset_wavs_subset.txt", "r")
  wavs_subset_filecontents = f.read()
  wavs_subset = wavs_subset_filecontents.split("\n")
  f.close()
  while("" in wavs_subset):
      wavs_subset.remove("")

if os.path.exists("curr_dataset_wav.txt"):
  curr_path_file = open("curr_dataset_wav.txt","r")
  path_i_start = int(curr_path_file.read())
  curr_path_file.close()
  
  curr_model_file = open("curr_model.txt","r")
  model_i_start = int(curr_model_file.read())
  curr_model_file.close()
  
  curr_len_file = open("curr_len.txt","r")
  win_i_start = int(curr_len_file.read())
  curr_len_file.close()
else:
  curr_path_file = open("curr_dataset_wav.txt","w")
  curr_path_file.write(str(0))
  curr_path_file.close()
  path_i_start = 0
  
  curr_model_file = open("curr_model.txt","w")
  curr_model_file.write(str(0))
  curr_model_file.close()
  model_i_start = 0
  
  curr_len_file = open("curr_len.txt","w")
  curr_len_file.write(str(0))
  curr_len_file.close()
  win_i_start = 0
  
  eval_results_file = open("curr_results.csv","w")
  eval_results_file.write("model,wav_id,reverb_scale,babble_num,babble_snr,noise_snr,window_size,response_time,SIR,mem\n")
  eval_results_file.close()

seeval = SEEval(".")

this_process = psutil.Process(os.getpid())

window_size = eval_window_size[win_i_start];

print("Evaluating model "+str(model_i_start)+" with window size: "+str(window_size))
seeval.load_model(model_i_start)
ini_mem = this_process.memory_info().rss

for path_i in range(path_i_start,paths_num):
  curr_path_file = open("curr_dataset_wav.txt","w")
  curr_path_file.write(str(path_i))
  curr_path_file.close()
  
  path = paths[path_i].rstrip()
  path_id = extract_id_from_path(path)
  
  if (path_id in wavs_subset) or do_full_run:
    print("  ->",str(path_i),":",path)
    eval_result = seeval.run_model(path, window_size)
    #print(eval_result)
    
    curr_mem = this_process.memory_info().rss
    
    if curr_mem >= ini_mem*10:
      print("Memory grew too much, from",str(ini_mem),"bytes to",str(curr_mem),"bytes. resetting.")
      exit()
    
    eval_results_file = open("curr_results.csv","a")
    eval_results_file.write(str(model_i_start)+","+eval_result["wav_id"]+","+str(eval_result["reverb_scale"])+","+str(eval_result["babble_num"])+","+str(eval_result["babble_snr"])+","+str(eval_result["noise_snr"])+","+str(eval_result["window_size"])+","+str(eval_result["response_time"])+","+str(eval_result["SIR"])+","+str(curr_mem)+"\n")
    eval_results_file.close()

curr_path_file = open("curr_dataset_wav.txt","w")
curr_path_file.write(str(0))
curr_path_file.close()

if win_i_start < len(eval_window_size)-1:
  curr_len_file = open("curr_len.txt","w")
  curr_len_file.write(str(win_i_start+1))
  curr_len_file.close()
else:
  curr_len_file = open("curr_len.txt","w")
  curr_len_file.write(str(0))
  curr_len_file.close()
  
  curr_model_file = open("curr_model.txt","w")
  curr_model_file.write(str(model_i_start+1))
  curr_model_file.close()
  
  shutil.copyfile("curr_results.csv", "curr_results-model_"+str(model_i_start)+".csv")
  
  eval_results_file = open("curr_results.csv","w")
  eval_results_file.write("model,wav_id,reverb_scale,babble_num,babble_snr,noise_snr,window_size,response_time,SIR,mem\n")
  eval_results_file.close()


