#!/usr/bin/env/python3
"""The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

"""
import sys
import os
import json
import random
import numpy as np
import torch,torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mini_librispeech_prepare import prepare_mini_librispeech
from speechbrain.processing.speech_augmentation import (
    AddBabble,
    AddNoise,
    AddReverb,
)

# Reading command line arguments
hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

# Load hyperparameters file with command-line overrides
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

# Create experiment directory
sb.create_experiment_directory(
    experiment_directory=hparams["output_folder"],
    hyperparams_to_save=hparams_file,
    overrides=overrides,
)

# Data preparation, to be run on only one process.
sb.utils.distributed.run_on_main(
    prepare_mini_librispeech,
    kwargs={
        "data_folder": hparams["data_folder"],
        "save_json_train": hparams["train_annotation"],
        "save_json_valid": hparams["valid_annotation"],
        "save_json_test": hparams["test_annotation"],
    },
)

if (hparams["only_download"] == False):
  wav_json_file = open(hparams["valid_annotation"])
  wavs_info = json.load(wav_json_file)
  wav_json_file.close()
  wavs_keys = list(wavs_info.keys())
  wavs_len = len(wavs_info)

  noise_snrs = np.arange (hparams["noise_snr_low"], hparams["noise_snr_high"]+hparams["noise_snr_step"], hparams["noise_snr_step"], dtype=np.float64)
  babble_snrs = np.arange (hparams["babble_snr_low"], hparams["babble_snr_high"]+hparams["babble_snr_step"], hparams["babble_snr_step"], dtype=np.float64)
  reverb_dilations = np.arange (hparams["reverb_dilation_low"], hparams["reverb_dilation_high"]+hparams["reverb_dilation_step"], hparams["reverb_dilation_step"], dtype=np.float64)
  babble_speakers = np.arange(0,hparams["babble_speaker_count"]+1,1)

  for reverb_dilation in reverb_dilations:
    print ("Reverb dilation: "+str(reverb_dilation))
    
    if reverb_dilation != 0:
      reverb_csv_path = hparams["reverb_csvs"]["large"]
      openrir_folder=hparams["rir_folder"]
      add_reverb = AddReverb(
                    reverb_prob=1.0,
                    csv_file=reverb_csv_path,
                    rir_scale_factor=reverb_dilation,
                    reverb_sample_rate=16000,
                    clean_sample_rate=16000,
                )
    else:
      reverb_csv_path = None
      reverb_prob=0.0
      openrir_folder=None
    print ("Using the following reverb csv file: "+str(reverb_csv_path))
    
    for noise_snr in noise_snrs:
      print ("  Noise SNR: "+str(noise_snr))
      add_noise = AddNoise(
                  mix_prob=1.0,
                  csv_file=None,
                  num_workers=0,
                  snr_low=noise_snr,
                  snr_high=noise_snr,
                  sorting="random",
                  noise_sample_rate=16000,
                  clean_sample_rate=16000
              )
      for babble_speaker in babble_speakers:
        print ("    # of interf.: "+str(babble_speaker))
        if babble_speaker > 0:
          babble_prob=1.0
        else:
          babble_prob=0.0
        
        for babble_snr in babble_snrs:
          if babble_speaker > 0:
            add_babble = AddBabble(
                    mix_prob=1.0,
                    speaker_count=babble_speaker,
                    snr_low=babble_snr,
                    snr_high=babble_snr,
                )
          else:
            babble_snr = 100.0
          
          print ("      Babble SNR: "+str(babble_snr))
          
          for wav_key, wav_info in wavs_info.items():
            noisy_wav_dir = hparams["output_folder"]+"/reverbscale_"+str(reverb_dilation)+"/noisesnr_"+str(noise_snr)+"/babblenum_"+str(babble_speaker)+"/babblesnr_"+str(babble_snr)+"/"+str(wav_key)+"/"
            
            os.makedirs(noisy_wav_dir, exist_ok=True)
            
            wav_info_file_path = noisy_wav_dir+"info.txt"
            wav_info_file = open(wav_info_file_path,'w')
            
            wav_path = wav_info["wav"].replace("{data_root}",hparams["data_folder"])
            wav_data = sb.dataio.dataio.read_audio(str(wav_path))
            
            wav_info_file.write(wav_path+"\n")
            
            print("        "+wav_key+"->"+wav_path)
            if babble_speaker > 0:
              lengths = torch.ones(babble_speaker+1)
              waveforms_len = len(wav_data)
              waveforms = torch.Tensor(babble_speaker+1,waveforms_len)
              waveforms[0,:] = wav_data.unsqueeze(0)
              for i in range(1,babble_speaker+1):
                int_key_i = random.randint(0,wavs_len-1)
                int_key = wavs_keys[int_key_i]
                int_path = wavs_info[int_key]["wav"].replace("{data_root}",hparams["data_folder"])
                
                wav_info_file.write(int_path+"\n")
                
                int_wav_data = sb.dataio.dataio.read_audio(str(int_path))
                int_wav_data_len = len(int_wav_data)
                
                if (waveforms_len > int_wav_data_len):
                  waveforms_len = int_wav_data_len
                  waveforms = waveforms[:,:waveforms_len]
                elif(waveforms_len < int_wav_data_len):
                  int_wav_data = int_wav_data[:waveforms_len]
                
                waveforms[i,:] = int_wav_data.unsqueeze(0)
            else:
              waveforms = wav_data.unsqueeze(0)
              lengths = torch.ones(1)
            
            # add reverb if needed
            if reverb_dilation != 0:
              noisy_waveforms = add_reverb(waveforms,lengths)
            else:
              noisy_waveforms = waveforms
              
            # add babble if needed
            if babble_speaker > 0:
              noisy_waveforms = add_babble(noisy_waveforms,lengths)
            
            # always add noise
            noisy_waveforms = add_noise(noisy_waveforms,lengths)
            
            
            #writing noisy wav data
            sb.dataio.dataio.write_audio(noisy_wav_dir+"noisy.wav",noisy_waveforms[0,:].squeeze(0),16000)
            
            #writing clean wav data
            #sb.dataio.dataio.write_audio(noisy_wav_dir+"clean.wav",waveforms[0,:].squeeze(0),16000)
            
            #writing clean interferences
            #if babble_speaker > 0:
            #  for i in range(1,babble_speaker+1):
            #    sb.dataio.dataio.write_audio(noisy_wav_dir+"interference_"+str(i)+".wav",waveforms[i,:].squeeze(0),16000)
            
            wav_info_file.close()
            
          if babble_speaker == 0:
            break


