# seeval
Speech Enhancement Evaluation

## Requirements
pip install json
pip install speechbrain
pip install denoiser
pip install mireval

# Corpus Creation

Change directory to: corpus

And run:

python mini_librispeech_prepare-script.py mini_librispeech_prepare-script.yaml

This will:
- Download the LibriSpeech corpus, the RIR and Noise datasets
- Create the data manifest of the downloaded corpora (train.json, valid.json, test.json)
- Create a subfolder named as the seed in which the evaluation corpus will be created
- Create the evaluation corpus

All of this will be created under the path: corpus/librispeech

You're welcome to change this in mini_librispeech_prepare-script.yaml by changing the data_folder to a path you're comfortable with. However, you should also change this path correspondingly in the paths inside the file corpus/reverb_large.csv.

# Running the Evaluation

First, create the manifest of the evaluations corpus (dataset_wavs.txt), by running:

python create_dataset_wavs.py corpus_path

where corpus_path is the path to the evaluation corpus, such as: ./corpus/librispeech/4234

## Running on a sub-set of the evaluation corpus

Create the sub-set (dataset_wavs_subset.txt):

python create_dataset_wavs_subset.py subset_len valid_json_path

where:
- subset_len is the length of the created sub-set, such as: 100
- valid_json_path is the path to the to the valid.json manifest file of LibriSpeech, such as: ./corpus/valid.json

And run:

python SEEval_script.py

## Running on the whole evaluation corpus
Just run:

python SEEval_script.py do_full_run

