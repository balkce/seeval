# Speech Enhancement Evaluation

## Requirements
```
pip install json speechbrain denoiser mireval
```

## Corpus Creation

Change directory to: 
```
cd corpus
```

And run:

```
python mini_librispeech_prepare-script.py mini_librispeech_prepare-script.yaml
```

This will:
- Download the LibriSpeech corpus, the RIR and Noise datasets
- Create the data manifest of the downloaded corpora (train.json, valid.json, test.json)
- Create a subfolder named as the seed in which the evaluation corpus will be created
- Create the evaluation corpus

All of this will be created under the path: _corpus/librispeech_

You're welcome to change this in _mini_librispeech_prepare-script.yaml_ by changing the _data_folder_ field. However, you should also change this  accordingly in the paths inside the file _corpus/reverb_large.csv_

## Running the Evaluation

First, create the manifest of the evaluations corpus (_dataset_wavs.txt_), by running:

```
python create_dataset_wavs.py CORPUSPATH
```

where _CORPUSPATH_ is the path to the evaluation corpus, such as: _./corpus/librispeech/4234_

You can either run the evaluation on a sub-set of the evaluation corpus, or on the whole evaluation corpus.

### Option 1: Running on a sub-set of the evaluation corpus

Create the sub-set (_dataset_wavs_subset.txt_):

```
python create_dataset_wavs_subset.py SUBSETLEN VALIDJSONPATH
```

where:
- _SUBSETLEN_ is the length of the created sub-set, such as: _100_
- _VALIDJSONPATH_ is the path to the to the valid.json manifest file of LibriSpeech, such as: _./corpus/valid.json_

And run:

```
python SEEval_script.py
```

### Option 2: Running on the whole evaluation corpus
Just run:

```
python SEEval_script.py do_full_run
```
