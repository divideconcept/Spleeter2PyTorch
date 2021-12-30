# Spleeter2PyTorch
Convert Spleeter pretrained models (TensorFlow) to PyTorch models + torchscript export + unmix example

Inspired by https://github.com/tuan3w/spleeter-pytorch and https://github.com/generalwave/spleeter.pytorch
It fixes issues from these models, make them simpler, and make the 5stems model work as well.

All you have to do is drop 2stems, 4stems and 5stems checkpoint folders from Spleeter in this folder, then run
python -m convert

You can optionally edit the first lines of convert.py:
```
input_folders={2:'2stems', 4:'4stems', 5:'5stems'} #you can optinally only keep the entries you need
input_shape=(1,2,512,1536) #used for tracing (B x C x T x F)
output_folder='' #where to store the traced models and the unmixed wav files (if an example file is provided)
unmix_example='unmix.wav' #must be a wav file, can be empty if no example is provided
```

Requires Tensorflow and PyTorch
