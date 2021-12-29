# Spleeter2PyTorch
Convert Spleeter pretrained models (TensorFlow) to PyTorch models + torchscript export + unmix example

Inspired by https://github.com/tuan3w/spleeter-pytorch and https://github.com/generalwave/spleeter.pytorch
It fixes issues from these models, make them simpler, and make the 5stems model work as well.

All you have to do is drop 2stems, 4stems and 5stems checkpoint folders from Spleeter in this folder, then run
python -m convert

Requires Tensorflow and PyTorch
