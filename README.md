##Scene Detection on Videos
=============

Installation
------------

You need to install:
- [Torch7](http://torch.ch/docs/getting-started.html#_)
- [cunn](https://github.com/torch/cunn) for training on GPU
- [cudnn](https://github.com/soumith/cudnn.torch) for faster training on GPU
- [tds](https://github.com/torch/tds) for some data structures
- [display](https://github.com/szym/display) for graphs 

You can install all of these with the commands:
```bash
# install torch first
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh

# install libraries
luarocks install cunn
luarocks install cudnn
luarocks install tds
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```

Model
-----
Siamese Network with Contrastive Loss function

Data Setup 
----------
By default, we assume you have a text file that lists your dataset. This text does not store your dataset; it just lists filepaths to it, and any meta data. Each line in this text file represents one training example, and its associated category ID. The syntax of the line should be: 
```
<filename><tab><number>
```
For example:
```
bedroom/IMG_050283.jpg    5
bedroom/IMG_237761.jpg    5
office/IMG_838222.jpg     10
```
The `<number>` should start counting at 1. 

After you create this file, open `main.lua` and change `data_list` to point to this file. You can specify a `data_root` too, which will be prepended to each filename.

Training
--------
Finally, to start training, just do:

```bash
$ CUDA_VISIBLE_DEVICES=0 th main.lua
```

During training, it will dump snapshots to the `checkpoints/` directory every epoch. Each time you start a new experiment, you should change the `name` (in `opt`), to avoid overwriting previous experiments. The code will not warn you about this (to keep things simple).

Evaluation
----------
To evaluate your model, you can use the `eval.lua` script. It mostly follows the same format as `main.lua`. It reads your validation/testing dataset from a file similar to before, and sequentially runs through it, calculating the accuracy.

Graphics, Logs
--------------
If you want to see graphics and the loss over time, in a different shell on the same machine, run this command:
```bash
$ th -ldisplay.start 8000 0.0.0.0
```
then navigate to ```http://HOST.csail.mit.edu:8000``` in your browser. Every 10th iteration it will
push graphs. 
=======
