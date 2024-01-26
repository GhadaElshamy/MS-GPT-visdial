Improving Visual Dialog Training Using Generative Pre-Trained Model
===========================================

## Table of Contents
* [Setup and Dependencies](#Setup-and-Dependencies)
* [Download Data](#Download-Data)
* [Pre-trained Checkpoints](#Pre-trained-Checkpoints)
* [Training](#Training)
* [Evaluation](#Evaluation)
* [Acknowledgements](#Acknowledgements)
* [License](#License)<br><br>


Setup and Dependencies
----------------------
This code is implemented using PyTorch v1.7+ and uses single A100 GPU <br>

1. Install Anaconda or Miniconda distribution based on Python3.8+ from their [downloads' site][1] or you can use google colab.
2. Clone this repository and create an environment:

```shell
git clone https://www.github.com/gicheonkang/gst-visdial

#(in case you used conda/miniconda)
conda env create -f env.yml
# activate the environment and install all dependencies
conda activate gst
cd gst-visdial/
```

Download Data
----------------------
Download the preprocessed original VisDial data, collected by [Das et al][2]. It includes Faster R-CNN bounding box image features of the MSCOCO dataset (80G) and preprocessed json files for dialog (2G). 
```shell
chmod +x scripts/download_preprocessed_human_visdial.sh
```


Pre-trained Checkpoints
--------------------------------------
Please download the checkpoints to `checkpoints/` directory.

| Model | Trained Data | Link |
|:-------:|:---------:|:------:|
|Base Model from [VisDial-BERT][3]| CC3M + VQA | [Download](https://www.dropbox.com/s/g38qemmqep1tt1a/basemodel)|


Training
--------
Answer generation model. Nearly 60G gpu memory is required to train the model. The argument `-enc_dec_a` denotes an encoder-decoder model for answerer model, and `gpt2` is the decoder used for answer generation.  
```shell
#Training of Generation
 python train_gen.py \
   -mode vd_train \
   -model enc_dec_a \
   -num_train_samples 0 \
   -num_epochs 70 \
   -batch_size 40 \
   -dec_model "gpt2" \
   -gpu_ids 0 \
   -start_path  checkpoints/basemodel
```


Evaluation
----------
```shell
# Evaluation of Generation V_1.0
!python evaluate_gen.py \
  -mode vd_eval_val \
  -start_path "checkpoints/vd_train__70.ckpt" \
  -save_path results \
  -save_name gen_70_epochs.txt \
  -dec_model "gpt2" \
  -gpu_ids 0
```



Acknowledgements
-----------------
We build our model on [VisDial-BERT][3] and [gst_VisDial][4] as reference code.

License
-------
MIT License


[1]: https://conda.io/docs/user-guide/install/download.html
[2]: https://arxiv.org/pdf/1611.08669.pdf
[3]: https://github.com/vmurahari3/visdial-bert
[4]: https://github.com/gicheonkang/gst-visdial.git

