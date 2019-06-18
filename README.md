# KorQuAD
[huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) repository contains op-for-op PyTorch reimplementations, pre-trained models and fine-tuning examples for Google's BERT model.
And as a result of submission using the run_squad.py code provided by [huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_squad.py) repository, it ranked 30th in the test set with EM= 71.47, F1= 89.71, respectively, as shown below. (2019.06.17)

<p align="center">
<img height=50 src="https://github.com/lyeoni/KorQuAD/blob/master/images/submission-3.png" />
</p>

A trained BERT model is publicly available. So, I'm going to cover **the process of submitting** a model and result for official evaluation on KorQuAD. Once your model has been evaluated officially, your scores will be added to the [leaderboard](https://korquad.github.io/). Thus I would assume you already completed model training in KorQuAD, and have a trained model archive.

## Overview
A pre-trained language model, **BERT**, is publicly available. For KorQuAD submission, what you have to do is to **fine-tune** the pre-trained BERT model on KorQuAD. And fine-tuning can be done simply by running `run_squad.py` (in [here](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_squad.py)) on the KorQuAD dataset.

And, even if you fine-tune BERT with the default hyper-parameters in [`run_squad.py`](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_squad.py), we can get the following results (It's a score that can be ranked in the 30th grade, based on June 2019):
```
$ python evaluate-v1.0.py KorQuAD_v1.0_dev.json predictions.json
{"exact_match": 70.29788708001385, "f1": 90.08062112089534}
```

## BERT Fine-tuning
Below table shows the results of the BERT models fine-tuned with various hyper-parameters.

- Model : model name with detailed description
- #steps : number of optimization steps in training
- EM (Exact Match) : ratio of accurate prediction to actualanswer text
- F1 : score on how the actual answer text overlaps with the prediction

|Model|#stpes|EM(dev)|F1(dev)|
|---|---:|---:|---:|
|**_change maximum sequence length_**||||
|BERT-Multilingual - _baseline_ ([description](https://github.com/lyeoni/KorQuAD/blob/master/logs/bert-base-multilingual-cased-default.txt))|7,587|70.298|90.081|
|BERT-Multilingual - _baseline_ (+ max_seq_length=512) ([description](https://github.com/lyeoni/KorQuAD/blob/master/logs/bert-base-multilingual-cased-1.txt))|6,336|70.80|90.104|
|**_change train batch size_**||||
|BERT-Multilingual - _baseline_ (+ train_batch_size=16) ([description](https://github.com/lyeoni/KorQuAD/blob/master/logs/bert-base-multilingual-cased-2.txt))|15,174|70.159|89.818|
|BERT-Multilingual - _baseline_ (+ max_seq_length=512, train_batch_size=16) ([description](https://github.com/lyeoni/KorQuAD/blob/master/logs/bert-base-multilingual-cased-3.txt))|12,669|69.830|89.407|
|**_change learning-rate_**||||
|BERT-Multilingual - _baseline_ (+ learning_rate=3e-5) ([description](https://github.com/lyeoni/KorQuAD/blob/master/logs/bert-base-multilingual-cased-4.txt))|7,587|70.419|90.241|
|BERT-Multilingual - _baseline_ (+ learning_rate=3e-5, train_batch_size=16) ([description](https://github.com/lyeoni/KorQuAD/blob/master/logs/bert-base-multilingual-cased-5.txt))|15,174|70.229|90.114|
|BERT-Multilingual - _baseline_ (+ max_seq_length=512, learning_rate=3e-5)|6,336|**70.644**|**90.246**|
|BERT-Multilingual - _baseline_ (+ max_seq_length=512, learning_rate=3e-5, train_batch_size=16)|12,669|70.419|90.179|


Tips for hyper-parameter tuning
- `pre-trained model` : `bert-large-cased`, pre-trained model, is not recommended
- `num_train_epochs` : set to default value
- `learning rate` : the smaller, the better

## Submission Process
To get official scores on the KorQuAD test set, we must submit(or upload) our model to the [CodaLab](https://worksheets.codalab.org/). This is because the integrity of test results should be preserved.

Here's a detailed process that guides you through the official evaluation of our model.

### 0. Create a CodaLab account
You can create a CodaLab account [here](https://worksheets.codalab.org/).
Click `sign up` in the top-right corner of the CodaLab homepage.

<p align="center">
<img src="https://github.com/lyeoni/KorQuAD/blob/master/images/codalab_signup.png" />
</p>

<br>

### 1. Create a CodaLab worksheet
Click the `My dashboard` in the top-right corner.
<p align="center">
<img src="https://github.com/lyeoni/KorQuAD/blob/master/images/codalab_dashboard.png" />
</p>

Click the `New Worksheet` in the upper-right corner and name your worksheet.
<p align="center">
<img src="https://github.com/lyeoni/KorQuAD/blob/master/images/codalab_worksheet.png" />
</p>

<br>

### 2. Upload trained model with source code
Begin by uploading archive for the trained model onto Codalab.

For example, I trained (BERT) model using [`run_squad.py`](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_squad.py) code from [huggingface/pytorch-pretrained-BERT repository](https://github.com/huggingface/pytorch-pretrained-BERT), and the archive of trained model consists of the following:
- run_squad.py : python script to generate the predictions
- vocab.txt : vocabulary file
- config.json : a configuration file for the model
- pytorch_model.bin : a PyTorch dump of a trained BERT model (saved with the usual torch.save())

CodaLab requires us that the prediction python script should run with the following arguments. Therefore, if your python script does not run with the following arguments, the script needs to be modified.

```
CodaLab> python <path-to-prediction-python-script> <input-data-json-file> <output-prediction-json-file>
```

If you use [`run_squad.py`](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_squad.py) code from [huggingface/pytorch-pretrained-BERT repository](https://github.com/huggingface/pytorch-pretrained-BERT) for training, and have difficulty modifying this to the above format, just use `run_squad_for_submission.py` in this repository. It runs with the arguments required by CodaLab, and generates predictions. And it should be in the same place as the archive of a trained model (vocab.txt, config.json, pytorch_model.bin).

<br>

### 3. Run your trained model on the dev set
Copy the dev data to the worksheet by using the following command into the CodaLab terminal. Do not upload the dev data directly!

```
CodaLab> cl add bundle korquad-data//KorQuAD_v1.0_dev.json .
```

Then run the command for the python script generates the predictions on the dev data. Be sure to replace `<path-to-prediction-python-script>` with your actual program path.
```
CodaLab> cl run :KorQuAD_v1.0_dev.json :<path-to-prediction-python-script> "python <path-to-prediction-python-script> KorQuAD_v1.0_dev.json predictions.json" -n run-predictions
```
#### Running jobs that use GPUs
If you run the following command on the default Docker image(codalab/ubuntu:1.9), the GPU is not available because the image does not support CUDA. This can be checked as follows.
```
CodaLab> cl run :check_cuda.py "python check_cuda.py"

# stdout>
# ('Python Version: ', '2.7.12')
# ('PyTorch Version: ', '0.4.1')
# ('CUDA Available: ', False)
# ('Device Count: ', 0)
```

To enable GPU acceleration, we need to include the `--request-gpus` flag like belows.
```
CodaLab> cl run :check_cuda.py "python check_cuda.py" --request-gpus 1

# stdout>
# ('Python Version: ', '2.7.12')
# ('PyTorch Version: ', '1.0.1.post2')
# ('CUDA Available: ', True)
# ('Device Count: ', 1L)
# ('Current Device: ', 0L)
# ('Device Name', u'Tesla M60')
```

cf. You can also increase the GPU memory by using `--request-memory` flag, because default GPU memory size (2GB) is too small.
```
CodaLab> cl run :check_cuda.py "python check_cuda.py" --request-gpus 1 --request-memory 11g
```


Now, all jobs (e.g., prediction) can be conducted on a GPU.

#### Specifying Environments with Docker
CodaLab uses Docker containers to define the environment of a run bundle. Each Docker container is based on a Docker image, which specifies the full environment, including which Linux kernel version, which libraries, etc. The default Docekr image is `codalab/ubuntu:1.9`, which consists of Ubuntu 14.04 plus some standard packages (e.g., Python).

In CodaLab, when you create a run, you can specify which Docker container you want to use like below.
```
CodaLab> cl run <command> --request-docker-image <docker-image-to-use>
```

**Here we use [lyeoni/pytorch_pretrained_bert](https://cloud.docker.com/repository/docker/lyeoni/pytorch_pretrained_bert) docker image that contains PyTorch 1.0.0 (with CUDA 9.0) and pre-trained BERT models re-implemented by PyTorch.**

[lyeoni/pytorch_pretrained_bert](https://cloud.docker.com/repository/docker/lyeoni/pytorch_pretrained_bert) image is based on 2 repository, [anibali's PyTorch docker image](https://github.com/anibali/docker-pytorch) that contains various CUDA-enabled version of the PyTorch image, and [PyTorch Pretrained BERT](https://github.com/huggingface/pytorch-pretrained-BERT) that contains PyTorch reimplementations, pre-trained models for Google's BERT model.

lyeoni/pytorch_pretrained_bert docker image makes both CUDA-enabled PyTorch and pre-trained BERT model easy to use in CodaLab.

```
CodaLab> cl run :check_cuda.py "python check_cuda.py" --request-docker-image lyeoni/pytorch_pretrained_bert --request-gpus 1

# Python Version:  3.6.5
# PyTorch Version:  1.0.0
# CUDA Available:  True
# Device Count:  1
# Current Device:  0
# Device Name Tesla K80
```

**The final command to generate the predictions on the dev set is like belows.**
```
CodaLab> cl run :config.json :vocab.txt :pytorch_model.bin :KorQuAD_v1.0_dev.json :run_squad.py "python run_squad.py KorQuAD_v1.0_dev.json predictions.json" -n run-predictions --request-docker-image lyeoni/pytorch_pretrained_bert --request-gpus 1 --request-memory 11g
```

<br>

### 4. Evaluate the predictions on the dev set
If the bundle state is **ready** not failed, we extract out the predictions file into a bundle of its own. Let's do this as follows:
```
# MODELNAME should not contain spaces (avoid using special characters too).
CodaLab> cl make run-predictions/predictions.json -n predictions-{MODELNAME}
```

Now, let's verify that we can evaluate the predictions on the dev set.
```
CodaLab> cl macro korquad-utils/dev-evaluate-v1.0 predictions-{MODELNAME}
```

Once this succeeds, you should see the scores for your model appended to the worksheet.

<br>

### 5. Submission
Follow the submission guide provided in the official site. This step requires only a formatted descriptions(e.g., model name, your name, institution, etc.) to add an official leaderboard.

- [KorQuAD Submission Guide (English Ver.)](https://worksheets.codalab.org/worksheets/0xee93409b823f436d88741ab115403559/)
- [KorQuAD Submission Guide (Korean Ver.)](https://worksheets.codalab.org/worksheets/0x7b06f2ebd0584748a3a281018e7d19b0/)

## Appendix
- In CodaLab, each run has a state, which evolves through the following values:
  - `created` : initial state
  - `staged` : for run bundles, meaning dependencies are ready
  - `preparing` : launch a worker just for this run, waiting for it
  - `running` : a worker is running the command
  - `ready/failed` : terminal states corresponding to a successful or unsuccessful run

## References
- [LG CNS AI Research Team] [KorQuAD, The Korean Question Answering Dataset](https://korquad.github.io/)
- [LG CNS AI Research Team] [KorQuAD Submission Guide (English Ver.)](https://worksheets.codalab.org/worksheets/0xee93409b823f436d88741ab115403559/)
- [LG CNS AI Research Team] [KorQuAD Submission Guide (Korean Ver.)](https://worksheets.codalab.org/worksheets/0x7b06f2ebd0584748a3a281018e7d19b0/)
- [CodaLab] [CodaLab, A collaborative platform for reproducible research](https://worksheets.codalab.org/)
- [codalab/codalab-worksheets] [Execution](https://github.com/codalab/codalab-worksheets/wiki/Execution)
- [codalab/codalab-worksheets] [CLI Reference](https://github.com/codalab/codalab-worksheets/wiki/CLI-Reference)
- [anibali/docker-pytorch] [PyTorch Docker image](https://github.com/anibali/docker-pytorch)
- [huggingface/pytorch-pretrained-BERT] [PyTorch Pretrained BERT: The Big & Extending Repository of pretrained Transformers](https://github.com/huggingface/pytorch-pretrained-BERT)
- [huggingface/pytorch-pretrained-BERT] [Fixing issue "Training beyond specified 't_total' steps with schedule 'warmup_linear'" reported in #556 #604](https://github.com/huggingface/pytorch-pretrained-BERT/pull/604)
- [lyeoni/pytorch_pretrained_bert] [Docker image with pyotrch and pretrained_bert](https://cloud.docker.com/repository/docker/lyeoni/pytorch_pretrained_bert)
- [pbaumgartner/pytorch-bert] [pytorch-bert-image](https://hub.docker.com/r/pbaumgartner/pytorch-bert/dockerfile)
