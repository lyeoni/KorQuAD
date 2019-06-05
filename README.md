# KorQuAD
QA system for KorQuAD (Korean Question Answering Dataset) using BERT
 
## Submission

To get official scores on the KorQuAD test set, we must submit(or upload) our model to the [CodaLab](https://worksheets.codalab.org/). This is because the integrity of test results should be preserved.

Here's a detailed process that guides you through the official evaluation of our model.

### 0. Create a CodaLab account
You can create a CodaLab account [here](https://worksheets.codalab.org/).
Click `sign up` in the top-right corner of the CodaLab homepage.

<p align="center">
<img src="https://github.com/lyeoni/KorQuAD/blob/master/images/codalab_signup.png" />
</p>

### 1. Create a CodaLab worksheet
Click the `My dashboard` in the top-right corner.
<p align="center">
<img src="https://github.com/lyeoni/KorQuAD/blob/master/images/codalab_dashboard.png" />
</p>

Click the `New Worksheet` in the upper-right corner and name your worksheet.
<p align="center">
<img src="https://github.com/lyeoni/KorQuAD/blob/master/images/codalab_worksheet.png" />
</p>

### 2. Upload trained model with source code
CodaLab requires us that the prediction python script should run with the following arguments. Therefore, if your python script does not run with the following arguments, the script needs to be modified.

```
CodaLab> python <path-to-prediction-python-script> <input-data-json-file> <output-prediction-json-file>
```

### 3. Run your trained model on the dev set
Copy the dev data to the worksheet by using the following command into the CodaLab terminal. Do not upload the dev data directly!

```
CodaLab> cl add bundle korquad-data//KorQuAD_v1.0_dev.json .
```

Then run the command for the python script generates the predictions. Be sure to replace `<path-to-prediction-python-script>` with your actual program path.
```
CodaLab> cl run :KorQuAD_v1.0_dev.json :<path-to-prediction-python-script> "python <path-to-prediction-python-script> KorQuAD_v1.0_dev.json predictions.json" -n run-predictions
```
#### Specifying Environments with Docker
CodaLab uses Docker containers to define the environment of a run bundle. Each Docker container is based on a Docker image, which specifies the full environment, including which Linux kernel version, which libraries, etc.

The default Docekr image is `codalab/ubuntu:1.9`, which consists of Ubuntu 14.04 plus some standard packages (e.g., Python).

In CodaLab, when you create a run, you can specify which Docker container you want to use like below.
```
CodaLab> cl run <command> --request-docker-image <docker-image-to-use>
```

Here we use [lyeoni/pytorch_pretrained_bert](https://cloud.docker.com/repository/docker/lyeoni/pytorch_pretrained_bert) image that includes PyTorch and [PyTorch Pretrained BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

```
CodaLab> cl run <command> --request-docker-image lyeoni/pytorch_pretrained_bert
```

#### Running jobs that use GPUs
If you run the following command on the default Docker image(codalab/ubuntu:1.9), the GPU is not available because the image does not support CUDA.
```
CodaLab> cl run :check_cuda.py "python check_cuda.py"

# stdout>
# ('Python Version: ', '2.7.12')
# ('PyTorch Version: ', '0.4.1')
# ('CUDA Available: ', False)
# ('Device Count: ', 0)
```

That's why you need to include the `--request-gpus` flag.
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


## References
- [LG CNS AI Research Team] [KorQuAD, The Korean Question Answering Dataset](https://korquad.github.io/)
- [LG CNS AI Research Team] [KorQuAD Submission Guide (English Ver.)](https://worksheets.codalab.org/worksheets/0xee93409b823f436d88741ab115403559/)
- [huggingface/pytorch-pretrained-BERT] [PyTorch Pretrained BERT: The Big & Extending Repository of pretrained Transformers](https://github.com/huggingface/pytorch-pretrained-BERT)
- [CodaLab] [CodaLab, A collaborative platform for reproducible research](https://worksheets.codalab.org/)
- [codalab/codalab-worksheets] [Execution](https://github.com/codalab/codalab-worksheets/wiki/Execution)
- [lyeoni/pytorch_pretrained_bert] [Docker image with pyotrch and pretrained_bert](https://cloud.docker.com/repository/docker/lyeoni/pytorch_pretrained_bert)