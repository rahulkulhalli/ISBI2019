# Inference Pipeline for our ISBI approach

## 1. Python version
It is to be noted that our model is compatible ONLY with <b>Python 3.5.3</b>. Using any other Python version will most likely result in a TensorFlow opcode error. The quickest workaround you can use to get this script running is by creating a `virtualenv`.

If you have <a href="https://www.anaconda.com/distribution/">conda</a>, issue the following commands to create a virtualenv with the name 'test' and a Python 3.5.3 interpreter.

`conda create -n test python==3.5.3`

To activate the env,

`conda activate test`

To deactivate,

`conda deactivate test`

## 2. Running the pipeline
After activating your environment (if you need to), navigate to the `assets` folder and run the following command <b>ONLY ONCE</b>. This will install all the necessary requirements for running the script. This may take several minutes.

```python
pip install -r requirements.txt
```

Next, run the `main.py` file with exactly one command line argument: the <b>absolute path</b> to your image directory.

`python main.py /absolute/path/to/images`

If any folder in your hierarchy contains a whitespace, please provide the path as a string.

`python main.py "/absolute/some path/to/images"`

## 3. Temporary directory creation
The <a href="https://keras.io/">Keras</a> `DirectoryIterator` object requires that the images be in the following order:

```
    /image_dir
      |
      |----Class A
             |
             |----A.1
             |----A.2
             .
             .
      |----Class B
             |
             |----B.1
             |----B.2
             .
             .
      |----Class C
             |
             .
             .
```

In the case of a test `DirectoryIterator` object, the folder structure would look like:

```
    /image_dir
      |
      |--test_folder
          |
          |----image1
          |----image2
          .
          .
          |----imageN
```

Our code will check whether you have a nested directory structure. If you don't, the script will create a temporary structure and <b>copy</b> all the images into that nested folder. After the predictions are generated, we delete the temporary directory as well.

## 4. Collecting the Output
Every time the script is run, a new CSV file with a 10-digit UUID code will be generated and placed in the `Output` directory. The script will tell the user the name of the generated file. 


## 5. Output format
It is to be noted that the output CSV format is the same as the submission format used during the competition. There is only one column (without a header) denoting the predicted label (1 for ALL, 0 for HEM). The CSV file is sorted in the ascending order of image names. A code snippet for how this was achieved:

```python
# test/100.jpg -> 100.jpg -> 100 -> int(100)
df["preds"] = df["preds"].apply(lambda x: int(x.split("/")[1].split(".")[0]))
    
df.sort_values(by="preds", axis=0, inplace=True, ascending=True)
```

## Note
The script does not check for GPU configuration and will not run on GPU(s) by default. If you'd like to perform inferences using your GPU(s), please add the following snippet right after __line 96__ in `load_and_predict.py` (If you're using multi-GPU support, it would also be a good idea to increase the `batch_size` parameter on __line 107__; just make sure that it is a multiple of the number of GPUs you have.):
 
 ```python
 
# Add the utils package.
from keras import utils

# num_gpus is the number of GPUs visible to TensorFlow.
model = utils.multi_gpu_model(model, gpus=num_gpus)
 ```
