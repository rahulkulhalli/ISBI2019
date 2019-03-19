# Inference Pipeline for our ISBI approach

## 1. Python version
It is to be noted that our model is compatible ONLY with <b>Python 3.5.3</b>. Using any other Python version will most likely result in an opcode error. The quickest workaround is by creating a `virtualenv`.

If you have <a href="https://www.anaconda.com/distribution/">Conda</a>, issue the following commands to create a virtualenv with the name 'test' and a Python 3.5.3 interpreter.

`conda create -n test python==3.5.3`

To activate the env, use the following command:

`conda activate test`

To deactivate,

`conda deactivate test`

## 2. Running the pipeline
After activating your environment (if you need to), navigate to the `assets` folder and run the following command <b>ONLY ONCE</b>. This will install all the necessary requirements for running the script. This may take several minutes.

`pip install -r requirements.txt`

Next, run the `main.py` file with exactly one command line argument: the <b>absolute path</b> to your images.

`python main.py /absolute/path/to/images`

If any folder in your hierarchy contains a whitespace, please provide the path as a string.

`python main.py "/absolute/some path/to/images"`

## 3. Collecting the Output
Every time the script is run, a new CSV file with a 10-digit UUID code will be generated and placed in the `Output` directory. The script will tell the user the name of the generated file. 


## 4. Output format
It is to be noted that the output CSV format is the same as the submission format used during the competition. There is only one column (without a header) denoting the predicted label (1 for ALL, 0 for HEM). The CSV file is sorted in the ascending order of image names. A code snippet for how this was achieved:

```python
# test/100.jpg -> 100.jpg -> 100 -> int(100)
df["preds"] = df["preds"].apply(lambda x: int(x.split("/")[1].split(".")[0]))
    
df.sort_values(by="preds", axis=0, inplace=True, ascending=True)
```
