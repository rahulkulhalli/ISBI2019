# Inference Pipeline for our ISBI approach

## 1. Python version
It is to be noted that our model is compatible ONLY <b>Python 3.6.4</b>. Using any other version of Python will most likely result in an opcode error. The quickest way to do this is by creating a `virtualenv`.

If you have Conda, issue the following commands to create a virtualenv with the name 'test' and with a Python 3.6.4 interpreter.

`conda create -n test python==3.6.4`

To activate the env, issue the following:

`conda activate test`

To deactivate,

`conda deactivate test`

## 2. Running the pipeline
After activating your environment (if you need to), run the `install_requirements.py` file <b>ONLY ONCE</b>. This will install all the necessary requirements for running the script. Run the file without any parameters.

`python install_requirements.py`

Next, run the `main.py` file with exactly one command line argument - the <b>absolute path</b> to your images.

`python main.py /absolute/path/to/images`

If your any folder in your hierarchy contains a whitespace, please provide the path as a string.

`python main.py "/absolute/path/to/images"`

## 3. Collecting the Output
Every time the script is run, a new CSV file with a 10-digit UUID code will be generated and placed in the `Output` directory. The script will tell the user the name of the file.
