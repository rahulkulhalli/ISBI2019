import argparse
import os
from tqdm import tqdm
import shutil
from download_model import download_model
from load_and_predict import load_model, predict_from_data


def parse_arguments():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("path",
                            type=str,
                            help="Path to images. Please specify an absolute path here.")
        return parser.parse_args()
    except Exception as e:
        print("Something went wrong while parsing the command line argument")


def preprocess_input_directory(path_to_images):

    created = False

    if not os.path.exists(path_to_images):
        raise IOError("No such file or Directory: {}".format(path_to_images))

    try:
        files = os.listdir(path_to_images)

        print(50*'-')

        # If there is no single folder in the directory...
        if len(files) > 0 and not os.path.isdir(os.path.join(path_to_images, files[0])):
            print("Nested folder not found. Creating temporary folder structure.")

            # If you're getting a 'permission denied' message, please add the mode flag to this method.
            os.makedirs(os.path.join(path_to_images, "test"))

            created = True

            print("Copying images to nested folder.")

            for ix in tqdm(range(len(files))):
                shutil.copy(src=os.path.join(path_to_images, files[ix]),
                            dst=os.path.join(path_to_images, "test", files[ix]))

            print("Copied images to temporary nested folder.")
            print(50*'-')

        else:
            print("Valid directory structure.")
            print(50 * '-')

    except IOError:
        print("Something went wrong while setting up the input directory.")
        print(50 * '-')

    return created


def cleanup(path_to_images):
    test_dir = os.path.join(path_to_images, "test")
    if os.path.exists(test_dir) and os.path.isdir(test_dir):
        print(50*'-')
        print("Removing temporary directory")

        try:
            shutil.rmtree(test_dir)
        except IOError:
            print("There was an error while deleting temporary folder.")


if __name__ == "__main__":
    args = parse_arguments()
    IP_PATH = args.path

    is_dir_created = preprocess_input_directory(IP_PATH)

    model_path = download_model()
    cpu_model = load_model(model_path)
    file_names, predictions = predict_from_data(cpu_model, IP_PATH)

    if is_dir_created:
        cleanup(IP_PATH)