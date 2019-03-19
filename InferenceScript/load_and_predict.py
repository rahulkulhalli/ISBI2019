import os
from keras import models, preprocessing, applications
import keras.backend as K
import tensorflow as tf
import pandas as pd
from uuid import uuid1


def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0., 1.)))
    _precision = true_positives / (predicted_positives + K.epsilon())
    possible_positives = K.sum(K.round(K.clip(y_true, 0., 1.)))
    _recall = true_positives / (possible_positives + K.epsilon())
    return 2 * (_precision * _recall) / (_precision + _recall + K.epsilon())


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0., 1.)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0., 1.)))
    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0., 1.)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0., 1.)))
    return true_positives / (possible_positives + K.epsilon())


def weighted_f1(y_true, y_pred):
    y_pred_bin = y_pred > 0.5
    y_true_inverse = K.cast(tf.math.logical_not(y_true > 0.5), K.floatx())
    y_pred_bin_inverse = K.cast(tf.math.logical_not(y_pred_bin), K.floatx())

    y_pred_bin = K.cast(y_pred_bin, K.floatx())

    y_true = K.cast((y_true > 0.5), K.floatx())

    tp_1 = K.sum(y_true * y_pred_bin)
    tp_0 = K.sum(y_true_inverse * y_pred_bin_inverse)

    tp_sum = tf.stack([tp_0, tp_1])

    pred_sum_0 = K.sum(
        K.cast(K.equal(y_pred_bin, 0.), K.floatx())
    )
    pred_sum_1 = K.sum(
        K.cast(K.equal(y_pred_bin, 1.), K.floatx())
    )
    pred_sum = tf.stack([pred_sum_0, pred_sum_1])

    true_sum_0 = K.sum(
        K.cast(K.equal(y_true, 0.), K.floatx())
    )
    true_sum_1 = K.sum(
        K.cast(K.equal(y_true, 1.), K.floatx())
    )
    true_sum = tf.stack([true_sum_0, true_sum_1])

    _precision = tp_sum / (pred_sum + K.epsilon())

    _recall = tp_sum / (true_sum + K.epsilon())

    f1 = (2 * _precision * _recall) / (_precision + _recall + K.epsilon())

    # Return Weighted Average
    return K.sum(f1 * true_sum) / (K.sum(true_sum) + K.epsilon())


def get_custom_metrics():
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weighted_f1": weighted_f1
    }


def get_random_str(str_length):
    return str(uuid1()).replace("-", "_")[:str_length]


def load_model(model_path):
    if not os.path.exists(model_path):
        raise IOError("No model object found at {}".format(model_path))

    try:
        print(50*'-')
        print("Loading model. This might take several minutes.")
        metrics = get_custom_metrics()

        gpu_model = models.load_model(model_path,
                                      compile=True,
                                      custom_objects=metrics)

        model = gpu_model.layers[-2]

        print("Model loaded.")
        print(50 * '-')
        return model
    except Exception as e:
        print("Something went wrong while loading the model.")


def predict_from_data(model, data_path):

    batch_size = 8

    if not os.path.exists(data_path):
        raise IOError("Could not locate data.")

    # Instantiate a DirectoryIterator instance.
    gen = preprocessing.image.ImageDataGenerator(preprocessing_function=applications
                                                 .resnext
                                                 .preprocess_input)

    dir_iterator = gen.flow_from_directory(data_path,
                                           target_size=(350, 350),
                                           shuffle=False,
                                           classes=None,
                                           class_mode=None,
                                           batch_size=batch_size)

    steps = dir_iterator.samples//batch_size + (dir_iterator.samples % batch_size != 0)*1

    preds = model.predict_generator(dir_iterator, verbose=1, steps=steps)

    print("Predictions done.")

    # Remove last axis.
    return dir_iterator.filenames, preds.flatten()


def prepare_data(file_names, preds, threshold=0.5):
    csv_name = get_random_str(10) + "Output.csv"

    print("Saving to {}".format(csv_name))

    df = pd.DataFrame({
        "filenames": file_names,
        "preds": preds
    })

    # Convert all predictions to 0-1 format.
    df["preds"] = (df["preds"] > threshold)*1

    # Apply the lambda function to clean-up file names.
    # test/1234.jpg -> 1234.jpg -> 1234 -> int(1234)
    try:
        df["filenames"] = df["filenames"].apply(lambda x: int(x.split("/")[1].split(".")[0]))
    except Exception:
        print("Something went wrong while cleaning-up file names.")

    # Sort in ascending order.
    df.sort_values(by="filenames", ascending=True, axis=0, inplace=True)

    df["preds"].to_csv(os.path.join(os.getcwd(), "output", csv_name),
                       header=None,
                       index=False)

    print("Dumped output. Script finished successfully.")
    print(50*'-')
