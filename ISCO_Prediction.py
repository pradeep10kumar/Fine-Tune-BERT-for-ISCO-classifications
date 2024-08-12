# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:34:42 2024

@author: Pradeep Kumar

"""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
import pandas as pd
from tensorflow.keras.models import load_model
from official.nlp.data import classifier_data_lib
from official.nlp.tools import tokenization
import joblib

model = load_model('best_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})


vocab_file = model.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = model.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file,do_lower_case)

# Parameters
max_seq_length = 128
label_list = 424
dummy_label = 100


# Define a function to preprocess the new data
def get_feature_new(text, max_seq_length, tokenizer, dummy_label):
    example = classifier_data_lib.InputExample(guid=None,
                                               text_a=text.numpy().decode('utf-8'),
                                               text_b=None,
                                               label=dummy_label)  # Use a valid dummy label
    feature = classifier_data_lib.convert_single_example(0, example, label_list, max_seq_length, tokenizer)
    return feature.input_ids, feature.input_mask, feature.segment_ids

def get_feature_map_new(text):
    input_ids, input_mask, segment_ids = tf.py_function(
        lambda text: get_feature_new(text, max_seq_length, tokenizer, dummy_label),
        inp=[text],
        Tout=[tf.int32, tf.int32, tf.int32]
    )
    input_ids.set_shape([max_seq_length])
    input_mask.set_shape([max_seq_length])
    segment_ids.set_shape([max_seq_length])
    
    x = {'input_word_ids': input_ids,
         'input_mask': input_mask,
         'input_type_ids': segment_ids}
    
    return x

def preprocess_new_data(texts):
    dataset = tf.data.Dataset.from_tensor_slices((texts,))
    dataset = dataset.map(get_feature_map_new,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(32, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

data = pd.read_csv('data.csv')


label_encoder = joblib.load('label_encoder.joblib')


# Preprocess the new data
sample_example = data['text'].to_list()
new_data_dataset =  preprocess_new_data(sample_example)
# Make predictions on the new data
predictions = model.predict(new_data_dataset)

# Decode the predictions
predicted_classes = [label_list[np.argmax(pred)] for pred in predictions]

print(predicted_classes)
highest_probabilities = [max(instance) for instance in predictions]
decoded_labels = label_encoder.inverse_transform(predicted_classes)

data['prob'] = highest_probabilities
data['predicted_isco'] = predicted_classes

data['target_isco'] =label_encoder.inverse_transform(data.target)
data['predicted_isco_decoded'] =label_encoder.inverse_transform(data.predicted_isco)