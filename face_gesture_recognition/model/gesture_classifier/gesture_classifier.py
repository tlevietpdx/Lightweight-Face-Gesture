#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class FaceGestureClassifier():
    def __init__(
        self,
        model_path='model/gesture_classifier/gesture_classifier.hdf5'
    ):
        self.model = tf.keras.models.load_model(model_path)

    def __call__(
        self,
        point_history,
    ):
        predict_result = self.model.predict(np.array([point_history]))
        # print(np.squeeze(predict_result))

        return np.argmax(np.squeeze(predict_result))