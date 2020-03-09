#*******************************************************************************
# Filename: test_spinkles.py
# Language: Python
# Author: nathantoner
# Created: 2020-03-09
#
# Description:
# Unit tests for Sprinkles implementation.
#
#*******************************************************************************
import unittest
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from ..tf_sprinkles import Sprinkles


class TestSprinkles(unittest.TestCase):
    """Tests the Sprinkles class of the tf_sprinkles module."""

    def setUp(self):
        self.n = 100
        self.length = 10

    def test_init(self):
        sprinkles = Sprinkles(self.n, self.length)
        self.assertEqual(sprinkles.n, self.n,
                         'num_holes not initialized correctly!')
        self.assertEqual(sprinkles.length, self.length,
                         'side_length not initialized correctly!')
        sprinkles = Sprinkles(self.n, self.length, mode='salt_pepper')
        self.assertEqual(
            sprinkles.n, self.n // 2,
            'num_holes not initialized correctly for salt & pepper!'
        )

    def test_call(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sprinkles = Sprinkles(self.n, self.length)
        img = Image.open(os.path.join(dir_path, 'data', 'cat.jpeg'))
        img = np.asarray(img) / 255.
        result = sprinkles(tf.constant(img, dtype=tf.float32))
        result = result.numpy()
        self.assertIsInstance(result, np.ndarray,
                              'result.numpy() of unexpected type!')
        sprinkles = Sprinkles(self.n, self.length, mode='gaussian')
        result = sprinkles(tf.constant(img, dtype=tf.float32))
        result = result.numpy()
        self.assertIsInstance(result, np.ndarray,
                              'result.numpy() of unexpected type!')
        sprinkles = Sprinkles(self.n, self.length, mode='salt_pepper')
        result = sprinkles(tf.constant(img, dtype=tf.float32))
        result = result.numpy()
        self.assertIsInstance(result, np.ndarray,
                              'result.numpy() of unexpected type!')
        sprinkles = Sprinkles(self.n, self.length, mode='other')
        with self.assertRaisesRegex(ValueError, 'Unknown mode',
                                    msg='ValueError not raised!'):
            result = sprinkles(tf.constant(img, dtype=tf.float32))



#*******************************************************************************
#                                END OF FILE
#*******************************************************************************
