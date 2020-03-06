"""Progressive Sprinkles implementation in TensorFlow.

Original code based on Stack Overflow question by Saquib Shamsi.

https://stackoverflow.com/questions/60567071/efficient-progressive-sprinkles-augmentation-in-tensorflow
"""


import tensorflow as tf


class Sprinkles:
    """Progressive Sprinkles Agumentation.

    Args:
        num_holes: number of holes to make in an image
        side_length: lenght of sides each hole will have.

    Returns:
        Image with number of holes of specified size cut out.
    """

    def __init__(self, num_holes, side_length):
        self.n = num_holes
        self.length = side_length

    def __call__(self, image):
        tf.cast(image, tf.float32)
        zeros = tf.zeros_like(image)
        img_shape = tf.shape(image)
        rows = img_shape[0]
        cols = img_shape[1]
        num_channels = img_shape[-1]
        row_range = tf.tile(tf.range(rows)[..., tf.newaxis], [1, self.n])
        col_range = tf.tile(tf.range(cols)[..., tf.newaxis], [1, self.n])
        r_idx = tf.random.uniform([self.n], minval=0, maxval=rows-1,
                                  dtype=tf.int32)
        c_idx = tf.random.uniform([self.n], minval=0, maxval=cols-1,
                                  dtype=tf.int32)
        r1 = tf.clip_by_value(r_idx - self.length // 2, 0, rows)
        r2 = tf.clip_by_value(r_idx + self.length // 2, 0, rows)
        c1 = tf.clip_by_value(c_idx - self.length // 2, 0, cols)
        c2 = tf.clip_by_value(c_idx + self.length // 2, 0, cols)
        row_mask = (row_range > r1) & (row_range < r2)
        col_mask = (col_range > c1) & (col_range < c2)

        # Combine masks into one layer and duplicate over channels.
        mask = row_mask[:, tf.newaxis] & col_mask
        mask = tf.reduce_any(mask, axis=-1)
        mask = mask[..., tf.newaxis]
        mask = tf.tile(mask, [1, 1, num_channels])

        filtered_image = tf.where(mask, zeros, image)
        return filtered_image
