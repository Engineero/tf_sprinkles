"""Progressive Sprinkles implementation in TensorFlow.

Original code based on Stack Overflow question by Saquib Shamsi.

https://stackoverflow.com/questions/60567071/efficient-progressive-sprinkles-augmentation-in-tensorflow
"""


import tensorflow as tf


class Cutout:
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
        img_shape = tf.shape(image)
        i = tf.range(img_shape[0])
        j = tf.range(img_shape[1])
        masking_fn = Cutout._mask_out(image, img_shape, i, j, self.length)
        idx = tf.constant(0, dtype=tf.int32)
        image, idx = tf.while_loop(
            cond=lambda x, ii: tf.less(ii, self.n),
            body=masking_fn,
            loop_vars=[image, idx]
        )
        return image

    @staticmethod
    def _mask_out(image, img_shape, row_range, col_range, hole_length):
        """Masks rows and columns to be replaced."""
        shape = tf.shape(image)
        rows = shape[0]
        cols = shape[1]
        channels = shape[2]

        def _create_hole(image, idx):
            # Do the masking.
            r = tf.random_uniform([], minval=0, maxval=rows, dtype=tf.int32)
            c = tf.random_uniform([], minval=0, maxval=cols, dtype=tf.int32)

            r1 = tf.clip_by_value(r - hole_length // 2, 0, rows)
            r2 = tf.clip_by_value(r + hole_length // 2, 0, rows)
            c1 = tf.clip_by_value(c - hole_length // 2, 0, cols)
            c2 = tf.clip_by_value(c + hole_length // 2, 0, cols)

            row_mask = (r1 <= row_range) & (row_range < r2)
            col_mask = (c1 <= col_range) & (col_range < c2)
            zeros = tf.zeros(shape)

            # Full mask of replaced elements
            mask = row_mask[:, tf.newaxis] & col_mask

            # Select elements from flattened arrays
            img_flat = tf.reshape(image, [-1, channels])
            zeros_flat = tf.reshape(zeros, [-1, channels])
            mask_flat = tf.reshape(mask, [-1])
            result_flat = tf.where(mask_flat, zeros_flat, img_flat)

            # Reshape back
            result = tf.reshape(result_flat, img_shape)
            return [result, idx + 1]
        return _create_hole
