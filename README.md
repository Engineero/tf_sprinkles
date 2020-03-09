# TF Sprinkles
Sprinkles augmentation implemented in TensorFlow.

Based on Less Wright's Medium article, [Progessive Sprinkles: a New Data
Augmentation for CNNs][0]. See also his [post on fast.ai][1].

To install:

    pip install tf_sprinkles

To use:

```python
from tf_sprinkles import Sprinkles
sprinkles = Sprinkles(num_holes, side_length)
```
  
Then call `sprinkles(image)` in the input pipeline for your image. A simple
example to get started using the `cat.jpeg` image located in the data folder
is:

```python
import numpy as np
import tensorflow as tf
from tf_sprinkles import Sprinkles
from PIL import Image
from matplotlib import pyplot as plt

sprinkles = Sprinkles(num_holes=100, side_length=10)
img = Image.open('data/cat.jpeg')
img = np.asarray(img) / 255.
result = sprinkles(tf.constant(img, dtype=tf.float32))
plt.imshow(result.numpy())
```

Which results in the following image with sprinkles.

![cat with sprinkles][2]

Note that the `mode` flag added in version 1.1.0 can be used to specify that
sprinkles should be filled with Gaussian noise (`mode='gaussian'`), randomly
filled with black or white (`mode='salt_pepper'`), or all black (the default
or `mode=None`).

[0]: https://medium.com/@lessw/progressive-sprinkles-a-new-data-augmentation-for-cnns-and-helps-achieve-new-98-nih-malaria-6056965f671a
[1]: https://forums.fast.ai/t/progressive-sprinkles-cutout-variation-my-new-data-augmentation-98-on-nih-malaria-dataset/50454
[2]: https://github.com/Engineero/tf_sprinkles/blob/develop/test/data/cat_sprinkled.png

