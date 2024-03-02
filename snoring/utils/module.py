from dataclasses import field
from snoring.utils.common import HyperParameters
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from jax import numpy as jnp
import jax
from flax.training.common_utils import shard

class DataModule(HyperParameters):
    """The base class of data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, root='../data'):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        # Use Tensorflow Datasets & Dataloader. JAX or Flax do not provide
        # any dataloading functionality
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))


class Module(nn.Module, HyperParameters):
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""
    # No need for save_hyperparam when using Python dataclass
    plot_train_per_epoch: int = field(default=2, init=False)
    plot_valid_per_epoch: int = field(default=1, init=False)
    # Use default_factory to make sure new plots are generated on each run

    def forward(self, X, *args, **kwargs):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X, *args, **kwargs)

    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        
        self.board.draw(x, value,
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, params, batch, state):
        raise NotImplementedError

    def validation_step(self, params, batch, state):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def apply_init(self, dummy_input, key):
        """Defined in :numref:`sec_lazy_init`"""
        params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
        return params
    




class FashionMNIST(DataModule):
    """The Fashion-MNIST dataset.

    Defined in :numref:`sec_fashion_mnist`"""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()

    def text_labels(self, indices):
        """Return text labels.
    
        Defined in :numref:`sec_fashion_mnist`"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        """Defined in :numref:`sec_fashion_mnist`"""
        data = self.train if train else self.val
        process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                                tf.cast(y, dtype='int32'))
        resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
        shuffle_buf = len(data[0]) if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
        

class AudiosetModule(DataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        
        input_dtype = jnp.bfloat16 if args.use_tpu else jnp.float32
        label_dtype = jnp.int16
        self.depth = 527
        
        train_filenames = tf.io.gfile.glob(args['train_dir']+'/*.tfrec')
        val_filenames = tf.io.gfile.glob(args['val_dir']+'/*.tfrec')
        test_filenames = tf.io.gfile.glob(args['test_dir']+'/*.tfrec')
    
    def decode_audio(self, audio_data):
        # image is of type `tf.uint8`
        audio, sr = tf.audio.decode_wav(audio_data["audio"], desired_channels=1)
        # image = tf.image.decode_jpeg(image_data, channels=3)
        # image = tf.reshape(image, [args['img_size'], args['img_size'], 3]) 
        return audio, sr
    
    def read_labeled_tfrecord(self, example):
        # Note how we are defining the example structure here
        labeled_format = {
                'audio': tf.io.FixedLenFeature([], tf.string),
                'ytid': tf.io.FixedLenFeature([], tf.string),
                'labels': tf.io.VarLenFeature(tf.int64),
        }
        parsed_example = tf.io.parse_single_example(example, labeled_format)
        audio, sr = self.decode_audio(parsed_example['audio'])
        
        labels = tf.sparse.to_dense(example['labels'])
        labels = tf.one_hot(labels, self.depth)
        labels = tf.reduce_max(labels, axis=0)
        
        return {'audio': audio, 'sr':sr, 'label': labels} 
    
    def to_jax(self, sample):
        sample['image'] = jnp.array(sample['image'], dtype=self.input_dtype)
        sample['label'] = jnp.array(sample['label'], dtype=self.label_dtype)
        # Convert labels to one_hot
        sample['label'] = jax.nn.one_hot(sample['label'], self.args['num_labels'], dtype=self.label_dtype, axis=-1)
        
        return sample
    
    def load_dataset(self, filenames, ordered=False, shuffle_buffer_size=1, drop_remainder=False,\
                augment=True):
        AUTO = tf.data.experimental.AUTOTUNE
        # tf.data runtime will optimize this parameter
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

        options = tf.data.Options()
        if not ordered:
            options.experimental_deterministic = False
            
        # Step 1: Read in the data, shuffle and batching
        dataset = dataset.with_options(options)\
                    .map(self.read_labeled_tfrecord,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                    .shuffle(shuffle_buffer_size)\
                    .batch(self.args['batch_size'] * self.args['device_count'], drop_remainder=drop_remainder)
        
        # We exemplify augmentation using RandAugment
        # if augment:
            # dataset = dataset.map(tf_randaugment, num_parallel_calls=AUTO)
        # Add `prefetch` at the last step to parallize as much as possible!
        # dataset = dataset.map(normalize_and_resize).prefetch(AUTO)
        # Finally, apply to_jax transformation
        return map(self.to_jax, tfds.as_numpy(dataset))