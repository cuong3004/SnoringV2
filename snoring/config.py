import os
import warnings
import tensorflow as tf
import jax 
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Seems like the new one will break jax
if 'TPU_NAME' in os.environ and 'KAGGLE_DATA_PROXY_TOKEN' in os.environ:
    use_tpu = True
    
    import requests 
    from jax.config import config
    if 'TPU_DRIVER_MODE' not in globals():
        url = 'http:' + os.environ['TPU_NAME'].split(':')[1] + ':8475/requestversion/tpu_driver_nightly'
        resp = requests.post(url)
        TPU_DRIVER_MODE = 1
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
    # Enforce bfloat16 multiplication
    config.update('jax_default_matmul_precision', 'bfloat16')
    print('Registered (Kaggle) TPU:', config.FLAGS.jax_backend_target)
else:
    use_tpu = False
# /////////////////////////////////////////

args = {
    'experiment_name': 'starter',
    # Efficientnetv2-m
    'model': 'm',
    'batch_size': 8, 
    'epochs': 16,
    'base_lr': 7e-5, # should directly correspond to `batch_size`
    
    # Data / augmentation 
    'img_size': 512, # 192, 224, 331, 512
    # The actual label number is 104
    'num_labels': 104 if not use_tpu else 128,
    'saving_dir': '/kaggle/working/',
    'device_count' : jax.device_count(),
    
    # Debugging-purposes
    'sanity_check': False,
}

# The effective lr should linearly scale with batch size
args['lr'] = args['base_lr'] * args['device_count']
if args['sanity_check']:
    args['epochs'] = 1

# Data-specific, should change for each dataset 
args['data_dir'] = '/kaggle/input/tpu-getting-started/tfrecords-jpeg-'\
                    + str(args['img_size'])+'x'+str(args['img_size'])
args['train_dir'] = os.path.join(args['data_dir'], 'train')
args['val_dir'] = os.path.join(args['data_dir'], 'val')
args['test_dir'] = os.path.join(args['data_dir'], 'test')

print('Running on', args['device_count'], 'processors')
print(jax.devices())
