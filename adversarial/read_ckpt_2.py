import tensorflow as tf
import tensorflow.compat.v1 as tf1

def print_checkpoint(save_path):
    reader = tf.train.load_checkpoint(save_path)
    shapes = reader.get_variable_to_shape_map()
    dtypes = reader.get_variable_to_dtype_map()
    print(f"Checkpoint at '{save_path}':")
    for key in shapes:
        print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
              f"value={reader.get_tensor(key)})")