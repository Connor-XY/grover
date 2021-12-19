from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print_tensors_in_checkpoint_file(file_name='gs://grover-models/discrimination/generator=base~discriminator=grover~discsize=base~dataset=p=0.96/model.ckpt-1562')

from tensorflow.python import pywrap_tensorflow
import os

checkpoint_path = 'gs://grover-models/discrimination/generator=base~discriminator=grover~discsize=base~dataset=p=0.96/model.ckpt-1562'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key)) # Remove this is you want to print only variable names