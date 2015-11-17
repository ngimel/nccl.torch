# nccl.torch
Torch7 FFI bindings for NVidia NCCL library.

# Installation
 - Install NCCL from https://github.com/NVIDIA/nccl
 - Have at least Cuda 7.0
 - Have libnccl.so in your library path 

# Collective operations supported
 - allReduce
 - reduce
 - broadcast
 - allGather

# Example usage
Argument to the collective call should be a table of contiguous tensors located on the different devices. 
Example: perform in-place allReduce on the table of tensors:

```lua
require 'nccl'
nccl.allReduce(inputs)
```
where inputs is a table of contiguous tensors of the same size located on the different devices.



