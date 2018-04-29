# nccl.torch
Torch7 FFI bindings for NVidia NCCL library.

# Installation
 - Install NCCL
 - Have at least Cuda 7.0
 - Have libnccl.so.{1|2} (or libnccl.{1|2}.dylib) either in your library path or in `NCCL_PATH`

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
