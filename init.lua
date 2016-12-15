require 'cutorch'
local ffi = require 'ffi'

local nccl = {}
_G.nccl = nccl

nccl.C = require 'nccl.ffi'
nccl.communicators = {}

local function errcheck(name, ...)
   local res = nccl.C[name](...)
   if res ~= 'ncclSuccess' then
      local msg = ffi.string(nccl.C.ncclGetErrorString(res))
      collectgarbage('restart')
      error(msg .. ' (nccl.' .. name .. ')')
   end
   return res
end

function nccl.createCommunicators(devices)
   if type(devices) == 'number' then
      devices = torch.range(0, devices-1):int()
   end
   assert(torch.type(devices) == 'torch.IntTensor', 'argument type not supported')

   local nDevices = devices:nElement()
   local key = table.concat(devices:totable(), ',')

   if not nccl.communicators[key] then
      --create communicator and register its garbage collector
      local comm = ffi.new('ncclComm_t[?]', nDevices)
      errcheck('ncclCommInitAll', comm, nDevices, devices:data())
      ffi.gc(comm, function(c)
         for i=0,nDevices-1 do
            nccl.C.ncclCommDestroy(c[i])
         end
      end)
      nccl.communicators[key] = comm
   end

   return nccl.communicators[key]
end

--TODO - make sure order of the GPUs is checked in the communicator
--TODO allow to use empty or wrong size outputs, as long as they are on the correct GPU
--TODO check the sizes of all the tensors

local function getComm(inputs, outputs)
   local devices = torch.IntTensor(#inputs)
   local types = {}
   for i,v in ipairs(inputs) do
      local device = v:getDevice()
      if outputs then
         assert(outputs[i]:getDevice() == device, 'input and output not on same device')
      end
      devices[i] = device-1 --zero-based for cuda
      local inputType = v:type()
      if outputs then
         assert(inputType == outputs[i]:type(), 'input and output types differ')
      end

      if inputType == 'torch.CudaHalfTensor' then
         types[i] = 'ncclHalf'
      elseif inputType == 'torch.CudaDoubleTensor' then
         types[i] = 'ncclDouble'
      else
         types[i] = 'ncclFloat'
      end
   end

   local comms = nccl.createCommunicators(devices)
   return comms, devices, types
end

local function checkroot(root, ntensors)
  if root == nil then return 1 end
  assert(root >= 1 and root <= ntensors, 'invalid root: ' .. tostring(root))
  return root
end

local function cudaStream()
   return ffi.C.THCState_getCurrentStream(cutorch.getState())
end

local function synchronize(devices)
   for i = 1, devices:nElement() do
      cutorch.setDevice(devices[i]+1)
      cutorch.streamSynchronize(cutorch.getStream())
   end
end

function nccl.allReduce(inputs, outputs, async)
   local curDevice = cutorch.getDevice()
   local comm, devices, types = getComm(inputs, outputs)
   local count = inputs[1]:nElement()
   outputs = outputs or inputs
   collectgarbage('stop')
   for i=1,#inputs do
      cutorch.setDevice(devices[i]+1)
      errcheck('ncclAllReduce', inputs[i]:data(), outputs[i]:data(), count,
         types[i], 'ncclSum', comm[i-1], cudaStream())
   end
   collectgarbage('restart')
   if not async then synchronize(devices) end
   cutorch.setDevice(curDevice)
end

function nccl.reduce(inputs, outputs, async, root)
   local curDevice = cutorch.getDevice()
   local comm, devices, types = getComm(inputs, outputs)
   local count = inputs[1]:nElement()
   root = checkroot(root, #inputs)
   outputs = outputs or inputs
   collectgarbage('stop')
   for i=1,#inputs do
      cutorch.setDevice(devices[i]+1)
      local output = outputs[i] and outputs[i]:data() or nil
      errcheck('ncclReduce', inputs[i]:data(), output, count, types[i],
         'ncclSum', root-1, comm[i-1], cudaStream())
   end
   collectgarbage('restart')
   if not async then synchronize(devices) end
   cutorch.setDevice(curDevice)
end

function nccl.bcast(inputs, async, root)
   root = checkroot(root, #inputs)
   local curDevice = cutorch.getDevice()
   local comm, devices, types = getComm(inputs)
   local count = inputs[1]:nElement()
   collectgarbage('stop')
   for i=1,#inputs do
      cutorch.setDevice(devices[i]+1)
      errcheck('ncclBcast', inputs[i]:data(), count, types[i],
         root-1, comm[i-1], cudaStream())
   end
   collectgarbage('restart')
   if not async then synchronize(devices) end
   cutorch.setDevice(curDevice)
end

function nccl.allGather(inputs, outputs, async)
   local curDevice = cutorch.getDevice()
   local comm, devices, types = getComm(inputs, outputs)
   local count = inputs[1]:nElement()
   assert(outputs, "can not do in-place allGather")
   collectgarbage('stop')
   for i=1,#inputs do
      cutorch.setDevice(devices[i]+1)
      errcheck('ncclAllGather', inputs[i]:data(), count, types[i],
         outputs[i]:data(), comm[i-1], cudaStream())
   end
   collectgarbage('restart')
   if not async then synchronize(devices) end
   cutorch.setDevice(curDevice)
end

function nccl.reduceScatter(inputs, outputs, async)
   local curDevice = cutorch.getDevice()
   local comm, devices, types = getComm(inputs, outputs)
   assert(outputs, "can not do in-place reduceScatter")
   assert(outputs[1], "output tensors should be allocated")
   local count = outputs[1]:nElement()
   collectgarbage('stop')
   for i=1,#inputs do
      cutorch.setDevice(devices[i]+1)
      errcheck('ncclReduceScatter', inputs[i]:data(), outputs[i]:data(), count,
         types[i], 'ncclSum', comm[i-1], cudaStream())
   end
   collectgarbage('restart')
   if not async then synchronize(devices) end
   cutorch.setDevice(curDevice)
end

return nccl
