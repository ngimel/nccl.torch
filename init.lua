require 'cutorch'
nccl = {}
include 'ffi.lua'
local C = nccl.C
local ffi = require 'ffi'


nccl.communicators = {}

function nccl.createCommunicators(devices)    
   assert(type(devices)=='number' or torch.typename(devices) == 'torch.IntTensor', "argument type not supported")
   if type(devices)=='number' then
      local devicestensor=torch.IntTensor(devices)
      for i=1,devices do devicestensor[i]=i-1 end
      devices = devicestensor       
   end       
   local ind=0
   for i=1,devices:nElement() do ind = ind + 2^devices[i] end
   if not nccl.communicators[ind] then      
      --create communicator and register its garbage collector
      nccl.communicators[ind]=ffi.new('struct ncclComm*[?]',devices:nElement())
      local function destroy(communicators)
         for i=1,devices:nElement() do
             C['ncclCommDestroy'](communicators[i])
         end
      end
      ffi.gc(nccl.communicators[ind],destroy)
      C['ncclCommInitAll'](nccl.communicators[ind],devices:nElement(),devices:data())
--      local function destroyComm(d)
--         C['ncclCommDestroy'](d[0])
--      end
--      ffi.gc(nccl.communicators[ind], destroyComm)
   end  
   return nccl.communicators[ind]
end




--TODO - make sure order of the GPUs is checked in the communicator
--TODO allow to use empty or wrong size outputs, as long as they are on the correct GPU
--TODO check the sizes of all the tensors

function getComm(inputs,outputs,outtoinputsizeratio)   
   local ind=0
   local devices = torch.IntTensor(#inputs)         
   for i,v in ipairs(inputs) do
      local curdevice=v:getDevice()    
      ind = ind + 2^(curdevice-1)      
      devices[i]=curdevice - 1 --zero-based for cuda
   end      
--try to find existing communicator
   comm=nccl.communicators[ind] or nccl.createCommunicators(devices)  
   return comm,devices
end



function synchronize(devices)
   for i = 1, devices:nElement() do
      cutorch.setDevice(devices[i]+1)
      cutorch.streamSynchronize(cutorch.getStream())
   end
end

function nccl.allReduce(inputs, outputs, async)  
   curDevice = cutorch.getDevice() 
   comm,devices = getComm(inputs,outputs,1)
   count = inputs[1]:nElement()
   outputs = outputs or inputs
   for i=1,#inputs do
      cutorch.setDevice(devices[i]+1)
      stream = ffi.C.THCState_getCurrentStream(cutorch.getState())
      C.ncclAllReduce(inputs[i]:data(),outputs[i]:data(),count,'ncclFloat','ncclSum',
         comm[i-1],stream) 
   end   
   if not async then synchronize(devices) end
   cutorch.setDevice(curDevice)   
end

function nccl.reduce(inputs,outputs,async)
   curDevice = cutorch.getDevice() 
   comm,devices = getComm(inputs,outputs,1)
   count = inputs[1]:nElement()
   outputs = outputs or inputs   
   for i=1,#inputs do
      cutorch.setDevice(devices[i]+1)
      stream = ffi.C.THCState_getCurrentStream(cutorch.getState())
      local output
      if outputs[i] then output = outputs[i]:data() end
      C.ncclReduce(inputs[i]:data(),output,count,'ncclFloat','ncclSum',
      devices[1],comm[i-1],stream)
   end
   if not async then synchronize(devices) end
   cutorch.setDevice(curDevice)      
end


function nccl.bcast(inputs, async)
   curDevice = cutorch.getDevice() 
   comm,devices = getComm(inputs,outputs,1)
   count = inputs[1]:nElement()   
   for i=1,#inputs do
      cutorch.setDevice(devices[i]+1)
      stream = ffi.C.THCState_getCurrentStream(cutorch.getState())      
      C.ncclBcast(inputs[i]:data(),count,'ncclFloat',devices[1],comm[i-1],stream)      
   end
   if not async then synchronize(devices) end
   cutorch.setDevice(curDevice)      
end

function nccl.allGather(inputs,outputs,async)
   curDevice = cutorch.getDevice() 
   comm,devices = getComm(inputs,outputs,1)
   count = inputs[1]:nElement()   
   assert(outputs, "can not do in-place allGather")
   for i=1,#inputs do
      cutorch.setDevice(devices[i]+1)
      stream = ffi.C.THCState_getCurrentStream(cutorch.getState())      
      C.ncclAllGather(inputs[i]:data(),count,'ncclFloat',outputs[i]:data(),comm[i-1],stream)      
   end
   if not async then synchronize(devices) end
   cutorch.setDevice(curDevice)      
end



function nccl.reduceScatter(inputs,outputs,async)
   curDevice = cutorch.getDevice()
   comm,devices = getComm(inputs,outputs,1)
   assert(outputs, "can not do in-place reduceScatter")
   assert(outputs[1],"output tensors should be allocated")
   count = outputs[1]:nElement()
   for i=1,#inputs do
      cutorch.setDevice(devices[i]+1)
      stream = ffi.C.THCState_getCurrentStream(cutorch.getState())
      C.ncclReduceScatter(inputs[i]:data(),outputs[i]:data(),count, 'ncclFloat', 'ncclSum', 
      comm[i-1],stream)
   end
   if not async then synchronize(devices) end
   cutorch.setDevice(curDevice)
end

