require 'nccl'
require 'cutorch'


local cmd = torch.CmdLine()
cmd:text()
cmd:text('nccl test script')
cmd:text()
cmd:text('Options:')
cmd:option('-nGPU',               2, 'Number of GPUs to use by default')
cmd:option('-precision',   'single', 'precision of Cuda Tensor, single, half, double')
cmd:option('-tensorSize',       128, 'size of Cuda Tensor (precision) to reduce')
cmd:option('-outOfPlace',          false, 'whether to perform reduction in-place or out-of-place')
cmd:option('-operation',      'allReduce','what collective to perform, allReduce, reduce, allGather, bcast')
cmd:option('-nsizes', 1, 'number of sizes to test')
cmd:option('-nrepetitions', 10, 'number of times to repeat operation for benchmarking')
cmd:text()

local opt = cmd:parse(arg or {})
--no point in in-place broadcast
if opt.operation == 'bcast' then opt.outOfPlace = false end
--allGather can not be done in-place
if opt.operation == 'allGather' then opt.outOfPlace = true end


local inputs = {}
local function getprecision(precision)
   if precision == 'single' then
      return 'CudaTensor', 4
   elseif precision == 'double' then
      return 'CudaDoubleTensor', 8
   elseif precision == 'half' then
      return 'CudaHalfTensor', 2
   else
      error('unrecognized precision ' .. precision)
   end
end
local size = opt.tensorSize
local tensorType, sizeOfDatatype = getprecision(opt.precision)
print("size\t","elapsed time\t", "alg bw\t", "link bw\t")
local val = 1
for i=1,opt.nsizes do
   for i=1,opt.nGPU do
      cutorch.setDevice(i)
      inputs[i] = torch[tensorType](size):fill(val)
   end
   local nrep = opt.nrepetitions   
   local outputs, result
   if opt.outOfPlace then      
      outputs = {}
      for i=1,opt.nGPU do
         cutorch.setDevice(i)
         outputs[i] = torch[tensorType](size)
      end
      result = outputs
   else
      outputs = nil
      result = inputs
   end
   --warm-up   
   nccl.allReduce(inputs,outputs)
   --reset inputs and outputs
   for i=1,opt.nGPU do
      cutorch.setDevice(i)
      inputs[i]:fill(val)
      if opt.outOfPlace then
         outputs[i]:fill(0)
      end
   end
   local tm = torch.Timer()
   for i=1,nrep do
      if opt.operation == 'allReduce' then
         nccl.allReduce(inputs,outputs)
      elseif opt.operation == 'reduce' then         
         nccl.reduce(inputs,outputs,nil,1)         
      elseif opt.operation == 'bcast' then
         for i=2,opt.nGPU do
            cutorch.setDevice(i)
            inputs[i]:fill(0)
         end
         nccl.bcast(inputs,nil,1)
      elseif opt.operation == 'allGather' then
         for i=1,opt.nGPU do
            cutorch.setDevice(i)
            outputs[i]:resize(size*opt.nGPU)     
         end
         nccl.allGather(inputs,outputs)
      end
   end
   local elapsed=tm:time().real/nrep
   local algbw = size * sizeOfDatatype / elapsed * 1e-9
   local linkbw = algbw
   if opt.operation == 'allReduce' then
      linkbw = algbw * 2 * (opt.nGPU-1)/ opt.nGPU
   elseif opt.operation == 'allGather' then
      algbw = algbw * opt.nGPU
      linkbw = algbw * (opt.nGPU-1)/opt.nGPU
   end
   print(size*sizeOfDatatype, elapsed, algbw, linkbw)
   local expected
   if opt.operation == 'allReduce' then
      if opt.outOfPlace then
         expected = opt.nGPU * val
      else
         expected = opt.nGPU^(nrep) * val
      end
   elseif opt.operation == 'bcast' or opt.operation == 'allGather' then
      expected = val
   elseif opt.operation == 'reduce' then
      if opt.outOfPlace then
         expected = opt.nGPU * val
      else
         expected = val + nrep * (opt.nGPU - 1)
      end
   end
   local lastGpu
   if opt.operation == 'allReduce' or opt.operation == 'allGather' or 
      opt.operation == 'bcast' then --check results on all the gpus
      lastGpu = opt.nGPU
   else
      lastGpu = result[1]:getDevice()      
   end
   for i=1, lastGpu do
      cutorch.setDevice(result[i]:getDevice())      
      assert(result[i]:max()==expected)         
      assert(result[i]:min()==expected)         
      if opt.outOfPlace then
         assert(inputs[i]:max(),val)         
         assert(inputs[i]:min(),val)
      end
   end
   size = size*2
end

