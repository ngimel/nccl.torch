local ffi = require 'ffi'

--[[
/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

--]]

local res
local version
local libnames = {
   'libnccl.so.2', 'libnccl.2.dylib',
   'libnccl.so.1', 'libnccl.1.dylib'}

local NCCL_PATH = os.getenv('NCCL_PATH')
if NCCL_PATH then
   io.stderr:write('Found Environment variable NCCL_PATH = ' .. NCCL_PATH .. '\n')
   table.insert(libnames, 1, NCCL_PATH)
end

local ok = false
local err
local errs = {}

for i=1,#libnames do
   ok, err = pcall(function () res = ffi.load(libnames[i]) end)
   errs[ libnames[i] ] = err
   if ok then
      version = assert(tonumber(libnames[i]:match('libnccl.?s?o?.?(%d).?d?y?l?i?b?')), [[
         Could not get library version from its file name. The name should
         be "libnccl.so.{1|2}" or "libnccl.{1|2}.dylib"
      ]])
      break
   end
end

if not ok then
   for lib, e in pairs(errs) do
      print('Tried loading ' .. lib .. ' but got error ' .. e)
   end
   error([[
      libnccl.so could not be loaded, please refer to error messages above. If libnccl.so was not found, 
      please install nccl, then make sure all the files named as libnccl.so* are placed in your library
      load path (for example /usr/local/lib), or manually add a path to LD_LIBRARY_PATH, or set NCCL_PATH
      to a value like "/some/path/libnccl.so.{1|2}" or "/some/path/libnccl.{1|2}.dylib"
   ]])
end

if version == 1 then
   ffi.cdef[[
/* Opaque handle to communicator */
typedef struct ncclComm* ncclComm_t;

//#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[128]; } ncclUniqueId;

/* Error type */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidDevicePointer    =  4,
               ncclInvalidRank             =  5,
               ncclUnsupportedDeviceCount  =  6,
               ncclDeviceNotFound          =  7,
               ncclInvalidDeviceIndex      =  8,
               ncclLibWrapperNotSet        =  9,
               ncclCudaMallocFailed        = 10,
               ncclRankMismatch            = 11,
               ncclInvalidArgument         = 12,
               ncclInvalidType             = 13,
               ncclInvalidOperation        = 14,
               nccl_NUM_RESULTS            = 15 } ncclResult_t;

/* Generates a unique Id with each call. Used to generate commId for
 * ncclCommInitAll. uniqueId will be created in such a way that it is
 * guaranteed to be unique accross the host. */
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);

/* Creates a new communicator (multi process version).
 * rank must be between 0 and ndev-1 and unique within a communicator clique.
 * ndev is number of logical devices
 * The communicator is created on the current CUDA device.
 * ncclCommInitRank implicitly syncronizes with other ranks, so INIT OF EACH RANK MUST
 * BE CALLED IN A SEPARATE HOST THREADS to avoid deadlock. */
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ndev, ncclUniqueId commId, int rank);

/* Creates a clique of communicators.
 * This is a convenience function to create a single-process communicator clique.
 * Returns an array of ndev newly initialized communicators in comm.
 * comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
 * If devlist is NULL, the first ndev CUDA devices are used.
 * Order of devlist defines user-order of processors within the communicator. */
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, int* devlist);

/* Frees resources associated with communicator object. */
void ncclCommDestroy(ncclComm_t comm);

/* Returns nice error message. */
const char* ncclGetErrorString(ncclResult_t result);

/* Sets count to number of devices in the communicator clique. */
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);

/* Returns cuda device number associated with communicator. */
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);

/* Returns user-ordered "rank" assocaiated with communicator. */
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);

/* Reduction opperation selector */
typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               nccl_NUM_OPS   = 4 } ncclRedOp_t;

/* Data types */
typedef enum { ncclChar       = 0,
               ncclInt        = 1,
//#ifdef CUDA_HAS_HALF
               ncclHalf       = 2,
//#endif
               ncclFloat      = 3,
               ncclDouble     = 4,
               nccl_NUM_TYPES = 5 } ncclDataType_t;

/* Reduces data arrays of length count in sendbuff into recvbuf using op operation.
 * recvbuf may be NULL on all calls except for root device.
 * On the root device, sendbuff and recvbuff are assumed to reside on
 * the same device.
 * Must be called separately for each communicator in communicator clique.
*/
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuf, int count, ncclDataType_t datatype,
                        ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

/* Reduces data arrays of length count in sendbuff using op operation, and leaves
 * identical copies of result on each GPUs recvbuff.
 * Sendbuff and recvbuff are assumed to reside on the same device.
 * Must be called separately for each communicator in communicator clique. */
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

/* Reduces data in sendbuff using op operation and leaves reduced result scattered
 * over the devices so that recvbuff on the i-th GPU will contain the i-th block of
 * the result. Sendbuff and recvbuff are assumed to reside on same device. Assumes
 * sendbuff has size at least ndev*recvcount elements, where ndev is number of
 * communicators in communicator clique
 * Must be called separately for each communicator in communicator clique.*/
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
    int recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);

/* Copies count values from root to all other devices.
 * Root specifies the source device in user-order
 * (see ncclCommInit).
 * Must be called separately for each communicator in communicator clique. */
ncclResult_t ncclBcast(void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);


/* Each device gathers count values from other GPUs.
 * Result is ordered by comm's logical device order.
 * Assumes recvbuff has size at least ndev*count, where ndev is number of communicators
 * in communicator clique.
 * Sendbuff and recvbuff are assumed to reside on same device.
 * Must be called separately for each communicator in communicator clique. */
ncclResult_t ncclAllGather(const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream);


/* The following collective operations are not implemented yet */
///* Gather count values from each device to recvbuff.
// * Result is ordered by comm's logical device order.
// * recvbuff may be NULL for all calls except for root device.
// * On the root device, sendbuff and recvbuff are assumed to reside on the same device.
// * Must be called separately for each communicator in communicator clique. */
// * All GPUs, including root, perform copies into recvbuff.
//ncclResult_t ncclGather(const void* sendbuff, int count, ncclDataType_t datatype,
//                        void* recvbuff, int root, ncclComm_t comm, cudaStream_t stream);

///* Root device scatters count values to each devices.
// * sendbuff may be NULL on all devices except a single root
// * device where it is assumed to have size at least nGPUs*count.
// * recvbuff allocated on each gpu, including root, size=count.
// * Result is ordered by comm's logical device order.
// * Called separately for each device in the ncclComm. */
//ncclResult_t ncclScatter(void* sendbuff, ncclDataType_t datatype, void* recvbuff,
//                         int count, int root, ncclComm_t comm, cudaStream_t stream);
//
///* All GPUs scatter blocks of count elements to other devices.
// * Must be called separately for each device in the ncclComm.
// * sendbuff and recvbuff assumed to reside on same device and
// * have size at least nGPUs*count.
// * Called separately for each device in the ncclComm. */
//ncclResult_t ncclAllToAll(void* sendbuff, int count, ncclDataType_t datatype,
//                          void* recvbuff, ncclComm_t comm, cudaStream_t stream);
]]

elseif version == 2 then

   ffi.cdef[[
/* Opaque handle to communicator */
typedef struct ncclComm* ncclComm_t;

//#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[128]; } ncclUniqueId;

/* Error type */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclNumResults              =  6 } ncclResult_t;

/* Generates a unique Id with each call. Used to generate commId for
 * ncclCommInitAll. uniqueId will be created in such a way that it is
 * guaranteed to be unique accross the host. */
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);

/* Creates a new communicator (multi process version).
 * rank must be between 0 and ndev-1 and unique within a communicator clique.
 * ndev is number of logical devices
 * The communicator is created on the current CUDA device.
 * ncclCommInitRank implicitly syncronizes with other ranks, so INIT OF EACH RANK MUST
 * BE CALLED IN A SEPARATE HOST THREADS to avoid deadlock. */
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ndev, ncclUniqueId commId, int rank);

/* Creates a clique of communicators.
 * This is a convenience function to create a single-process communicator clique.
 * Returns an array of ndev newly initialized communicators in comm.
 * comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
 * If devlist is NULL, the first ndev CUDA devices are used.
 * Order of devlist defines user-order of processors within the communicator. */
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, int* devlist);

/* Frees resources associated with communicator object. */
void ncclCommDestroy(ncclComm_t comm);

/* Returns nice error message. */
const char* ncclGetErrorString(ncclResult_t result);

/* Sets count to number of devices in the communicator clique. */
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);

/* Returns cuda device number associated with communicator. */
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);

/* Returns user-ordered "rank" assocaiated with communicator. */
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);

/* Reduction operation selector */
typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               ncclNumOps     = 4 } ncclRedOp_t;

/* Data types */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
               ncclNumTypes   = 9 } ncclDataType_t;

/* Reduces data arrays of length count in sendbuff into recvbuf using op operation.
 * recvbuf may be NULL on all calls except for root device.
 * On the root device, sendbuff and recvbuff are assumed to reside on
 * the same device.
 * Must be called separately for each communicator in communicator clique.
*/
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuf, size_t count, ncclDataType_t datatype,
                        ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

/* Reduces data arrays of length count in sendbuff using op operation, and leaves
 * identical copies of result on each GPUs recvbuff.
 * Sendbuff and recvbuff are assumed to reside on the same device.
 * Must be called separately for each communicator in communicator clique. */
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

/* Reduces data in sendbuff using op operation and leaves reduced result scattered
 * over the devices so that recvbuff on the i-th GPU will contain the i-th block of
 * the result. Sendbuff and recvbuff are assumed to reside on same device. Assumes
 * sendbuff has size at least ndev*recvcount elements, where ndev is number of
 * communicators in communicator clique
 * Must be called separately for each communicator in communicator clique.*/
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);

/* Copies count values from root to all other devices.
 * Root specifies the source device in user-order
 * (see ncclCommInit).
 * Must be called separately for each communicator in communicator clique. */
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

/* Each device gathers count values from other GPUs.
 * Result is ordered by comm's logical device order.
 * Assumes recvbuff has size at least ndev*count, where ndev is number of communicators
 * in communicator clique.
 * Sendbuff and recvbuff are assumed to reside on same device.
 * Must be called separately for each communicator in communicator clique. */
ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

/*
 * Group Start
 *
 * Start a group call. All subsequent calls to NCCL may not block due to
 * inter-CPU synchronization.
 */
ncclResult_t ncclGroupStart();

/* 
 * Group End
 *
 * End a group call. Wait for all calls since ncclGroupStart to complete 
 * before returning.
 */
ncclResult_t ncclGroupEnd();

/* The following collective operations are not implemented yet */
///* Gather count values from each device to recvbuff.
// * Result is ordered by comm's logical device order.
// * recvbuff may be NULL for all calls except for root device.
// * On the root device, sendbuff and recvbuff are assumed to reside on the same device.
// * Must be called separately for each communicator in communicator clique. */
// * All GPUs, including root, perform copies into recvbuff.
//ncclResult_t ncclGather(const void* sendbuff, int count, ncclDataType_t datatype,
//                        void* recvbuff, int root, ncclComm_t comm, cudaStream_t stream);

///* Root device scatters count values to each devices.
// * sendbuff may be NULL on all devices except a single root
// * device where it is assumed to have size at least nGPUs*count.
// * recvbuff allocated on each gpu, including root, size=count.
// * Result is ordered by comm's logical device order.
// * Called separately for each device in the ncclComm. */
//ncclResult_t ncclScatter(void* sendbuff, ncclDataType_t datatype, void* recvbuff,
//                         int count, int root, ncclComm_t comm, cudaStream_t stream);
//
///* All GPUs scatter blocks of count elements to other devices.
// * Must be called separately for each device in the ncclComm.
// * sendbuff and recvbuff assumed to reside on same device and
// * have size at least nGPUs*count.
// * Called separately for each device in the ncclComm. */
//ncclResult_t ncclAllToAll(void* sendbuff, int count, ncclDataType_t datatype,
//                          void* recvbuff, ncclComm_t comm, cudaStream_t stream);
]]
else
   error('NCCL version ' .. version .. ' not supported')
end

return {res, version}

