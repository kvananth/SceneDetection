require 'torch'
require 'nn'
require 'optim'

-- to specify these at runtime, you can do, e.g.:
--    $ lr=0.001 th main.lua
opt = {
  dataset = 'hdf5',   -- indicates what dataset load to use (in data.lua)
  nThreads = 32,        -- how many threads to pre-fetch data
  batchSize = 2,      -- self-explanatory
  loadSize = 256,       -- when loading images, resize first to this size
  fineSize = 225,       -- crop this size from the loaded image
  patchSize = 64,       -- size of each grid (i.e, batch_sizex3x64x64)
  nClasses = 100,       -- number of category
  lr = 0.005,           -- learning rate
  lr_decay = 10000,     -- how often to decay learning rate (in epoch's)
  beta1 = 0.9,          -- momentum term for adam
  meanIter = 0,         -- how many iterations to retrieve for mean estimation
  saveIter = 10000,     -- write check point on this interval
  niter = 100000,       -- number of iterations through dataset
  gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  cudnn = 1,            -- whether to use cudnn or not
  finetune = '',        -- if set, will load this network instead of starting from scratch
  randomize = 1,        -- whether to shuffle the data file or not
  cropping = 'random',  -- options for data augmentation
  display_port = 8000,  -- port to push graphs
  name = 'puzzleMnist',--paths.basename(paths.thisfile()):sub(1,-5), -- the name of the experiment (by default, filename)
  --data_root = '/nfs_mount/datasets/other/medical/nlm/NLMCXR_png/',
  --data_list = '/nfs_mount/data/ananth/medical_data/trainList_6k.txt',
  mean = {-0.083300798050439,-0.10651495109198,-0.17295466315224},
  labelDim = 1,
  labelName = "/images",
  labelFile = "/nfs_mount/data/ananth/medical_data/train.h5",
  --labelFile = "/nfs_mount/data/ananth/medical_data/train_6k.h5",
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.hostname = sys.execute('hostname -s') .. ':' ..opt.display_port

print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- if using GPU, select indicated one
if opt.gpu > 0 then
  require 'cudnn'
  require 'cunn'
  cutorch.setDevice(opt.gpu)
end

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- define the model
local net = nn.Sequential()
if opt.finetune == '' then -- build network from scratch
  local features = torch.load('data/imagenet_pretrained_alexnet.t7')
  
  --[[local features = nn.Sequential()
  
  features:add(cudnn.SpatialConvolution(3,96,11,11,4,4,5,5))       -- 224 -> 55
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2,1,1))               -- 55 ->  27
  features:add(cudnn.SpatialCrossMapLRN(5, 0.0001,0.75,2))
  features:add(cudnn.SpatialConvolution(96,256,5,5,1,1,2,2))       --  27 -> 27
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2,1,1))                   --  27 ->  13
  features:add(cudnn.SpatialCrossMapLRN(5, 0.0001,0.75,2))
  features:add(cudnn.SpatialConvolution(256,384,3,3,1,1,1,1))      --  13 ->  13
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialConvolution(384,384,3,3,1,1,1,1))      --  13 ->  13
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
  features:add(nn.View(-1):setNumInputDims(3))
  features:add(nn.Linear(4096, 4096))
  features:add(cudnn.ReLU())

  local siamese_encoder = nn.ParallelTable()
  siamese_encoder:add(features)
  siamese_encoder:add(features:clone('weight','bias', 'gradWeight','gradBias'))

  local classifier = nn.Sequential()
  classifier:add(nn.Linear(2*1024, 4096))
  classifier:add(cudnn.ReLU())
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(4096, opt.nClasses))

  net:add(nn.SplitTable(2))
  net:add(siamese_encoder)
  net:add(nn.JoinTable(2))
  net:add(classifier) ]]--

  -- initialize the model
  local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
    end
  end
  net:apply(weights_init) -- loop over all layers, applying weights_init

else -- load in existing network
  print('loading ' .. opt.finetune)
  net = torch.load(opt.finetune)
end

print(net)

-- define the loss
local criterion = nn.CrossEntropyCriterion()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 2, 3, opt.patchSize, opt.patchSize)
local label = torch.Tensor(opt.batchSize)
local err

-- timers to roughly profile performance
local tm = torch.Timer()
local data_tm = torch.Timer()

-- ship everything to GPU if needed
if opt.gpu > 0 then
  input = input:cuda()
  label = label:cuda()
  net:cuda()
  criterion:cuda()
end

-- convert to cudnn if needed
if opt.gpu > 0 and opt.cudnn > 0 then
  net = cudnn.convert(net, cudnn)
end

-- get a vector of parameters
local parameters, gradParameters = net:getParameters()

-- show graphics
disp = require 'display'
disp.url = 'http://localhost:' .. opt.display_port .. '/events'

-- optimization closure
-- the optimizer will call this function to get the gradients
local data_im,data_label
local fx = function(x)
  gradParameters:zero()
  
  -- fetch data
  data_tm:reset(); data_tm:resume()
  data_im,data_label = data:getBatch()
  data_tm:stop()

  print(#data_im)
  -- ship data to GPU
  input:copy(data_im:squeeze())
  label:copy(data_label)
  
  -- forward, backwards
  local output = net:forward(input)
  err = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  net:backward(input, df_do)
 
  local _,preds = output:float():sort(2, true)
   
   top1 = 0  
   top5 = 0
   for i=1, opt.batchSize do 
     local rank = torch.eq(preds[i], data_label[i]):nonzero()[1][1] 
     if rank == 1 then 
        top1 = top1 + 1 
     end
     if rank <= 5 then 
        top5 = top5 + 1
     end
   end 
   top1 = top1:div(opt.batchSize)
   top5 = top5:div(opt.batchSize)
  -- return gradients
  return err, gradParameters
end

local history = {}

-- parameters for the optimization
-- very important: you must only create this table once! 
-- the optimizer will add fields to this table (such as momentum)
local optimState = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}


print('Starting Optimization...')

-- train main loop
for counter = 1,opt.niter do
  collectgarbage() -- necessary sometimes
  
  tm:reset()

  -- do one iteration
  optim.adam(fx, parameters, optimState)
  
  -- logging
  if false then --counter % 10 == 1 then
    table.insert(history, {counter, err})
    disp.plot(history, {win=1, title=opt.name, labels = {"iteration", "err"}})
  end

  if false then --counter % 100 == 1 then
    w = net.modules[2].modules[1].modules[1].weight:float():clone()
    for i=1,w:size(1) do w[i]:mul(1./w[i]:norm()) end
    disp.image(w, {win=2, title=(opt.name .. ' conv1')})
    --disp.image(data_im:narrow(2,1,1), {win=3, title=(opt.name .. ' batch')})
  end
  
  print(('%s %s Iter: [%7d / %7d]  Time: %.3f  DataTime: %.3f  Err: %.4f top-1: %.2f top-5: %.2f'):format(
          opt.name, opt.hostname, counter, opt.niter, tm:time().real, data_tm:time().real,
          err, top1, top5))

  -- save checkpoint
  -- :clearState() compacts the model so it takes less space on disk
  if counter % opt.saveIter == 0 then
    print('Saving ' .. opt.name .. '/iter' .. counter .. '_net.t7')
    paths.mkdir('checkpoints')
    paths.mkdir('checkpoints/' .. opt.name)
    torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_net.t7', net:clearState())
    --torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_optim.t7', optimState)
    torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_history.t7', history)
  end

  -- decay the learning rate, if requested
  if opt.lr_decay > 0 and counter % opt.lr_decay == 0 then
    opt.lr = opt.lr / 5
    print('Decreasing learning rate to ' .. opt.lr)

    -- create new optimState to reset momentum
    optimState = {
      learningRate = opt.lr,
      beta1 = opt.beta1,
    }
  end
end
