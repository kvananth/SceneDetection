require 'torch'
require 'nn'
require 'optim'

-- to specify these at runtime, you can do, e.g.:
--    $ lr=0.001 th main.lua
opt = {
    dataset = 'simple',   -- indicates what dataset load to use (in data.lua)
    nThreads = 8,        -- how many threads to pre-fetch data
    batchSize = 16,      -- self-explanatory
    loadSize = 256,       -- when loading images, resize first to this size
    fineSize = 224,       -- crop this size from the loaded image
    nClasses = 401,       -- number of category
    lr = 0.001,           -- learning rate
    lr_decay = 5000,     -- how often to decay learning rate (in epoch's)
    beta1 = 0.9,          -- momentum term for adam
    meanIter = 0,         -- how many iterations to retrieve for mean estimation
    saveIter = 2000,     -- write check point on this interval
    niter = 100000,       -- number of iterations through dataset
    gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
    cudnn = 1,            -- whether to use cudnn or not
    finetune = '',        -- if set, will load this network instead of starting from scratch
    randomize = 0,        -- whether to shuffle the data file or not
    cropping = 'random',  -- options for data augmentation
    display_port = 9000,  -- port to push graphs
    name = 'full', -- the name of the experiment (by default, filename)
    data_root = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/frames/',
    data_list = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/train.txt',
    data_list_val = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/test.txt',
    mean = {-0.083300798050439,-0.10651495109198,-0.17295466315224},
    margin = 1, -- margin for loss function
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- if using GPU, select indicated one
if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.gpu)
end

local DataLoader = paths.dofile('data/data.lua')
opt['split'] = 'train'
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Train dataset size: ", data:size())

--local Val_DataLoader = paths.dofile('data/data.lua'
opt['split'] = 'val'
local val_data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Val dataset size: ", val_data:size())

function safe_unpack(self)
    if self.unpack and self.model then
        return self:unpack()
    else
        local model = self.model
        for k,v in ipairs(model:listModules()) do
            if v.weight and not v.gradWeight then
                v.gradWeight = v.weight:clone()
                v.gradBias = v.bias:clone()
            end
        end
        return model
    end
end


-- define the model
local net
if opt.finetune == '' then -- build network from scratch

    local premodel = torch.load('data/imagenet_pretrained_alexnet.t7')
    local prefeatures = safe_unpack(premodel.features)
    prefeatures:add(cudnn.SpatialMaxPooling(3,3,2,2))
    prefeatures:add(nn.View(-1):setNumInputDims(3))

    local pretop = safe_unpack(premodel.top)
    pretop.modules[1] = nn.Linear(12544, 4096)
    pretop:add(nn.Linear(4096,512))

    --prefeatures:insert(nn.Probe())

    local siamese = nn.Sequential():add(prefeatures):add(pretop)

    --parameters, gradParameters = siamese:getParameters()

    local siamese_encoder = nn.ParallelTable()
    siamese_encoder:add(siamese)
    siamese_encoder:add(siamese:clone('weight','bias', 'gradWeight','gradBias'))

    net = nn.Sequential()
    net:add(nn.SplitTable(2))
    net:add(siamese_encoder)
    net:add(nn.PairwiseDistance(2))

else -- load in existing network
  print('loading ' .. opt.finetune)
  net = torch.load(opt.finetune)
end

print(net)

-- define the loss
local criterion = nn.CrossEntropyCriterion()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
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
  require 'cudnn'
  net = cudnn.convert(net, cudnn)
end

local parameters, gradParameters = net:getParameters()
print(net)

-- show graphics
disp = require 'display'
opt.hostname = sys.execute('hostname -s') .. ':' ..opt.display_port
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

  -- ship data to GPU
  input:copy(data_im:squeeze())
  label:copy(data_label)
  
  -- forward, backwards
  local output = net:forward(input)
  err = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  net:backward(input, df_do)
  
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
  if counter % 10 == 1 then
    table.insert(history, {counter, err})
    disp.plot(history, {win=1, title=opt.name, labels = {"iteration", "err"}})
  end

  if counter % 100 == 1 then
    w = net.modules[1].weight:float():clone()
    for i=1,w:size(1) do w[i]:mul(1./w[i]:norm()) end
    disp.image(w, {win=2, title=(opt.name .. ' conv1')})
    disp.image(data_im, {win=3, title=(opt.name .. ' batch')})
  end
  
  print(('%s %s Iter: [%7d / %7d]  Time: %.3f  DataTime: %.3f  Err: %.4f'):format(
          opt.name, opt.hostname, counter, opt.niter, tm:time().real, data_tm:time().real,
          err))

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
    opt.lr = opt.lr / 10
    print('Decreasing learning rate to ' .. opt.lr)

    -- create new optimState to reset momentum
    optimState = {
      learningRate = opt.lr,
      beta1 = opt.beta1,
    }
  end
end
