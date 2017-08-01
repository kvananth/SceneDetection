require 'torch'
require 'nn'
require 'optim'
-- to specify these at runtime, you can do, e.g.:
--    $ lr=0.001 th main.lua
opt = {
    dataset = 'simple',   -- indicates what dataset load to use (in data.lua)
    nThreads = 16,        -- how many threads to pre-fetch data
    batchSize = 32,      -- self-explanatory
    loadSize = 256,       -- when loading images, resize first to this size
    fineSize = 224,       -- crop this size from the loaded image
    nClasses = 401,       -- number of category
    lr = 0.001,           -- learning rate
    lr_decay = 2000,     -- how often to decay learning rate (in epoch's)
    beta1 = 0.9,          -- momentum term for adam
    meanIter = 0,         -- how many iterations to retrieve for mean estimation
    saveIter = 500,     -- write check point on this interval
    niter = 100000,       -- number of iterations through dataset
    gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
    cudnn = 1,            -- whether to use cudnn or not
    finetune = '',        -- if set, will load this network instead of starting from scratch
    randomize = 1,        -- whether to shuffle the data file or not
    cropping = 'random',  -- options for data augmentation
    display_port = 9001,  -- port to push graphs
    name = 'sep', -- the name of the experiment (by default, filename)
    data_root = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/frames/',
    data_list = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/train.txt',
    data_list_val = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/test.txt',
    mean = {-0.083300798050439,-0.10651495109198,-0.17295466315224},
    margin = 2, -- margin for loss function
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

local parameters, gradParameters
-- define the model
local net
if opt.finetune == '' then -- build network from scratch
     
    local premodel = torch.load('data/imagenet_pretrained_alexnet.t7')
    local prefeatures = safe_unpack(premodel.features)
    prefeatures:add(cudnn.SpatialMaxPooling(3,3,2,2))
    prefeatures:add(nn.View(-1):setNumInputDims(3))
    local pretop = safe_unpack(premodel.top)
    pretop.modules[1] = nn.Linear(12544, 4096)

    --prefeatures:insert(nn.Probe())

    net= nn.Sequential():add(prefeatures):add(pretop)
    distMetric = nn.PairwiseDistance(2)

    --[[local siamese_encoder = nn.ParallelTable()
    siamese_encoder:add(siamese)
    siamese_encoder:add(siamese:clone('weight','bias', 'gradWeight','gradBias'))

    net = nn.Sequential()
    net:add(nn.SplitTable(2))
    net:add(siamese)
    net:add(nn.PairwiseDistance(2))]]

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
    --net:apply(weights_init) -- loop over all layers, applying weights_init

else -- load in existing network
    print('loading ' .. opt.finetune)
    net = torch.load(opt.finetune)
end

parameters, gradParameters = net:getParameters()
print(net)

-- define the loss
--local criterion = nn.CrossEntropyCriterion()
local criterion = nn.HingeEmbeddingCriterion(opt.margin)


-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 2, 3, opt.fineSize, opt.fineSize)
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
    distMetric:cuda()
    criterion:cuda()
end

-- convert to cudnn if needed
if opt.gpu > 0 and opt.cudnn > 0 then
    net = cudnn.convert(net, cudnn)
end

--[[for i=1,2 do
		net.modules[2].modules[i].modules[1].accGradParameters = function(x) end
		net.modules[2].modules[i].modules[5].accGradParameters = function(x) end
		net.modules[2].modules[i].modules[9].accGradParameters = function(x) end
end]]


-- show graphics
disp = require 'display'
opt.hostname = sys.execute('hostname -s') .. ':' ..opt.display_port
disp.url = 'http://localhost:' .. opt.display_port .. '/events'

-- optimization closure
-- the optimizer will call this function to get the gradients
-- this matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix({-1,1})
local acc = 0
local data_im,data_label
local preds
collectgarbage()

function eval()
    local ntest = math.huge
    local maxiter = math.floor(math.min(val_data:size(), ntest) / opt.batchSize)
    local outputs, labels, extra
    local counter = 0
    local Err = 0
    net:evaluate()
    acc = 0
    val_data:resetCounter()

    for iter = 1, maxiter do
        collectgarbage()
        err = 0
        data_tm:reset(); data_tm:resume()
        data_im, data_label, extra = val_data:getBatch()
        data_tm:stop()
        
        input:copy(data_im:squeeze())
	      label:copy(data_label)

        local output = net:forward(input)
        err = criterion:forward(output, label)
        Err = Err + err
        print(output:view(1,opt.batchSize))

        output:apply(function(x)
            l = 1 if x > opt.margin then l = -1 end
            return l end)

        local ac =  output:eq(data_label:cuda()):sum()
        acc = acc + ac
        counter = counter + opt.batchSize

        print(('Eval [%8d / %8d] Err: %.6f Acc: %.2f'):format(iter, maxiter, err, ac/opt.batchSize))
    end
    print(('Eval Summary Err: %.6f Acc: %.2f'):format(Err/maxiter, acc/counter))
    return Err/maxiter, acc/counter
end

local fx = function(x)
    net:training()
    gradParameters:zero()

    -- fetch data
    data_tm:reset(); data_tm:resume()
    data_im,data_label, extra = data:getBatch()
    --torch.save('traindata.t7', {data_im, data_label, extra})
    --os.exit()
    data_tm:stop()

    -- ship data to GPU
    input:copy(data_im:squeeze())
    label:copy(data_label)
    
    --print("before ", net.modules[2].modules[1].modules[1].modules[13].gradWeight[1][1])

    -- forward, backwards
    local features1= net:forward(input:narrow(2,1,1):reshape(opt.batchSize,3, opt.fineSize, opt.fineSize))
    local features1= net:forward(input:narrow(2,2,1):reshape(opt.batchSize,3, opt.fineSize, opt.fineSize))
    dist = distMetric:forward({features1, features2})
    err = criterion:forward(dist, label)

    local df_do = criterion:backward(output, label)
    --print("output: ", output:normal():view(2,opt.batchSize/2))
    --print("DEBUG df_do: ", df_do:normal():view(2, opt.batchSize/2))
    net:backward(input:narrow(2,1,1):reshape(opt.batchSize,3, opt.fineSize, opt.fineSize), df_do)
    gradParameters:zero()
    net:backward(input:narrow(2,2,1):reshape(opt.batchSize,3, opt.fineSize, opt.fineSize), df_do)

    --print("after update ", net.modules[2].modules[1].modules[1].modules[13].gradWeight[1][1])

    -- locals:
    local norm,sign= torch.norm,torch.sign
    -- Loss:
    local lambda = 0.1 --0.2
    err = err + lambda * norm(parameters,2)^2/2
    -- Gradients:
    gradParameters:add(parameters:clone():mul(lambda))
    
    output:apply(function(x) ll = 1
                             if x > opt.margin then
                                ll = -1
                             end
                             return ll 
                             end)
    
 

    preds = output:eq(label)
    acc = preds:sum()
    acc = acc*100/output:size(1)
    -- return gradients
    return err, gradParameters
end

local history = {}
local lr_history = {}
local val_history = {}
local acc_history = {}

-- parameters for the optimization
-- very important: you must only create this table once! 
-- the optimizer will add fields to this table (such as momentum)
local optimState = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
    learningRateDecay = 0.0,
    momentum = 0.9,
    dampening = 0.0,
    weightDecay = 5e-4
}


print('Starting Optimization...')
local valerr, valacc = 0

-- train main loop
for counter = 1,opt.niter do
    collectgarbage() -- necessary sometimes

    tm:reset()

    -- do one iteration
    --optim.adam(fx, parameters, optimState)
    optim.sgd(fx, parameters, optimState)

    if counter%100 == 0 then torch.save('traindata.t7', {data_im, data_label, extra, preds}) end

    print(('%s %s Iter: [%7d / %7d]  Time: %.3f  DataTime: %.3f  Err: %.4f Acc: %.2f'):format(
        opt.name, opt.hostname, counter, opt.niter, tm:time().real, data_tm:time().real, err, acc))
    --print(confusion)

    if counter % opt.saveIter == 0 then
        valerr, valacc = eval()

        --table.insert(val_history, {counter, valerr})
        --disp.plot(val_history, {win=5, title=opt.name, labels = {"iteration", "valError"}})
        table.insert(acc_history, {counter, valacc})
        disp.plot(acc_history, {win=6, title=opt.name, labels = {"iteration", "accuracy"}})

        -- save checkpoint
        -- :clearState() compacts the model so it takes less space on disk
        print('Saving ' .. opt.name .. '/iter' .. counter .. '_net.t7')
        paths.mkdir('checkpoints')
        paths.mkdir('checkpoints/' .. opt.name)
        torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_net.t7', net:clearState())
        --torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_optim.t7', optimState)
        torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_history.t7', history)

    end

    -- logging
    if counter % 10 == 1 then
        table.insert(history, {counter, err, valerr})
        disp.plot(history, {win=1, title=opt.name, labels = {"iteration", "Train Err", "Val Err"}})
    end

    if counter % 100 == 1 then
        w = net.modules[2].modules[1].modules[1].modules[1].weight:float():clone()
        for i=1,w:size(1) do w[i]:mul(1./w[i]:norm()) end
        disp.image(w, {win=2, title=(opt.name .. ' conv1')})

        local correctPreds = preds:eq(1):nonzero()
        local incorrectPreds = preds:eq(0):nonzero()

        local nn = 2 -- # of imgs to display
        preds = preds:narrow(1,1,nn)
        local im1 = data_im:narrow(2,1,1):reshape(opt.batchSize,3, opt.fineSize, opt.fineSize):narrow(1,1,nn)
        local im2 = data_im:narrow(2,2,1):reshape(opt.batchSize,3, opt.fineSize, opt.fineSize):narrow(1,1,nn)
        local disp_imgs1, disp_imgs2

        for i=1,nn do
            dim = 3
            if i<6 then
                local im = im1[i]:cat(im2[i],3)
                disp_imgs1 = (i==1) and im or disp_imgs1:cat(im,dim)
            else
                local im = im1[i]:cat(im2[i],3)
                disp_imgs2 = (i==6) and im or disp_imgs2:cat(im,dim)
            end
        end
        --disp.images({disp_imgs1:cat(disp_imgs2,2)}, { win=3, width=1000})

        table.insert(lr_history, {counter, optimState.learningRate})
        disp.plot(lr_history, {win=4, title=opt.name, labels = {"iteration", "lr"}})
    end

    confusion:zero()
    acc = 0

    -- decay the learning rate, if requested
    if opt.lr_decay > 0 and counter % opt.lr_decay == 0 then
        opt.lr = opt.lr / 2
        print('Decreasing learning rate to ' .. opt.lr)

        -- create new optimState to reset momentum
        optimState = {
            learningRate = opt.lr,
            beta1 = opt.beta1,
            learningRateDecay = 0.0,
            momentum = 0.9,
            dampening = 0.0,
            weightDecay = 5e-4
        }
    end
end
