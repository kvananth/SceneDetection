require 'torch'
require 'nn'
require 'optim'
-- to specify these at runtime, you can do, e.g.:
--    $ lr=0.001 th main.lua
opt = {
    dataset = 'hdf5_binary',   -- indicates what dataset load to use (in data.lua)
    nThreads = 16,        -- how many threads to pre-fetch data
    batchSize = 32,      -- self-explanatory
    loadSize = 256,       -- when loading images, resize first to this size
    fineSize = 224,       -- crop this size from the loaded image
    nClasses = 401,       -- number of category
    lr = 0.001,           -- learning rate
    lr_decay = 1500,     -- how often to decay learning rate (in epoch's)
    beta1 = 0.9,          -- momentum term for adam
    meanIter = 0,         -- how many iterations to retrieve for mean estimation
    saveIter = 200,     -- write check point on this interval
    niter = 100000,       -- number of iterations through dataset
    gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
    cudnn = 1,            -- whether to use cudnn or not
    finetune = '',        -- if set, will load this network instead of starting from scratch
    randomize = 1,        -- whether to shuffle the data file or not
    cropping = 'random',  -- options for data augmentation
    display_port = 9000,  -- port to push graphs
    name = 'full_vgg_reg', -- the name of the experiment (by default, filename)
    data_root = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/frames/',
    data_list = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/train_small.txt',
    data_list_val = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/test_caves.txt',
    mean = {-0.083300798050439,-0.10651495109198,-0.17295466315224},
    margin = 1, -- margin for loss function
    labelDim = 1,
    labelName = 'scenes',
    labelFile = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/temporalData/train_full_orig.h5',
    --labelFile = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/train_augmented.h5',
    --labelFile_val = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/temporalData/test_full.h5'
    labelFile_val = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/temporalData/test_full_balanced_orig.h5'
    --labelFile_val = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/val_balanced.h5'
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
opt['nThreads'] = 0
opt['randomize'] = 0
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

    local modelType = 'vgg' -- alex/vgg/res

    if modelType == 'alex' then
        premodel = torch.load('data/imagenet_pretrained_alexnet.t7')
    elseif modelType == 'vgg' then
        premodel = torch.load('data/imagenet_pretrained_vgg.t7')
    elseif modelType == 'res' then
        premodel = torch.load('data/resnet-101.t7')
    end
    if modelType == 'alex' or modelType == 'vgg' then
        prefeats = safe_unpack(premodel.features)
        pretop = safe_unpack(premodel.top)
    end

    if modelType == 'alex' then prefeats:add(cudnn.SpatialMaxPooling(4,4,2,2)) end-- for alexnet
    if modelType == 'vgg' then prefeats:add(cudnn.SpatialMaxPooling(2,2,2,2)) end -- for vgg
    if modelType == 'alex' or modelType == 'vgg' then
        prefeats:add(nn.View(-1):setNumInputDims(3))
        for i=1, #pretop do prefeats:add(pretop:get(i)) end
    end
    local classifier = nn.Sequential()
    --classifier:add(nn.Linear(4096,1024))
    --classifier:add(cudnn.ReLU(true))
    --classifier:add(nn.Dropout(0.5))
    if modelType == 'alex' or modelType == 'vgg' then classifier:add(nn.Linear(4096,2)) else classifier:add(nn.Linear(1000,2)) end

    net = nn.Sequential()
    if modelType == 'alex' or modelType == 'vgg' then net:add(prefeats) else net:add(premodel) end
    net:add(classifier)

else -- load in existing network
    print('loading ' .. opt.finetune)
    net = torch.load(opt.finetune)
end

-- define the loss
--local weightVector = torch.Tensor({(2932+775)/2932, (775+2932)/775})
--local weightVector = torch.Tensor({1, 2})
local criterion = nn.CrossEntropyCriterion()
--local criterion = nn.HingeEmbeddingCriterion(opt.margin)

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
    net = cudnn.convert(net, cudnn)
end


local parameters, gradParameters = net:parameters()
print(net)

-- show graphics
disp = require 'display'
opt.hostname = sys.execute('hostname -s') .. ':' ..opt.display_port
disp.url = 'http://localhost:' .. opt.display_port .. '/events'

-- optimization closure
-- the optimizer will call this function to get the gradients
-- this matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix({1,2})

local acc = 0
local data_im,data_label
local preds
collectgarbage()
local currentScore, bestScore
function eval()
    local ntest = math.huge
    local maxiter = math.floor(math.min(val_data:size(), ntest) / opt.batchSize)
    local outputs, labels, extra
    local counter = 0
    local Err = 0
    net:evaluate()
    acc = 0
    val_data:resetCounter()
    confusion:zero()

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
        --print(output:view(1,opt.batchSize))

        ac = 0
        local _,preds = output:float():sort(2, true)
        preds = preds:narrow(2,1,1)
        for i=1, opt.batchSize do
            confusion:add(preds[i][1], data_label[i])

            if preds[i][1] == data_label[i] then
                ac = ac + 1
            end
        end

        torch.save("checkpoints/eval_" .. iter, {data_im, data_label, preds, output})
        acc = acc + ac
        counter = counter + opt.batchSize

        print(('Eval [%8d / %8d] Err: %.6f Acc: %.2f'):format(iter, maxiter, err, ac/opt.batchSize))
    end
    print("-----------------------------------------------------------------------")
    print(('Eval Summary Err: %.6f Acc: %.2f'):format(Err/maxiter, acc/counter))
    print(confusion)
    print("-----------------------------------------------------------------------")

    return Err/maxiter, acc/counter
end

local fx = function(x)
    net:training()
    net:zeroGradParameters()
    -- fetch data
    data_tm:reset(); data_tm:resume()
    data_im,data_label, extra = data:getBatch()
    data_tm:stop()

    -- ship data to GPU
    input:copy(data_im:squeeze())
    label:copy(data_label)
    -- forward, backwards
    local output = net:forward(input)
    err = criterion:forward(output, label)

    local df_do = criterion:backward(output, label)
    net:backward(input, df_do)

    local norm,sign= torch.norm,torch.sign
    -- Loss:
    local lambda = 0.1 --0.2
    --err = err + lambda * norm(parameters,2)^2/2
    -- Gradients:
    --gradParameters:add(parameters:clone():mul(lambda))


    acc = 0
    local _,preds = output:float():sort(2, true)
    preds = preds:narrow(2,1,1)
    for i=1, opt.batchSize do
        if preds[i][1] == data_label[i][1] then
            acc = acc + 1
        end
    end

    acc = acc*100/opt.batchSize

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
optimState = {}
opt['cnn_lr'] = 0.0001
opt['classifier_lr'] = 0.001
for i=1, #parameters do
    -- VGG: 29, ALEXNET: 13
    if i<29 then table.insert(optimState, {
        learningRate = opt.cnn_lr,
        learningRateDecay = 0.0,
        momentum = 0.9,
        dampening = 0.0,
        weightDecay = 5e-4
    }
    )
    else
        table.insert(optimState, {
            learningRate = opt.classifier_lr,
            learningRateDecay = 0.0,
            momentum = 0.9,
            dampening = 0.0,
            weightDecay = 5e-4
        }
        )
    end
end

print('Starting Optimization...')
local valerr, valacc = 0

-- train main loop
for counter = 1,opt.niter do
    collectgarbage() -- necessary sometimes

    tm:reset()

    -- do one iteration
    fx()

    if counter%100 == 0 then torch.save('traindata.t7', {data_im, data_label, extra, preds}) end

    print(('%s %s Iter: [%7d / %7d]  Time: %.3f  DataTime: %.3f  Err: %.4f Acc: %.2f'):format(
        opt.name, opt.hostname, counter, opt.niter, tm:time().real, data_tm:time().real, err, acc))

    for i=1, #parameters do
        local feval = function(x)
            local norm,sign= torch.norm,torch.sign
            -- Loss:
            local lambda = 0.1 --0.2
            local error = err + lambda * norm(parameters[i],2)^2/2
            gradParameters[i]:add(parameters[i]:clone():mul(lambda))

            return error, gradParameters[i]
        end

        optim.sgd(feval, parameters[i], optimState[i])
    end

    --optim.adam(fx, parameters, optimState)
    --optim.sgd(fx, parameters, optimState)


    if counter == 1 or counter % opt.saveIter == 0 then
        valerr, valacc = eval()

        currentScore = valacc

        --table.insert(val_history, {counter, valerr})
        --disp.plot(val_history, {win=5, title=opt.name, labels = {"iteration", "valError"}})
        table.insert(acc_history, {counter, valacc})
        disp.plot(acc_history, {win=6, title=opt.name, labels = {"iteration", "accuracy"}})

        if counter == 1 or currentScore > bestScore then
            bestScore = currentScore

            -- save checkpoint
            -- :clearState() compacts the model so it takes less space on disk
            print('Saving ' .. opt.name .. '/iter' .. counter .. '_net.t7')
            paths.mkdir('checkpoints')
            paths.mkdir('checkpoints/' .. opt.name)
            torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_net.t7', net:clearState())
            torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_optim.t7', optimState)
            torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_history.t7', history)

        end

    end

    -- logging
    if counter % 10 == 1 then
        table.insert(history, {counter, err, valerr})
        disp.plot(history, {win=1, title=opt.name, labels = {"iteration", "Train Err", "Val Err"}})
    end

    if false then
        if counter % 100 == 1 then
            --w = net.modules[2].modules[1].modules[1].modules[1].weight:float():clone()
            --for i=1,w:size(1) do w[i]:mul(1./w[i]:norm()) end
            --disp.image(w, {win=2, title=(opt.name .. ' conv1')})

            local nn = 8 -- # of imgs to display
            --preds = preds:narrow(1,1,nn)
            print(data_im:size())
            disp.images(data[{{1,1}, {}, {}, {}}], { win=3, width=1000})

            --table.insert(lr_history, {counter, optimState.learningRate})
            --disp.plot(lr_history, {win=4, title=opt.name, labels = {"iteration", "lr"}})
        end
    end

    if opt.lr_decay > 0 and counter % opt.lr_decay == 0 then
        opt.classifier_lr = opt.classifier_lr / 2
        opt.cnn_lr = opt.cnn_lr / 2

        for i =1, #parameters do
            if i<15 then table.insert(optimState, {
                learningRate = opt.cnn_lr,
                learningRateDecay = 0.0,
                momentum = 0.9,
                dampening = 0.0,
                weightDecay = 5e-4
            }
            )
            else
                table.insert(optimState, {
                    learningRate = opt.classifier_lr,
                    learningRateDecay = 0.0,
                    momentum = 0.9,
                    dampening = 0.0,
                    weightDecay = 5e-4
                }
                )
            end
        end
        print('Decreasing learning rate')

    end
end
