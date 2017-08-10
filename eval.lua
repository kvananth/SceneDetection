require 'torch'
require 'nn'
require 'optim'
opt = {
    dataset = 'hdf5_binary',
    nThreads = 0,
    batchSize = 32,
    loadSize = 256,
    fineSize = 224,
    gpu = 1,
    cudnn = 1,
    model = 'checkpoints/full/',
    ntest = math.huge,
    randomize = 0,
    cropping = 'center',
    data_root = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/frames/',
    data_list_val = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/test_caves.txt',
    mean = {-0.083300798050439,-0.10651495109198,-0.17295466315224},
    margin = 1,
    labelDim = 1,
    labelName = 'scenes',
    --labelFile = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/temporalData/train_jungle.h5',
    labelFile = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/temporalData/train_test.h5',
    --labelFile = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/temporalData/train_jungle_deserts_ice.h5',
    --labelFile_val = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/temporalData/test_freshWater.h5'
    --labelFile_val = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/temporalData/test_full.h5'
    labelFile_val = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/temporalData/test.h5'
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpu > 0 then
    require 'cunn'
    cutorch.setDevice(opt.gpu)
end
if opt.gpu > 0 and opt.cudnn > 0 then
    require 'cudnn'
end

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
opt['split'] = 'val'
opt['nThreads'] = 0 -- important, otherwise crashes, weird!!!
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- load in network
assert(opt.model ~= '', 'no model specified')
print('loading ' .. opt.model)
local net = torch.load(opt.model)
net:evaluate()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize)

--local criterion = nn.HingeEmbeddingCriterion(opt.margin)
local criterion = nn.CrossEntropyCriterion()

-- ship to GPU
if opt.gpu > 0 then
    input = input:cuda()
    label = label:cuda()
    net:cuda()
    criterion:cuda()
end

-- eval
local tm = torch.Timer()
local data_tm = torch.Timer()

local ntest = math.huge
local maxiter = math.floor(math.min(data:size(), ntest) / opt.batchSize)
local outputs, labels, extra
local counter = 0
local Err = 0
local confusion = optim.ConfusionMatrix({1,2})

net:evaluate()
local acc = 0
data:resetCounter()
confusion:zero()
confusion:zero()
for iter = 1, maxiter do
    collectgarbage()
    err = 0
    data_tm:reset(); data_tm:resume()
    data_im, data_label, extra = data:getBatch()
    data_tm:stop()


    input:copy(data_im:squeeze())
    label:copy(data_label)

    local output = net:forward(input)
    err = criterion:forward(output, label)
    Err = Err + err

    local _,preds = output:float():sort(2, true)
    preds = preds:narrow(2,1,1)
    ac=0
    for i=1, opt.batchSize do
        confusion:add(preds[i][1], data_label[i][1])
        if preds[i][1] == data_label[i][1] then
            ac = ac + 1
        end
    end

    acc = acc + ac
    counter = counter + opt.batchSize

    print(('Eval [%8d / %8d] Err: %.6f Acc: %.2f'):format(iter, maxiter, err, ac/opt.batchSize))
end

print(('Eval Summary Err: %.6f Acc: %.2f'):format(Err/maxiter, acc/counter))
--torch.save("preds.t7", {outputs:float(), labels})
print(('Summary  %s \t Accuracy: %.4f'):format(opt.model, acc/counter))
