require 'torch'
require 'nn'
require 'optim'
opt = {
  dataset = 'simple',
  nThreads = 16,
  batchSize = 64,
  loadSize = 256,
  fineSize = 224,
  gpu = 1,
  cudnn = 1,
  model = '',
  ntest = math.huge,
  randomize = 0,
  cropping = 'center',
  data_root = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/frames/',
  data_list = '/mnt/data/story_break_data/BBC_Planet_Earth_Dataset/test_full.txt',
  mean = {-0.083300798050439,-0.10651495109198,-0.17295466315224}
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
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- load in network
assert(opt.model ~= '', 'no model specified')
print('loading ' .. opt.model)
local net = torch.load(opt.model)
net:evaluate()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 2, 3, opt.fineSize, opt.fineSize)

-- ship to GPU
if opt.gpu > 0 then
  input = input:cuda()
  net:cuda()
end

-- eval
local acc = 0
local counter = 0
local maxiter = math.floor(math.min(data:size(), opt.ntest) / opt.batchSize)
local outputs, labels
confusion = optim.ConfusionMatrix({-1,1})

for iter = 1, maxiter do
  collectgarbage()
  
  local data_im,data_label = data:getBatch()

  input:copy(data_im)
  local output = net:forward(input)
  print(output:view(8,8)) 
  outputs = (iter==1) and output or outputs:cat(output,1)
  labels = (iter==1) and data_label or labels:cat(data_label,1)

  
  output:apply(function(x)
  		      l = -1
        	      if x > 1 then
                          l = 1
                      end
                      return l
                      end);

  --for i = 1,opt.batchSize do
      --confusion:add(output[i], data_label[i])
  --end
  acc = acc + output:eq(data_label:cuda()):sum()
  counter = counter + opt.batchSize 
end

torch.save("preds.t7", {outputs:float(), labels})

print(('Summary  %s \t Accuracy: %.4f'):format(opt.model, acc/counter))
--print(confusion)
