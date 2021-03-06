--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

-- Heavily moidifed by Carl to make it simpler

require 'torch'
require 'image'
require 'hdf5'
tds = require 'tds'
torch.setdefaulttensortype('torch.FloatTensor')
local class = require('pl.class')

local dataset = torch.class('dataLoader')

-- this function reads in the data files
function dataset:__init(args)
  for k,v in pairs(args) do self[k] = v end

  assert(self.labelDim > 0, "must specify labelDim (the dimension of the label vector)")
  --assert(self.labelName, "must specify labelName (the variable in the labelFile to read)")
  assert(self.labelFile, "must specify labelFile (the hdf5 to load)")
  
  local labelFile
  if args.split == 'val' then
      labelFile = args.labelFile_val
  else
      labelFile = args.labelFile
  end
  
  local fd = hdf5.open(labelFile, 'r')
  self.shots = fd:read('/shots'):all()
  self.scenes = fd:read('/scenes'):all()

  if (args.split == 'val') then
    self.val_data = self.shots:cat(self.scenes, 1)
    self.val_label = torch.Tensor(self.shots:size(1)):fill(1):cat(torch.Tensor(self.scenes:size(1)):fill(2),1)
    assert(self.val_data:size(1) == self.val_label:size(1))
  end
  
  fd:close()

  print('found ' .. self.shots:size(1) .. ' shots ' .. self.scenes:size(1) .. ' scenes')
end

function dataset:size()
  return (self.shots:size(1) + self.scenes:size(1))
end

-- converts a table of samples (and corresponding labels) to a clean tensor
function dataset:tableToOutput(dataTable, labelTable, extraTable)
   local data, scalarLabels, labels
   local quantity = #labelTable
   assert(dataTable[1]:dim() == 3)
   local data = torch.Tensor(quantity, 3, self.fineSize, self.fineSize):fill(0)
   local scalarLabels = torch.Tensor(quantity, self.labelDim):fill(-1111)
   local randPerm = torch.randperm(#dataTable)

   for i=1,#dataTable do
      local idx = randPerm[i]
      data[i] = dataTable[idx]
      scalarLabels[i] = labelTable[idx]
   end
   --torch.save('trainData.t7', {data, scalarLabels, extraTable})
   
   return data, scalarLabels, extraTable
end

-- sampler, samples with replacement from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local dataTable = {}
   local labelTable = {}
   local extraTable = {}
   for i=1,quantity do
      local out = torch.Tensor(3, self.fineSize, self.fineSize)
      local data_label

      if i>quantity/2 then
          local idx = torch.random(1, self.scenes:size(1))
 	  out = self.scenes[idx]
          data_label = 2
      else	
          local idx = torch.random(1, self.shots:size(1))
	  out = self.shots[idx]
          data_label = 1
      end

      out = self:trainHook(out) 
      table.insert(dataTable, out)
      table.insert(labelTable, data_label)
      table.insert(extraTable, data_label)
   end
   return self:tableToOutput(dataTable,labelTable,extraTable)
end

-- gets data in a certain range
function dataset:get(start_idx,stop_idx)
   assert(start_idx)
   assert(stop_idx)
   assert(start_idx<stop_idx)
   local count = 1
   local dataTable = {}
   local labelTable = {}
   local extraTable = {}
   assert(start_idx<=self.val_data:size(1))
   local out = torch.Tensor(1+stop_idx-start_idx, 3, self.fineSize, self.fineSize)
   local labels = torch.Tensor(1+stop_idx-start_idx)

   for idx=start_idx,stop_idx do
      if idx > self.val_data:size(1) then
        break
      end

      local outt = self.val_data[idx]
      out[count] = self:trainHook(outt) 
      labels[count] = self.val_label[idx]
      count = count + 1
   end
   return out, labels, extraTable
end

-- function to load the image, jitter it appropriately (random crops etc.)
function dataset:trainHook(input)
   collectgarbage()
   --local input = self:loadImage(path)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = self.fineSize
   local oH = self.fineSize 
   local h1
   local w1
   if self.cropping == 'random' then
     h1 = math.ceil(torch.uniform(1e-2, iH-oH))
     w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   elseif self.cropping == 'center' then
     h1 = math.ceil((iH-oH)/2)
     w1 = math.ceil((iW-oW)/2)
   else
     assert(false, 'unknown mode ' .. self.cropping)
   end
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)

   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out) end
   --out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

    -- subtract mean
    --[[for c=1,3 do
      out[{ c, {}, {} }]:add(-self.mean[c])
    end]]

   return out
end

-- reads an image disk
-- if it fails to read the image, it will use a blank image
-- and write to stdout about the failure
-- this means the program will not crash if there is an occassional bad apple
function dataset:loadImage(path)
  local ok,input = pcall(image.load, path, 3, 'float') 
  if not ok then
     print('warning: failed loading: ' .. path)
     input = torch.zeros(3, opt.loadSize, opt.loadSize) 
  else
    local iW = input:size(3)
    local iH = input:size(2)
    if iW < iH then
        input = image.scale(input, opt.loadSize, opt.loadSize * iH / iW)
    else
        input = image.scale(input, opt.loadSize * iW / iH, opt.loadSize) 
    end
  end

  return input
end

-- data.lua expects a variable called trainLoader
trainLoader = dataLoader(opt)
