require 'image'
require 'cudnn'
require 'cunn'
require 'xlua'
require 'optim'
require 'nn'
dofile './provider.lua'
local c = require 'trepl.colorize'
local tablex = require 'pl.tablex'


opt = lapp[[
--model                    (default "logs/vgg/trainedModel.net")     model address
]]

print(opt.model)

if #arg < 2 then
  io.stderr:write('Usage: th example_classify.lua [MODEL] [FILE]...\n')
  os.exit(1)
end

local model_path = opt.model


-- loads the training data
local provider = torch.load 'provider.t7'

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
-- provider.trainData.data is a floatTensor
local indices = torch.randperm(provider.trainData.data:size(1)):long():split(1)


local model = torch.load(model_path)
model:add(nn.SoftMax():cuda())
model:evaluate()

-- model definition should set numInputDims
-- hacking around it for the moment
local view = model:findModules('nn.View')
if #view > 0 then
  view[1].numInputDims = 3
end

print(c.blue '==>' ..' calculating + saving feature vectors of training set ')

array = {}
num = 1
for t,v in ipairs(indices) do
    local input = provider.trainData.data:index(1,v)
	-- floatTensor of size 1*3*32*32
   
    local target = provider.trainData.labels:index(1,v)
    -- DoubleTensor of size 1
  
    local output = model:get(53):forward(input:cuda()):squeeze()
    -- output os a cudaTemsor of size 6*512
    
    -- save this information in a table. Each row is the feature vector + the label for it
	
	--print('here')
	array[num] = {output, target}
	print(table[num])
    break
end
  
  
