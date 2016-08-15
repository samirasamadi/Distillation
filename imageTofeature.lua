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

if #arg < 1 then
  io.stderr:write('Usage: th imageTofeature.lua [MODEL]...\n')
  os.exit(1)
end

model_path = opt.model


-- loads the training data
provider = torch.load 'provider.t7'

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
--provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()
-- provider.trainData.data is a floatTensor
indices = torch.randperm(provider.testData.data:size(1)):long():split(1)


model = torch.load(model_path)
model:add(nn.SoftMax():cuda())
model:evaluate()

-- model definition should set numInputDims
-- hacking around it for the moment
view = model:findModules('nn.View')
if #view > 0 then
  view[1].numInputDims = 3
end

print(c.blue '==>' ..' calculating feature vectors of training set ')

model1 = torch.load(model_path)
model1:add(nn.SoftMax():cuda())


view = model1:findModules('nn.View')
if #view > 0 then
  view[1].numInputDims = 3
end

for j = 1, 2 do 
  model1:remove()
end

model1:evaluate()


testPoints_table = {}
num = 1
for t,v in ipairs(indices) do
    local input = provider.testData.data:index(1,v)
	-- floatTensor of size 1*3*32*32
   
    local hardLabel = provider.testData.labels:index(1,v)
    -- DoubleTensor of size 1
	
	local softLabels = model:forward(input:cuda()):squeeze():double()
	-- CudaTensor of size 10
	
	softLabels = torch.reshape(softLabels, 10, 1)
	
	local max = torch.max(softLabels, 1)
	-- local hardLabel = 1
		
	-- Saving hardLabels generated by model
	--for i = 1, 10 do
	--	if torch.all(torch.eq(softLabels[i], max)) then
	--		hardLabel = i
	--	end
	--end 
	
    local featureTensor = model1:forward(input:cuda()):squeeze():double()
	
    -- cudaTensor of size 512
	
	local tmp = {}
	table.insert(tmp, featureTensor:clone())
    table.insert(tmp, softLabels:clone())
	table.insert(tmp, hardLabel)
	
    -- save this information in an array. Each row is the feature vector + the label for it
	table.insert(testPoints_table, tmp)
	
	num = num + 1	
	
end

print(c.blue '==>' ..' saving feature vectors of training set ')
torch.save ('testPoints_table.dat', testPoints_table)
print('finish saving')
