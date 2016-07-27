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
-- local image_paths = opt.dir



-- loads the training data
-- require 'provider'
local provider = torch.load 'provider.t7'

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
-- provider.trainData.data is a floatTensor

 length = provider.trainData.data:size(1):long()
 print(length)

-- local inputs = provider.trainData.data:index(1, length)
-- print('inputs:size()', inputs:size())
-- local targets = provider.trainData.labels:index(1, length)
-- print('targets:size()', targets:size())
-- local outputs = model:forward(inputs)

-- normalize a given image

local function normalize(imgRGB)

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7)):float()

  -- rgb -> yuv
  local yuv = image.rgb2yuv(imgRGB)
  -- normalize y locally:
  yuv[1] = normalization(yuv[{{1}}])

  -- normalize u globally:
  local mean_u = provider.trainData.mean_u
  local std_u = provider.trainData.std_u
  yuv:select(1,2):add(-mean_u)
  yuv:select(1,2):div(std_u)
  -- normalize v globally:
  local mean_v = provider.trainData.mean_v
  local std_v = provider.trainData.std_v
  yuv:select(1,3):add(-mean_v)
  yuv:select(1,3):div(std_v)

  return yuv
end

local model = torch.load(model_path)
model:add(nn.SoftMax():cuda())
model:evaluate()

-- model definition should set numInputDims
-- hacking around it for the moment
local view = model:findModules('nn.View')
if #view > 0 then
  view[1].numInputDims = 3
end

local cls = {'airplane', 'automobile', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

features = {}
-- for file in paths.files(opt.dir) do
--	local path = "test/"..file
--	print(file)

for input in provider.trainData.data do
	print(input)
	print('************')
  -- load image
  local img = image.load(path, 3, 'float'):mul(255)

  -- resize it to 32x32
  img = image.scale(img, 32, 32)
  -- normalize
  img = normalize(img)
  -- make it batch mode (for BatchNormalization)
  img = img:view(1, 3, 32, 32)
  
  -- get features
  
  local feature = model:get(53):forward(img:cuda()):squeeze() 
  print(feature:size())
  
  
end
