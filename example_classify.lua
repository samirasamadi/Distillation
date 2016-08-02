require 'image'
require 'cudnn'
require 'cunn'
local tablex = require 'pl.tablex'

if #arg < 2 then
  io.stderr:write('Usage: th example_classify.lua [MODEL] [FILE]...\n')
  os.exit(1)
end
for _, f in ipairs(arg) do
  if not paths.filep(f) then
    io.stderr:write('file not found: ' .. f .. '\n')
    os.exit(1)
  end
end

local model_path = arg[1]
local image_paths = tablex.sub(arg, 2, -1)

-- loads the normalization parameters
require 'provider'
local provider = torch.load 'provider.t7'

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

for _, img_path in ipairs(image_paths) do
  -- load image
  local img = image.load(img_path, 3, 'float'):mul(255)

  -- resize it to 32x32
  img = image.scale(img, 32, 32)
  -- normalize
  img = normalize(img)
  -- make it batch mode (for BatchNormalization)
  img = img:view(1, 3, 32, 32)


  -- get probabilities
  
  print(model)
  print('************')
  
  
  model1 = torch.load(model_path)
  model1:add(nn.SoftMax():cuda())
  model1:evaluate()

  view = model1:findModules('nn.View')
  if #view > 0 then
    view[1].numInputDims = 3
  end
  
  
  for j = 1, 2 do 
	  model1:remove()
  end
  
  print(model1)
  print('************')
 
  local features = model1:forward(img:cuda()):squeeze()
  print('features', features)
  print('************')
  
  
  local output = model:forward(img:cuda()):squeeze()
  print('original output', output)
  print('************')
  
  for i = 1, 53 do
   	model:remove(1) 
  end
  
  print(model)
  print('*****TILL HERE*******')
  
  
  --print(model:get(1):get(1))
  
  local output2 = model:forward(torch.reshape(features, 512, 1))
  print('ssssss')
  output2 = model:get(1):get(2):forward(output2)
  output2 = torch.reshape(output2, 512, 1)
  output2 = model:get(1):get(3):forward(output2)
  print('after layer 3')
  output2 = model:get(1):get(4):forward(output2)
  print('after layer 4')
  output2 = model:get(1):get(5):forward(output2)
  print('after layer 5')
  print(output2)
  output2 = torch.reshape(output2, 512)
  output2 = model:get(1):get(6):forward(output2)
   print('after layer 6')
  local finalOutput = model:get(2):forward(output2)
  
  print('finalOutput', finalOutput)

  -- display
  --print('Probabilities for '..img_path)
  --for cl_id, cl in ipairs(cls) do
  --  print(string.format('%-10s: %-05.2f%%', cl, output[cl_id] * 100))
  --end
end
