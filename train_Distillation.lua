require 'xlua'
require 'optim'
require 'nn'
dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 50)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
   --trainSize               (default 1000)          How many critical points do we want to train the network on?
   --testSize                (default 100)           How many test points do we want to evaluate the network on?
]]

print(opt)


local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))

-- here it goes inside vgg_bn_drop
model:add(cast(dofile('models/'..opt.model..'.lua')))
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end

local model2 = model:get(2):get(54)
model2:add(nn.SoftMax())
model2:cuda()
print(model2)

print(c.blue '==>' ..' loading data')
trainPoints = torch.load('criticalPoints_feature.dat')
-- criticalPoints is a table. At each row: the first element is the feature vector for that critical point and the second element is soft label of that critical point.
testPoints = torch.load('testFeature_originalLabels.dat')
-- testPoints is a table. The first element is the feature vector, the second element is softLabels, the third element is hardLabels

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test_Distillation.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model2:getParameters()

print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.CrossEntropyCriterion())


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  
  print(c.blue '==>'.." online epoch # " .. epoch ..']')


  local indices = torch.randperm(opt.trainSize)
  -- indices is a torch double Tensor of size 1000 (1000*1)
  

  local tic = torch.tic()
  -- ipairs do a single iteration over elements of the array (here indices)
	  
  for k = 1, opt.trainSize do
	  
	  index = indices[k]
	  
  	
	-- it's inputs and not input and targets and not target since we might take batches of input for gradient descent.
	local inputs =  trainPoints[index][1]:clone()
	-- input is torch.CudaTensor of size 512 (512*1)
	
	local targets = trainPoints[index][2]:clone()
	-- targets is torch.CudaTensor of size 10x1

    local feval = function(x)
	
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
	  
      
      local outputs = model2:forward(inputs:view(1,512))
	  outputs = torch.reshape(outputs, 10, 1):cuda()
	  -- outputs is a torch.CudaTensor of size 10x1
	  
      local f = criterion:forward(outputs, targets)
	  print('f is', f)
      local df_do = criterion:backward(outputs, targets)
	  print(df_do)
	  
      model2:backward(inputs, df_do)
	  
	  confusion:batchAdd(outputs, targets)
	  print('after batchAdd')
      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model2:evaluate()
  print(c.blue '==>'.." testing")
  
  for i=1, opt.testSize do
    local outputs = model2:forward(testpoints[i][3])
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      do
        os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
        os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
        local f = io.open(opt.save..'/test.base64')
        if f then base64im = f:read'*all' end
      end

      local file = io.open(opt.save..'/report.html','w')
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]]):format(opt.save,epoch,base64im))
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end
      file:write'</table><pre>\n'
      file:write(tostring(confusion)..'\n')
      file:write(tostring(model)..'\n')
      file:write'</pre></body></html>'
      file:close()
    end
  end

-- save model every 50 epochs
  if epoch % 1 == 0 then
    local trainedModel = paths.concat(opt.save, 'trainedModel.net')
    print('==> saving model to '..trainedModel)
    torch.save(trainedModel, model:get(3):clearState())
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()
end


