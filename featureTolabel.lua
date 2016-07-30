-- This should contain a function which gets the feature vector as input and output the soft and hard labels for it
require 'image'
require 'cudnn'
require 'cunn'

--local function featureTolabel(featureVector){
	
	-- load the test featureVector
	local featureVectors = torch.load('trainFeatures.dat')
	local featureVector = featureVectors[1][1]
	print('featureVector', featureVector)
	
	
	-- load the model
	model_path = "logs/vgg/trainedModel.net"
	
	local model = torch.load(model_path)
	model:add(nn.SoftMax():cuda())
	model:evaluate()
	
	-- model definition should set numInputDims
	-- hacking around it for the moment
	local view = model:findModules('nn.View')
	if #view > 0 then
	  view[1].numInputDims = 3
	end
	
	print(model)
	print('**************')
	
	for i = 1, 53 do
		model:remove(1) 
	end
	
	print(model)
	
	local softLabels = model:forward(featureVector:cuda()):squeeze()
	print(softLabels)
	--}
--end

