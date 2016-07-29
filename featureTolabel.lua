-- This should contain a function which gets the feature vector as input and output the soft and hard labels for it
require 'image'
require 'cudnn'
require 'cunn'

--local function featureTolabel(featureVector){
	
	-- load the test featureVector
	local featureVectors = torch.load('trainFeatures.dat')
	local featureVector = featureVectors[1][1]
	print(featureVector)
	
	-- load the model
	model_path = "logs/vgg/trainedModel.net"
	
	local model = torch.load(model_path)
	
	for i = 1, 53 do
		model:remove(1) 
	end
	
	model:add(nn.SoftMax():cuda())
	model:evaluate()
	
	print(model)
	
	local softLabels = model:forward(featureVector:cuda()):squeeze()
	print(softLabels)
	--}
--end

