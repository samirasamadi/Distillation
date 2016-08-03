-- This should contain a function which gets the feature vector as input and output the soft and hard labels for it

require 'image'
require 'cudnn'
require 'cunn'
	
function featureTolabel(featureVector)
	
	-- load the test featureVector
	--local featureVectors = torch.load('trainFeatures.dat')
	--local featureVector = featureVectors[1][1]
	--print('featureVector', featureVector)
	
	
	-- load the model
	model_path = "logs/vgg/trainedModel.net"
	
	local model = torch.load(model_path)
	
	-- model definition should set numInputDims
	-- hacking around it for the moment
	local view = model:findModules('nn.View')
	if #view > 0 then
	  view[1].numInputDims = 3
	end
	
	--print(model)
	
	local model2 = model:get(54)
    model2:add(nn.SoftMax())
    model2:cuda()
	
    --print(model2)
    
	local softLabels_feature = model2:forward(featureVector:view(1,512))
	softLabels_feature = torch.reshape(softLabels_feature, 10, 1)
	local max = torch.max(softLabels_feature, 1)
	local hardLabel_feature  = 1
	
	for i = 1, 10 do
		if torch.eq(softLabels_feature[i], max) then
			hardLabel_feature = i
		end
	end

	-- print(featureLabels)
	local output = {softLabels_feature, hardLabel_feature}
	
	return output
	
end

