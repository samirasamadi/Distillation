--1 choose two feature vectors from the saved file and do binary serach on them to find the critical point between them. This includes passing the midpoint to featureTolabel function each time and getting the soft and hard lables of the points using that function.

require 'image'
require 'cudnn'
require 'cunn'
local c = require 'trepl.colorize'

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

------------------------------------------------------------------------------------------------

opt = lapp[[
--model                    (default "logs/vgg/trainedModel.net")     model address
--trainSize                (default 10)                           size of training set
]]

print(opt.model)

if #arg < 2 then
  io.stderr:write('Usage: th imageTofeature.lua [MODEL] [Size of training set]...\n')
  os.exit(1)
end

local model_path = opt.model

local points = torch.load('points.dat')
-- points is a table. Each row of the table has three components: featureTensor, softLabels, hardLabel. Use points[i][1] to get feature vector of the ith training point 


local length = opt.trainSize
print('length', length)

	
local criticalPoints = {}
local criticalSoftLabels = {}
local output = {}

local maxIterations = 5
local k = 0


print(c.blue '==>' ..' calculating critical points ')
	
for i = 1, length do
	for j = i+1, length do
		print(i, j)
	    feature_x = points[i][1]
		-- print('feature_x', feature_x)
		
		-- print('points[i][1]', points[i][1])
		
		feature_y = points[j][1]
		--print('feature_y', feature_y)
		
		hardlabel_x = points[i][3]
		hardlabel_y = points[j][3]
		
		if torch.all(torch.ne(hardlabel_x, hardlabel_y)) then
			k = k + 1
		end
			
		iterationsNum = 0
		while (torch.ne(hardlabel_x, hardlabel_y) and iterationsNum < maxIterations) do
			
			feature_mid = (feature_x+feature_y)
			feature_mid:cmul(torch.Tensor(512):fill(.5):cuda())
			
			-- print(feature_mid)
			hardlabel_mid = featureTolabel(feature_mid)[2]
			-- the output of featureTolabel is two dimensional. The first dimension is the soft label and the second dimension is the hard label for the feature vector. The hard label is just the index with maximum value in soft label.
			
			if torch.ne(hardlabel_x, hardlabel_mid) then
				feature_y = feature_mid:clone()
			else
				feature_x = feature_mid:clone()
			end	
					
			iterationsNum = iterationsNum + 1		 
		end
		
		criticalPoints[k] = feature_mid:clone()
		criticalSoftLabels[k] = featureTolabel(feature_mid)[1]
		output[k] = {criticalPoints[k], criticalSoftLabels[k]}		
	end
	
end	


print(output)
print(output[1][2])
print(output[2][2])


		 
