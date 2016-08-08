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
	local model_path = "logs/vgg/trainedModel.net"
	
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
	model2:evaluate()
	
	local softLabels_feature = model2:forward(featureVector:view(1,512))
	
	softLabels_feature = torch.reshape(softLabels_feature, 10, 1)
	local max = torch.max(softLabels_feature, 1)
	local hardLabel_feature = 1
	
	for i = 1, 10 do
		if torch.all(torch.eq(softLabels_feature[i], max)) then
			hardLabel_feature = i
		end
	end

	local output = {softLabels_feature, hardLabel_feature}
	
	return output
	
end

------------------------------------------------------------------------------------------------

opt = lapp[[
--trainSize                (default 5)                           size of training set
]]

if #arg < 1 then
  io.stderr:write('Usage: th ciritcsl.lua [Size of training set]...\n')
  os.exit(1)
end

model_path = opt.model

points = torch.load('points_table.dat')
-- points is a table. Each row of the table has three components: featureTensor, softLabels, hardLabel. Use points[i][1] to get feature vector of the ith training point 

length = opt.trainSize
print('length', length)
	
criticalPoints = {}
criticalSoftLabels = {}
output = {}

maxIterations = 3
k = 0

print(c.blue '==>' ..' calculating critical points ')
	
for i = 1, length do
	for j = i+1, length do
		
	    feature_x = points[i][1]:clone()
		feature_y = points[j][1]:clone()	
		
		hardlabel_x = points[i][3]
		hardlabel_y = points[j][3] 
		
		softlabel_x = points[i][2]:clone()
		softlabel_y = points[j][2]:clone()
		
		--print('softlabel_x\n', softlabel_x)
		--print('hardlabel_x\n', hardlabel_x)
		--print('*************************')
		--print('softlabel_y\n', softlabel_y)
		--print('hardlabel_y\n', hardlabel_y)
		--print('*************************')
		--print('*************************')
		
		
		-- print('before if')
		if hardlabel_x ~= hardlabel_y then
			-- print('inside if')
			k = k+1
			
			iterationsNum = 0
			while ( hardlabel_x ~= hardlabel_y and iterationsNum < maxIterations ) do
			
				tmp =  feature_x + feature_y
		    	feature_mid = tmp:clone()
				feature_mid:cmul(torch.Tensor(512):fill(.5):cuda())
				
			
				softlabel_mid = featureTolabel(feature_mid)[1]:clone()
				hardlabel_mid = featureTolabel(feature_mid)[2]
				-- the output of featureTolabel is two dimensional. The first dimension is the soft label and the second dimension is the hard label for the feature vector. The hard label is just the index with maximum value in soft label.
				
				
				if hardlabel_x ~= hardlabel_mid then
					feature_y = feature_mid:clone()
				else
					feature_x = feature_mid:clone()
				end	
					
				iterationsNum = iterationsNum + 1	
				 
			end
			
			criticalPoints[k] = feature_mid:clone()
			criticalSoftLabels[k] = featureTolabel(feature_mid)[1]
		
			table.insert(output, {criticalPoints[k], criticalSoftLabels[k]})	
		end
		-- print('after if')
	end
	
end	
print('k', k)
print('Printing values')
print(output[1][2])
print(output[2][2])
print(output[3][2])
print(output[4][2])
print(output[5][2])
print(output[6][2])

		 
