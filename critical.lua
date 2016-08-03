--1 choose two feature vectors from the saved file and do binary serach on them to find the critical point between them. This includes passing the midpoint to featureTolabel function each time and getting the soft and hard lables of the points using that function.

require 'cudnn'
require 'cunn'

opt = lapp[[
--model                    (default "logs/vgg/trainedModel.net")     model address
--trainSize                (default 50)                           size of training set
]]

print(opt.model)

if #arg < 2 then
  io.stderr:write('Usage: th imageTofeature.lua [MODEL] [Size of training set]...\n')
  os.exit(1)
end

local model_path = opt.model

points = torch.load('points.dat')
-- points is a table. Each row of the table has three components: featureTensor, softLabels, hardLabel. Use points[i][1] to get feature vector of the ith training point 


length = opt.trainSize
print('length', length)

	
criticalPoints = torch.zeros(1, length^2)
maxIterations = 5
k = 0
	
for i = 1, length do
	for j = i+1, length do
		print(i, j)
	    local feature_x = points[i][1]
		print(feature_x):nDimension()
		local feature_y = points[j][1]
		print(feature_y):dim()
		
		local hardlabel_x = points[i][3]
		local hardlabel_y = points[j][3]
		
		if torch.ne(hardlabel_x, hardlabel_y) then
			k = k + 1
		end
		
		iterationsNum = 0
		while torch.ne(hardlabel_x, hardlabel_y) or iterationsNum < maxIterations do
			
			local feature_mid = .5*(feature_x+feature_y)
			local hardlabel_mid = featureTolabel(feature_mid)[2]
			-- the output of featureTolabel is two dimensional. The first dimension is the soft label and the second dimension is the hard label for the feature vector. The hard label is just the index with maximum value in soft label.
			
			if torch.ne(hardlabel_x, hardlabel_mid) then
				feature_y = feature_mid
			else
				feature_x = feature_mid
			end	
					
			iterationsNum = iterationsNum + 1
			 
		end
		
		criticalPoints[1][k] = .5*(feature_x+feature_y) 
			
	end
end			 
