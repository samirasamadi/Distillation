require 'cudnn'
require 'cunn'

--1 choose two feature vectors from the saved file and do binary serach on them to find the critical point between them. This includes passing the midpoint to featureTolabel function each time and getting the soft and hard lables of the points using that function.


points = torch.load('trainFeatures.dat')
-- points is a table. Each row of the table has three components: featureTensor, softLabels, hardLabel. Use points[i][1] to get feature vector of the ith training point 


--length = points:size(2)
print (points)
	
criticalPoints = torch.zeros(1, length^2)
maxIterations = 5
k = 0
	
for i = 1, length do
	for j = i+1, length do

	    feature_x = points[i][1]
		feature_y = points[j][1]
		
		hardlabel_x = points[i][3]
		hardlabel_y = points[j][3]
		
		if torch.ne(hardlabel_x, hardlabel_y) then
			k = k + 1
		end
		
		iterationsNum = 0
		while torch.ne(hardlabel_x, hardlabel_y) or iterationsNum < maxIterations do
			
			feature_mid = .5*(feature_x+feature_y)
			hardlabel_mid = featureTolabel(feature_mid)[2]
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
