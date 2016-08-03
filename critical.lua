
--1 choose two feature vectors from the saved file and do binary serach on them to find the critical point between them. This includes passing the midpoint to featureTolabel function each time and getting the soft and hard lables of the points using that function.

	
local function findCriticalPoints(points, labels){
	-- Points are passed to the function as feature vectors
	-- f is the function that converts any feature vector to soft labels and then outputs the label with highest probability as hard label 
		
	-- load feature vectors of training set already saved in file
	--print(loadedArray[1][1]) 
	

	trainedFeatureVectors = torch.load('trainFeatures.dat')
	
	length = points:size(2)
	print('*****', length, '*****')
	
	criticalPoints = torch.zeros(1, length^2)
	maxIterations = 5
	k = 0
	
	for i = 1, length do
		for j = i+1 in length do
		
			x = points[1][i]
			y = points[1][j]
			
			label_x = labels[i]
			label_y = labels[j]
			
			if torch.ne(label_x, label_y) then
				k = k + 1
			end
			
			iterationsNum = 0
			while torch.ne(label_x, label_y) or iterationsNum < maxIterations then	
				
				mid = .5*(x+y)
				label_mid = featureTolabel(mid)[2]
				
				if torch.ne(label_x, label_mid) then
					y = mid
				else
					x = mid
				end	
					
				iterationsNum = iterationsNum + 1
				 
			end
			
			critical[1][k] = .5*(x+y) 
				
		end
	end			
	return critical 
end
}