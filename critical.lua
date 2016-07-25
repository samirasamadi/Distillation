local function findCriticalPoints(points, labels, f){
	-- Points are passed to the function as feature vectors
	-- f is the function that converts any feature vector to soft labels and then outputs the label with highest probability as hard label 
	
	length = points:size(2)
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
				
				if torch.ne(label_x, f(mid)) then
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