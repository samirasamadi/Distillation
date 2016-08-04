mytable = {}

for i=1, 3 do
	
	tmp = {i, i*i}
	table.insert{mytable, tmp}
		
end

print(mytable)