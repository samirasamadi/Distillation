mytable = {}

for i=1, 3 do
	
	tmp = {i, i*i}
	table.insert(mytable, tmp)
		
end

print(mytable[1][2])
print(mytable[2][2])
print(mytable[3][2])