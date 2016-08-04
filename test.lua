mytable = {}

for i=1, 3 do
	a = torch.Tensor(3):fill(i)
	b = torch.Tensor(1):fill(i*i)
	tmp = {a, b
	table.insert(mytable, tmp)		
end

print(mytable[1][2])
print(mytable[2][2])
print(mytable[3][2])