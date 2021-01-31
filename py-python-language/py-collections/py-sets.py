print("Understanding Set collection")

oddset={item for item in range(1, 15) if(item%2)!=0}
print(oddset)
evenset={item for item in range(1, 15) if(item%2)==0}
print(evenset)

emptyset=set()
print(type(emptyset))

print('Mathematical Set Operations')
print('A intersection B: {0}'.format(oddset & evenset))
print('A union B: {0}'.format(oddset | evenset))
print('A - B: {0}'.format(oddset-evenset))
print('A subset B: {0}'.format(oddset <= evenset))
print('A superset B: {0}'.format(oddset >= evenset))
