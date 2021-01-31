print('Understanding Boolean datatype')

a, b, c, d = 0, 1, 42, -3
print(' '.join([str(bool(a)), str(bool(b)), str(bool(c)), str(bool(d))]))
print('Boolean value for empty list: ', bool([]))
print('Boolean value for empty string: ', bool(""))
print('Boolean value for Non-Empty String', bool("Hi"))