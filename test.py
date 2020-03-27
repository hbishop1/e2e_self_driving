def f(**kwargs):
    return reduce(lambda x,y: x+y, kwargs.iterkeys(),'')

print(f(org = 'met1',test='python'))
