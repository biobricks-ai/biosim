import logging

def collect(ds,fn = lambda i,x: i+1):
    for i,x in enumerate(ds):
        yield fn(i,x)

def count_and_collect(ds):
    return list(collect(ds))[-1]