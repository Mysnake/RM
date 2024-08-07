from taylornet1 import TaylorLinearNet as TaylorNet
# from taylornet2 import TaylorLinearNet as TaylorNetShare
from taylornet2 import TaylorLinearNet2 as TaylorNetShare


def gettaylormodel(no_share=True,inputsize=None,outputsize=None,d=None):
    return TaylorNet(inputsize, outputsize, d) if no_share else TaylorNetShare(inputsize,outputsize,d)

