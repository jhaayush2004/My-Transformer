# Importing packages and modules
import numpy as np
import math

# L:Length of input sequence, d_k:Query dimension, d_v:value dimension, d_q:Query dimension
L,d_k,d_v,d_q=10,8,8,8

#generating random query, key and value matrices
q=np.random.randn(L, d_q)
k=np.random.randn(L, d_k)
v=np.random.randn(L, d_v)

#calculating dot product of queries and keys matrices giving us the similarity among their respective input tokens.
np.matmul(q, k.T)

#scaling by sqrt(d_k) to maintain numerical stability of calculated attention score matrix or simply to normalize it.
scaled = np.matmul(q, k.T)/math.sqrt(d_k)

#masking is mostly applied in decoder part to prevent data leakage or to prevent availability of future results to decoder.
#Generates a lower triangular mask matrix to prevent the decoder from having access to future information, thereby preventing data leakage.
mask = np.tril((np.ones((L,L))))
mask[mask==0]=-np.infty
mask[mask==1]=0

# scaled and mask are added to prevent data leakage .
scaled+mask

# softmax function 
def softmax(x):
    return (np.exp(x).T/np.sum(np.exp(x),axis=-1)).T

#applying softmax on scaled+mask makes all future results zero and thus making it unavailable for decoder.
#Applies the softmax function to obtain normalized attention scores, ensuring that the attention weights sum up to 1.
attention = softmax(scaled+mask)

#This new_v is final attention matrix.Calculates the weighted sum of values using the attention scores to obtain the final attention matrix.
new_v =  np.matmul(attention, v)
