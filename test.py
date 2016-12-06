import numpy as np

vfile = 'vectors.txt'

size =  2604459
emb = np.random.random((size, 50))
c = 0

i2w = {}
w2i = {}

with open(vfile) as f:
  for l in f:
    if c >= size:
      break
    
    if c % (int(size/100)) == 0:
      print(c)

    vectors = l.split(' ')
    i2w[c] = [vectors[0]]
    w2i[vectors[0]] = c
    emb[c] = np.array(vectors[1:])
    c+=1
   
def cos_sim(embedding, index):
   row = embedding[index]
   mul = row.dot(embedding.T)
   row_norm = np.linalg.norm(row)
   mul = mul / row_norm
   size = np.linalg.norm(embedding, axis=1) 
   mul = mul / size.T 
   return mul.T 


def most_sim(weight, index):
  sims = cos_sim(weight, index)
  most_sims = sims.argsort()[::-1][:5]
  k = [(i, sims[i]) for i in most_sims]
  return k

for i in range(1000):
  a = most_sim(emb, i)
  words = [i2w[j[0]][0].replace('%', ' ') for j in a]
  print(words[0], ': ', words[1:])
