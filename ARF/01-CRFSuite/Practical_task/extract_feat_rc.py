from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import PIL
from PIL import ImageOps

def main():
  if len(sys.argv) != 3:
    print("Usage: python extract_feat.py png-image-file png-label-file")
    sys.exit(0)

  imgfile = sys.argv[1]
  labfile = sys.argv[2]

  img = PIL.Image.open(imgfile)
  imgarr = np.array(img)
  lab = PIL.Image.open(labfile)
  labarr = np.array(lab)
  normv = (np.min(labarr) + np.max(labarr))/2

  #print(imgarr.shape, labarr.shape)

  (nr, nc) = imgarr.shape
  if nr != labarr.shape[0] or nc != labarr.shape[1]:
    print("Image and labels dimensions do not match")
    sys.exit(0)

  ws = 25
  for r in range(0,nr,ws):
    for c in range(0,nc,ws):
      w = imgarr[r:r+ws,c:c+ws]
      g=int(np.round(np.mean(w)))
      lw = labarr[r:r+ws,c:c+ws]
      l = int(np.round(np.mean(lw)))
      if l < normv:
        l=0
      else:
        l=1
      print(l,"\tg[0][0]=",g,"\tr[0][0]=",r,"\tc[0][0]=",c,sep='')

if __name__ == '__main__':
    main()

