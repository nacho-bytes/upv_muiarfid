#!/bin/bash

## Take all training files and obtaining the CRFSuite features datafile

for f in images/b*.png ; do                              ## For each training image (b*); for test it would be d*
  b=`echo "$f" | sed "s/^.*\///g" | sed "s/.png//g"`     ## Obtain filename without extension (e.g., images/b01-000.png -> b01-000)
  python extract_feat.py $f images_lines/$b.line.png     ## Extract features (mean gray level); for features with position, use extract_feat_rc.py
  echo ""                                                ## Blank line between consecutive images
done > train.txt                                         ## CRFSuite features for training; for test change file name
