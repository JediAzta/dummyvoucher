#!/usr/bin/python3

import sys, os, json, pathlib
from classes.vouchers import Vouchers

vouchers = Vouchers()
wwwroot = ''

with open(os.path.join(str(pathlib.Path().resolve()), 'classes/config.json'), 'r') as configFile:
  wwwroot = json.load(configFile)['wwwroot'] 


if len(sys.argv) != 2:
  print('Usage: scan.py [voucher_name]')
  print()
  print('Where voucher_name is the voucher name (with the web sub path stored)')
  exit()
else:
  imgFile = os.path.join(wwwroot, sys.argv[1])
  if not os.path.isfile(imgFile):
    print(sys.argv[1], ' is not exist')
    exit()
  #print('Processing...')
  result = vouchers.regonize(imgFile, True)
  #print(sys.argv[1], ' regonized as below')
  print(result)
  #print()
  #print('Detail of the ticket')
  
  #for k, v in result.items():
   # print(k + ': ' + str(v)) 

