import json, pathlib
import classes.verifai as verifai

class Vouchers:
  def __init__(self):
    self.configFile = str(pathlib.Path().resolve()) + '/classes/config.json'
    
    with open(self.configFile, 'r') as configFile:
      self.config = json.load(configFile)
    # Testing Purpose only
    self.changeImgPath(str(pathlib.Path().resolve()) + '/samples')
    #self.__isValid = False
         
  def __pathCheck(self):
    if self.config['path'] == '':
      return str(pathlib.Path().resolve()) + '/classes'
    else:
      return self.config['path']
  
  def retrieveImgPath(self):
    return self.__pathCheck()

  def changeImgPath(self, path):
    self.config['path'] = path
    try:
      with open(self.configFile, 'w') as configFile:
        json.dump(self.config, configFile)
        return True
    except:
      return False

  def voucherCount(self):
    return 0

  def regonize(self, name, absolutePath=False):
    imgFile=''
    if absolutePath:
      imgFile = name
    else:
      imgFile = self.__pathCheck() + '/' + name
    try:
      return verifai.verifaiCheck(imgFile)
    except Exception as e:
      return {'isValid': False, 'error': str(e)}
