import numpy as np
import cv2 as cv
from imutils.perspective import four_point_transform
import re
import SIFT_TM
from paddleocr import PaddleOCR
import pathlib

#import json
resizeHeight = 520


def initJson():
  initJson = {
    'is_valid': False,
    'is_defect': False,
    'have_stub': False,
    'have_mirror_logo': False,
    'have_mkv_logo': False,
    'is_a4_document': False,
    'is_ticket': False,
    'is_out_of_focus': False,
    'is_too_dark': False,
    'quality_issue': '',
    'location1': '',
    'location2': '',
    'date': '',
    'weekday': '',
    'time': '',
    'price': '',
    'gate': '',
    'section': '',
    'row': '',
    'seat': '',
    'ticket_x1': 0,
    'ticket_x2': 0,
    'ticket_y1': 0,
    'ticket_y2': 0,
    'qrcode_x1': 0,
    'qrcode_x2': 0,
    'qrcode_y1': 0,
    'qrcode_y2': 0,
    'pii_x1': 0,
    'pii_x2': 0,
    'pii_y1': 0,
    'pii_y2': 0,
  }
  return initJson


# this func is to locate the ticket information position
#Due to performance, noise may affect the location
#TODO if allow use pytesseract find the text, can remove the noise
# not for image correction
def TicketPos(image):
  #  change the color space to YUV
  image_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)

  # grap only the Y component
  image_y = np.zeros(image_yuv.shape[0:2], np.uint8)
  image_y[:, :] = image_yuv[:, :, 0]
  # blur the image to reduce high frequency noises
  image_blurred = cv.GaussianBlur(image_y, (3, 3), 0)
  # find edges in the image
  edges = cv.Canny(image_blurred, 50, 200, apertureSize=3)
  # find contours
  contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
  # Concatenate all contours
  cnts = np.concatenate(contours)

  # Determine and draw bounding rectangle
  x, y, w, h = cv.boundingRect(cnts)
  #cv.rectangle(image, (x-2, y-2), (x + w + 2, y + h +2), 255, 2)

  return (x, y, x + w, y + h)  #20230428


#this func is to find if any qr code
# if there is qr code return the position x1, y1, x2, y2
def QRCode(image):
  #init QR code detection
  qcd = cv.QRCodeDetector()

  retval, points, straight_qrcode = qcd.detectAndDecode(image)
  #if contain QR code return positioin
  if (points is not None):
    # position of QR code is in points
    x, y, w, h = int(points[0][0][0]), int(points[0][0][1]), int(
      points[0][2][0]), int(points[0][2][1])
    #cv.rectangle(image, (x, y), (w,h), (255,255,255), -1)
    return (x, y, x + w, y + h)
  else:
    #print("No QR Code detected/ Unclear QR code.")
    return None


# map colour names to HSV ranges
color_list = [['red', [0, 160, 70], [10, 250, 250]],
              ['pink', [0, 50, 70], [10, 160, 250]],
              ['yellow', [15, 50, 70], [30, 250, 250]],
              ['green', [40, 50, 70], [70, 250, 250]],
              ['cyan', [80, 50, 70], [90, 250, 250]],
              ['blue', [90, 50, 70], [128, 255, 255]],
              ['purple', [129, 50, 70], [160, 255, 255]],
              ['red', [170, 160, 70], [180, 250, 250]],
              ['orange', [10, 100, 20], [25, 255, 255]],
              ['pink', [170, 50, 70], [180, 160, 250]]]


def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
  # initialize the dimensions of the image to be resized and
  # grab the image size
  dim = None
  (h, w) = image.shape[:2]

  # if both the width and height are None, then return the
  # original image
  if width is None and height is None:
    return image

  # check to see if the width is None
  if width is None:
    # calculate the ratio of the height and construct the
    # dimensions
    r = height / float(h)
    dim = (int(w * r), height)

  # otherwise, the height is None
  else:
    # calculate the ratio of the width and construct the
    # dimensions
    r = width / float(w)
    dim = (width, int(h * r))

  # resize the image
  resized = cv.resize(image, dim, interpolation=inter)

  # return the resized image
  return resized


# this func is to detect which type of ticket is it by code
def detect_main_color(hsv_image, colors):
  color_found = 'undefined'
  max_count = 0

  for color_name, lower_val, upper_val in colors:
    # threshold the HSV image - any matching color will show up as white
    mask = cv.inRange(hsv_image, np.array(lower_val), np.array(upper_val))
    count = np.sum(mask)
    if count > max_count:
      color_found = color_name
      max_count = count

  return color_found


def detectSim(imagePathStr, logoPathStr, minNo=None):
  imageR = cv.imread(imagePathStr, 0)
  imageR = image_resize(imageR, resizeHeight)
  logoR = cv.imread(logoPathStr, 0)
  #logoR = cv.resize(logoR, 200)
  # read images
  temp_img_gray = logoR
  map_img_gray = imageR

  # equalize histograms
  temp_img_eq = cv.equalizeHist(temp_img_gray)
  map_img_eq = cv.equalizeHist(map_img_gray)
  coords = SIFT_TM.get_matched_coordinates(temp_img_eq, map_img_eq, minNo)
  if (len(coords) < 2):
    return None
  else:
    #plt.imshow(SIFT_img)
    #plt.show()
    return coords


def detectLogo(image, logoPathStr, minNo=None):
  imageR = image
  logoR = cv.imread(logoPathStr, 0)
  # read images
  temp_img_gray = logoR
  map_img_gray = imageR

  # equalize histograms
  temp_img_eq = cv.equalizeHist(temp_img_gray)
  map_img_eq = cv.equalizeHist(map_img_gray)
  coords = SIFT_TM.get_matched_coordinates(temp_img_eq, map_img_eq, minNo)
  if (len(coords) < 2):
    #print("No Match")
    return None
  else:
    #plt.imshow(SIFT_img)
    #plt.show()
    return coords


def detectLogoFromSame(image, croppedImage, minNo=None):
  imageR = image
  logoR = croppedImage
  # read images
  temp_img_gray = logoR
  map_img_gray = imageR

  # equalize histograms
  temp_img_eq = cv.equalizeHist(temp_img_gray)
  map_img_eq = cv.equalizeHist(map_img_gray)
  coords = SIFT_TM.get_matched_coordinates(temp_img_eq, map_img_eq, minNo)
  if (len(coords) < 2):
    #print("No Match")
    return None
  else:
    #plt.imshow(SIFT_img)
    #plt.show()
    return coords


def readImage(imgPathStr):
  image = cv.imread(imgPathStr)
  (h, w) = image.shape[:2]
  image = image_resize(image, resizeHeight)
  return image, [
    float(w) / float(w * (resizeHeight / float(h))),
    float(h) / float(resizeHeight)
  ]


def checkHist(image):
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  bright_thres = 0.5
  dark_thres = 0.3
  dark_part = cv.inRange(gray, 0, 30)
  bright_part = cv.inRange(gray, 220, 255)
  # use histogram
  # dark_pixel = np.sum(hist[:30])
  # bright_pixel = np.sum(hist[220:256])
  total_pixel = np.size(gray)
  dark_pixel = np.sum(dark_part > 0)
  bright_pixel = np.sum(bright_part > 0)
  if (dark_pixel / total_pixel > bright_thres):
    return "Image under exposure"
  #print(bright_pixel/total_pixel)
  if bright_pixel / total_pixel > dark_thres:
    return "Image too bright"
  return ''


def sharpen_image(image):
  sharp_img = cv.filter2D(image, -1,
                          np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
  return sharp_img


def canny_segmentation(img, low_threshold=100, high_threshold=200):
  edges = cv.Canny(img, low_threshold, high_threshold)
  return edges


def get_bounding_box(image, thresh=0.95):
  nonzero_indices = np.nonzero(image.T)
  min_row, max_row = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
  min_col, max_col = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
  box_size = max_row - min_row + 1, max_col - min_col + 1
  box_size_thresh = (int(box_size[0] * thresh), int(box_size[1] * thresh))
  #box_size_thresh = (int(box_size[0]), int(box_size[1]))
  #coordinates of the box that contains 95% of the highest pixel values
  top_left = (min_row + int(
    (box_size[0] - box_size_thresh[0]) / 2), min_col + int(
      (box_size[1] - box_size_thresh[1]) / 2))
  bottom_right = (top_left[0] + box_size_thresh[0],
                  top_left[1] + box_size_thresh[1])
  return (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1])


def is_blurry(image,
              thresh=1500,
              crop_edges_thresh=0.75,
              canny_thresh_low=100,
              canny_thresh_high=200):
  if (len(image.shape) < 3):
    gray = image
  else:
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

  gray = sharpen_image(gray)
  seg = canny_segmentation(gray, canny_thresh_low, canny_thresh_high)
  bb_thresh = get_bounding_box(seg, crop_edges_thresh)
  im_crop = gray[bb_thresh[0][1]:bb_thresh[1][1],
                 bb_thresh[0][0]:bb_thresh[1][0]]
  edges = cv.Laplacian(im_crop, cv.CV_64F)
  if edges.var() < thresh:
    return "Image too blurry", edges.var()
  return '', edges.var()


def biggestContour(contours):
  biggest = np.array([])
  max_area = 0
  for i in contours:
    area = cv.contourArea(i)
    if area > 2000:
      peri = cv.arcLength(i, True)
      approx = cv.approxPolyDP(i, 0.02 * peri, True)
      if area > max_area and len(approx) == 4:
        biggest = approx
        max_area = area
  return biggest, max_area


def reorder(myPoints):

  myPoints = myPoints.reshape((4, 2))
  myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
  add = myPoints.sum(1)

  myPointsNew[0] = myPoints[np.argmin(add)]
  myPointsNew[3] = myPoints[np.argmax(add)]
  diff = np.diff(myPoints, axis=1)
  myPointsNew[1] = myPoints[np.argmin(diff)]
  myPointsNew[2] = myPoints[np.argmax(diff)]

  return myPointsNew


#Check A4
def checkA4(image):
  img = image
  imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
  imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
  imgThreshold = cv.Canny(imgBlur, 200, 255)  # APPLY CANNY BLUR
  kernel = np.ones((5, 5))
  imgDial = cv.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
  imgThreshold = cv.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
  contours, hierarchy = cv.findContours(imgThreshold, cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)
  biggest, maxArea = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
  if biggest.size != 0:
    biggest = reorder(biggest)
    rect = cv.minAreaRect(biggest)
    (x, y), (w, h), angle = rect
    aspect_ratio = w / h
    #print(aspect_ratio)
    if (aspect_ratio > 1.4):
      return True
  return False


def detectFace(image):
  img = image
  gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  face_classifier = cv.CascadeClassifier(cv.data.haarcascades +
                                         "haarcascade_frontalface_default.xml")
  face = face_classifier.detectMultiScale(gray_image,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(20, 20))
  #print(face)
  return face


def checkTicketType(image):
  if ((image.shape[1] / image.shape[0]) > 1.8):
    return 'Full'
  else:
    return 'Cut'


def checkMainColor(image):
  hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
  mainColor_img = detect_main_color(hsv, color_list)
  if (mainColor_img == 'purple'):
    ticketFrom = 'CityLine'
    ticketFormat = 3
  elif (mainColor_img == 'orange' or mainColor_img == 'pink'):
    ticketFrom = 'URBTIX'
    ticketFormat = 1
  elif (mainColor_img == 'blue'):
    ticketFrom = 'URBTIX'
    ticketFormat = 2
  else:
    ticketFrom = 'Unknown'
    ticketFormat = 0

  return ticketFrom, ticketFormat


def ocrProccess(image, tType, tFormat):
  x0Per, x1Per, y0Per, y1Per = 0.41, 0.99, 0.25, 0.96
  if (tType == 'Full'):
    if (tFormat == 3):
      x0Per, x1Per = 0.40, 0.975
  elif (tType == 'Cut'):
    x0Per, x1Per, y0Per, y1Per = 0.40, 0.999, 0.01, 0.9

  #not using Tesseract
  #return pytesseract.image_to_string(image[int(image.shape[0] * x0Per):int(image.shape[0] * x1Per),int(image.shape[1] * y0Per):int(image.shape[1] * y1Per)],lang='eng+chi_tra',config='--psm 6 --oem 3')
  ocr = PaddleOCR(
    use_angle_cls=False,
    lang='ch',
    use_gpu=False,
    use_tensorrt=False,
    use_onnx=False,
    #detection_use_dilation = False,
    #classification_enabled = False,
    #classification_threshold = 0.9,
    rec_algorithm='CRNN',
    rec_batch_num=8,
    #drop_score = 0.5,
    show_log=False
  )  #,rec_algorithm='SVTR_LCNet') # need to run only once to download and load model into memory
  result = ocr.ocr(
    image[int(image.shape[0] * x0Per):int(image.shape[0] * x1Per),
          int(image.shape[1] * y0Per):int(image.shape[1] * y1Per)],
    cls=False)
  resStr = []
  for line in result[0]:
    resStr.append(line[1][0])
  return resStr


def getInfo(str, tFormat, toJson):
  #For extracting date time, gate and seats information
  date_pattern_str = r'^\d{4}[1/]\d{2}[1/]\d{2}'
  #date_pattern2_str = r'^\d{4}1\d{2}1\d{2}'
  price_pattern = r'[1-9]?[.,]?\d{3}[.,]?[0-9]?'
  Time_pattern_str = r'[0-9B]?[0-9B]:\d{2}[P]?[MH]?'
  similarGateList = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'VELLOW']
  dGateList = ['RED', 'GREEN', 'BLUE', 'YELLOW']
  dayList = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
  dWCList = ['WCHM', 'WCHU']
  seatPattern = r'[89][0-9]'
  twodigitPattern = r'\d{2}'
  studioNameList = ['HONGKONG', 'COLISEUM', 'HONG', 'KONG ']
  chStudioNameList = ['香港', '红', '館', '馆']
  wcSectionValid = False
  checkedWC = False
  hvGate = False
  hvRow = False
  price = None
  for i in range(0, len(str)):
    if any(ext in str[i] for ext in studioNameList):
      location1 = "HONG KONG COLISEUM"
      toJson['location1'] = location1
    if any(ext in str[i] for ext in chStudioNameList):
      location2 = "香港體育館 (紅館)"
      toJson['location2'] = location2
  #Find date
    if re.match(date_pattern_str, str[i]):
      date = str[i]
      if (date[4] == '1'):
        date = date[0:4] + "/" + date[5:10]
      if (date[7] == '1'):
        date = date[:7] + "/" + date[8:10]
      """
            if(date[:2] == '28'):
                date = date[0] + '0' + date[2:]
            """
      date = date[0:10]
      toJson['date'] = date[0:10]
    """   
        #elif re.match(date_pattern2_str, str[i]):
            #print(str[i])
            #dateStr = str[i].split(" ")
           # date = str[i][0:4]+"/"+str[i][5:7]+"/"+str[i][8:10]
            #print(date)
        if re.match(r'\d{1}:\d{2}',str[i]):
            if("PM" in str[i] and len(str[i])>2 or len(str[i]) == 4):
                timeStr = str[i]
                hourInt = int(timeStr[0:1])+12
                if(hourInt > 23):
                        hourInt = 20
                Time = "{hour}:{min}".format(hour=hourInt, min=timeStr[2:4])
                toJson['time'] = Time
        """
    #Check time
    if (re.search(Time_pattern_str, str[i])):
      time = re.search(Time_pattern_str, str[i]).group(0)
      #print("check Time")
      #print(time)
      if (("P" in time) and len(time) > 2):
        time = time[:time.find("P")]
      hourInt = time.replace("B", "8")
      hourInt = re.sub("[^0-9]", "", hourInt[:2])
      hourInt = int(hourInt) + 12
      if (hourInt > 23 or hourInt == ''):
        hourInt = 20
      minInt = re.sub("[^0-9]", "", time[2:])
      Time = "{hour}:{min}".format(hour=hourInt, min=minInt)
      toJson['time'] = Time
      #print(Time)

    if ((any(ext in str[i] for ext in dayList))):
      resWeekday = str[i]
      weekday = [ext for ext in dayList if ext in resWeekday][0]
      """
            if(resWeekday==-1):
                resWeekday = str[i].find('（')
            time = str[i].replace('（','(')
            weekday = time[resWeekday+1:resWeekday+4]
            """
      toJson['weekday'] = weekday

    if re.search(price_pattern, str[i]) or re.search(r'H[CK]D', str[i]):
      priceStr = str[i]
      if (re.search(r'H[CK]D', priceStr)):
        priceUnitStr = re.search(r'H[CK]D', priceStr).group(0)
        priceStr = priceStr[priceStr.find(priceUnitStr):]
        priceStr = priceStr.replace(priceUnitStr, '')

      if priceStr is None or len(priceStr) > 9:
        continue
      priceStr = priceStr.replace(' ', '')
      if (priceStr.find('1.') > -1):
        priceStr = priceStr[priceStr.find(
          '1.')] + priceStr[priceStr.find('1.') + 2:]
      if (priceStr.find('1,') > -1):
        priceStr = priceStr[priceStr.find(
          '1,')] + priceStr[priceStr.find('1,') + 2:]
      #priceStr = re.match(price_pattern,priceStr).groups()[0]

      priceStr = re.sub("[^0-9.]", "", priceStr)
      if priceStr == '':
        continue
      if (600 < float(priceStr) < 1100 or priceStr == '8' or priceStr == '88'):
        price = 880.0
      elif (900 < float(priceStr) < 1300 or priceStr == '1' or priceStr == '12'
            or priceStr == '128'):
        price = 1280.0
      elif (400 < float(priceStr) < 500 or priceStr == '4'
            or priceStr == '48'):
        price = 480.0
      elif (0 < float(priceStr) < 9):
        price = 0.0
      if (price is not None):
        toJson['price'] = price

    if any(ext in str[i] for ext in similarGateList):
      #print(str[i])
      gateStr = str[i]

      gate = [ext for ext in similarGateList if ext in gateStr][0]
      if gate == 'VELLOW':
        gate = 'YELLOW'
      toJson['gate'] = gate
      hvGate = True
      section = gateStr.replace(gate, '')
      #print(gateStr)
      section = section.replace(' ', '')
      section = section.replace('I', '1')
      section = section.replace('Z', '5')
      if ('WP' in section):
        sectionNo = section.replace('WP', '')
        sectionNo = int(sectionNo)
        toJson['section'] = section
        if (0 < sectionNo < 12):
          wcSectionValid = True
      elif (re.match(twodigitPattern, section)):
        #print(int(section))
        sectionNo = int(section)
        toJson['section'] = section

      for j in range(1, 7):
        if (re.search(seatPattern, str[i - j]) and 'HKD' not in str[i - j]):
          seatStr = re.sub("[^0-9]", "", str[i - j])[-2:]
          #seatStr = seatStr[-2:]
          #print("1st Seat Check")
          #print(seatStr)
          if not (int(seatStr) > 200):

            rowStr = re.sub("[^0-9]", "", str[i - j].replace(seatStr, ''))

            seat = seatStr

            toJson['seat'] = seat
            if (len(rowStr) > 0):
              if (0 < int(rowStr) < 40):
                row = rowStr
                toJson['row'] = row
                hvRow = True
        if (re.match(twodigitPattern, str[i - j])
            or re.match(r'\d{1}', str[i - j]) and 'HKD' not in str[i - j]):
          rowStr = re.sub("[^0-9]", "", str[i - j])
          if (0 < int(rowStr) < 40):
            row = rowStr
            toJson['row'] = row
            hvRow = True
        if (re.match(twodigitPattern, str[i - j])):
          secionStr = re.sub("[^0-9]", "", str[i - j])
          if (39 < int(secionStr) < 80):
            section = secionStr
            toJson['section'] = section
    if (re.search(seatPattern, str[i]) and hvGate and 'HKD' not in str[i]):
      seatStr = re.sub("[^0-9]", "", str[i])[-2:]
      #seatStr = seatStr[-2:]
      if not (int(seatStr) > 200):
        rowStr = re.sub("[^0-9]", "", str[i].replace(seatStr, ''))
        seat = seatStr
        toJson['seat'] = seat
        if (len(rowStr) > 0):
          if (0 < int(rowStr) < 40):
            row = rowStr
            toJson['row'] = row
            hvRow = True
        #hvRow = False
      for j in range(1, 6):
        if ((re.match(twodigitPattern, str[i - j])
             or re.match(r'\d{1}', str[i - j])) and 'HKD' not in str[i - j]
            and not hvRow):
          rowStr = re.sub("[^0-9]", "", str[i - j])
          if (0 < int(rowStr) < 40):
            row = rowStr
            toJson['row'] = row
            hvRow = True
        if (re.match(twodigitPattern, str[i - j]) and hvRow):
          sectionStr = re.sub("[^0-9]", "", str[i - j])
          if (39 < int(sectionStr) < 80):
            section = sectionStr
            toJson['section'] = section
    if (re.match(twodigitPattern, str[i]) and 'HKD' not in str[i] and hvRow):
      sectionStr = str[i]
      if (39 < int(sectionStr) < 80):
        section = sectionStr
        toJson['section'] = section
    if ("WP" in str[i]):
      sectionStr = str[i]
      section = sectionStr[sectionStr.find('WP'):]
      toJson['section'] = section
      sectionNo = section.replace('WP', '')
      sectionNo = int(sectionNo)
      if (0 < sectionNo < 12):
        #print(section)
        wcSectionValid = True
    if (wcSectionValid and not checkedWC) or any(ext in str[i]
                                                 for ext in dWCList):
      if any(ext in str[i] for ext in dWCList):
        row = [ext for ext in dWCList if ext in str[i]][0]
        seat = row
        toJson['row'], toJson['seat'] = row, seat
      elif (str[i + 1].replace(' ', '').isdigit()):
        row = '-'
        seat = int(str[i + 1].replace(' ', ''))
        toJson['row'], toJson['seat'] = row, seat
      elif (i + 2 < len(str) and (re.match(twodigitPattern, str[i - j])
                                  or re.match(r'\d{1}', str[i + 2]))):
        row = '-'
        seat = int(str[i + 2].replace(' ', ''))
        toJson['row'], toJson['seat'] = row, seat
      checkedWC = True
  #return date, weekday, time, gate, section, row, seat
  return toJson


def verifaiCheck(imgStr):

  concertLogoPathStr = str(
    pathlib.Path().resolve()) + '/classes/logo/logo2.jpg'
  sampleTicketPathStr = str(
    pathlib.Path().resolve()) + '/classes/logo/sample.jpg'
  piiSamplePathStr = str(pathlib.Path().resolve()) + '/classes/logo/pii.jpg'
  qrSamplePathStr = str(
    pathlib.Path().resolve()) + '/classes/logo/sampleqr.jpg'
  sampleTicketimg = cv.imread(sampleTicketPathStr)
  forCheckimg = cv.imread(imgStr, 0)
  forCheckimg = image_resize(forCheckimg, resizeHeight)
  #init Json
  isValid = False
  isDefect = False
  toJson = initJson()
  img, originalSizePer = readImage(imgStr)

  TicketFrom, TicketFormat = checkMainColor(img)
  if (TicketFormat == 2):
    sampleTicketPathStr = str(
      pathlib.Path().resolve()) + '/classes/logo/sample2.jpg'
  imgMsgStr, blurVal = is_blurry(img)
  if (imgMsgStr != ''):
    toJson['is_out_of_focus'] = True
    toJson['quality_issue'] = imgMsgStr
    return toJson
  if (blurVal < 10000):
    toJson['quality_issue'] = 'a bit blurry/unclear'
  #Check the ticket is from Urbtix or Cityline
  #For this checking, as detectLogo func will significantly increase the process time
  #May choose another func to process
  #Check if is ticket
  ticketCoords = detectSim(imgStr, sampleTicketPathStr, 30)
  if ticketCoords is None:
    isTicket = False
  else:
    isTicket = True
    ticketCoords[ticketCoords < 0] = 0
    resultimage = four_point_transform(img, ticketCoords.reshape(4, 2))
    forQRCheckimg = four_point_transform(forCheckimg,
                                         ticketCoords.reshape(4, 2))
    Tix = TicketPos(resultimage)
    img_crop = resultimage[Tix[1]:Tix[3], Tix[0]:Tix[2]]
    forQRCheckimg = forQRCheckimg[Tix[1]:Tix[3], Tix[0]:Tix[2]]
    #return coordinates
    toJson['ticket_x1'], toJson['ticket_y1'] = int(
      ticketCoords[0][0][0] * originalSizePer[0]), int(ticketCoords[0][0][1] *
                                                       originalSizePer[1])

    toJson['ticket_x2'], toJson['ticket_y2'] = int(
      ticketCoords[1][0][0] * originalSizePer[0]), int(ticketCoords[1][0][1] *
                                                       originalSizePer[1])
    toJson['ticket_x3'], toJson['ticket_y3'] = int(
      ticketCoords[2][0][0] * originalSizePer[0]), int(ticketCoords[2][0][1] *
                                                       originalSizePer[1])
    toJson['ticket_x4'], toJson['ticket_y4'] = int(
      ticketCoords[3][0][0] * originalSizePer[0]), int(ticketCoords[3][0][1] *
                                                       originalSizePer[1])
    #old method
    #toJson['ticket_x1'], toJson['ticket_x2'], toJson['ticket_y1'], toJson['ticket_y2'] = Tix[0], Tix[2], Tix[1], Tix[3]
  toJson['is_ticket'] = isTicket
  #Tix = TicketPos(img)
  if (isTicket):

    #Check if same with the concert logo
    #if not will still process, but will label as not valid
    concertLogoCoords = detectLogo(forCheckimg, concertLogoPathStr, 30)
    if (concertLogoCoords is None):
      have_mirror_logo = False
      #print("Not for mirror concert?")
    else:
      have_mirror_logo = True
      isValid = True
    toJson['is_valid'], toJson["have_mirror_logo"] = isValid, have_mirror_logo
    TicketType = checkTicketType(img_crop)
    if (TicketType == 'Full'):
      toJson["have_stub"] = True
    else:
      toJson["have_stub"] = False

    #check the coordinates of PII
    piiCoords = detectLogo(forCheckimg, piiSamplePathStr, 10)
    if (piiCoords is not None):
      toJson['pii_x1'], toJson['pii_y1'] = int(
        piiCoords[0][0][0] * originalSizePer[0]), int(piiCoords[0][0][1] *
                                                      originalSizePer[1])

      toJson['pii_x2'], toJson['pii_y2'] = int(
        piiCoords[1][0][0] * originalSizePer[0]), int(piiCoords[1][0][1] *
                                                      originalSizePer[1])
      toJson['pii_x3'], toJson['pii_y3'] = int(
        piiCoords[2][0][0] * originalSizePer[0]), int(piiCoords[2][0][1] *
                                                      originalSizePer[1])
      toJson['pii_x4'], toJson['pii_y4'] = int(
        piiCoords[3][0][0] * originalSizePer[0]), int(piiCoords[3][0][1] *
                                                      originalSizePer[1])
    else:
      x0Per, x1Per, y0Per, y1Per = 0.65, 0.75, 0.57, 0.94
      if (TicketType == 'Full'):
        if (TicketFormat == 3):
          x0Per, x1Per = 0.62, 0.75
      elif (TicketType == 'Cut'):
        x0Per, x1Per, y0Per, y1Per = 0.6, 0.73, 0.5, 0.9

      toJson['pii_x1'], toJson['pii_y1'] = int(
        img_crop.shape[1] * x0Per * originalSizePer[0]), int(
          img_crop.shape[0] * y0Per * originalSizePer[1])

      toJson['pii_x2'], toJson['pii_y2'] = int(
        img_crop.shape[1] * x1Per * originalSizePer[0]), int(
          img_crop.shape[0] * y1Per * originalSizePer[1])

    #try get QR code

    qrCodeCoords = None  #detectLogo(forCheckimg,qrSamplePathStr, 5)
    if (qrCodeCoords is None):
      if (TicketType == 'Full'):
        forQRCheckimg = forQRCheckimg[int(forQRCheckimg.shape[0] *
                                          0.142):int(forQRCheckimg.shape[0] *
                                                     0.43),
                                      int(forQRCheckimg.shape[1] *
                                          0.73):int(forQRCheckimg.shape[1] *
                                                    0.98)]
      else:
        forQRCheckimg = forQRCheckimg[int(forQRCheckimg.shape[0] *
                                          0.142):int(forQRCheckimg.shape[0] *
                                                     0.43),
                                      int(forQRCheckimg.shape[1] *
                                          0.71):int(forQRCheckimg.shape[1] *
                                                    0.92)]

      qrCodeCoords = None  #detectLogo(forQRCheckimg, qrSamplePathStr)
      TixQR = QRCode(forQRCheckimg)
      if qrCodeCoords is not None or TixQR is not None:
        qrCodeCoords = detectLogoFromSame(forCheckimg, forQRCheckimg, 4)
        if qrCodeCoords is not None:
          toJson['qrcode_x1'], toJson['qrcode_y1'] = int(
            qrCodeCoords[0][0][0] * originalSizePer[0]), int(
              qrCodeCoords[0][0][1] * originalSizePer[1])

          toJson['qrcode_x2'], toJson['qrcode_y2'] = int(
            qrCodeCoords[1][0][0] * originalSizePer[0]), int(
              qrCodeCoords[1][0][1] * originalSizePer[1])
          toJson['qrcode_x3'], toJson['qrcode_y3'] = int(
            qrCodeCoords[2][0][0] * originalSizePer[0]), int(
              qrCodeCoords[2][0][1] * originalSizePer[1])
          toJson['qrcode_x4'], toJson['qrcode_y4'] = int(
            qrCodeCoords[3][0][0] * originalSizePer[0]), int(
              qrCodeCoords[3][0][1] * originalSizePer[1])
      else:
        #Simple version
        TixQR = QRCode(forCheckimg)
        if (TixQR is not None):
          toJson['qrcode_x1'], toJson['qrcode_x2'], toJson[
            'qrcode_y1'], toJson['qrcode_y2'] = int(
              TixQR[0] * originalSizePer[0]), int(
                TixQR[2] * originalSizePer[1]), int(
                  TixQR[1] * originalSizePer[0]), int(TixQR[3] *
                                                      originalSizePer[1])
    else:
      toJson['qrcode_x1'], toJson['qrcode_y1'] = int(
        qrCodeCoords[0][0][0] * originalSizePer[0]), int(
          qrCodeCoords[0][0][1] * originalSizePer[1])

      toJson['qrcode_x2'], toJson['qrcode_y2'] = int(
        qrCodeCoords[1][0][0] * originalSizePer[0]), int(
          qrCodeCoords[1][0][1] * originalSizePer[1])
      toJson['qrcode_x3'], toJson['qrcode_y3'] = int(
        qrCodeCoords[2][0][0] * originalSizePer[0]), int(
          qrCodeCoords[2][0][1] * originalSizePer[1])
      toJson['qrcode_x4'], toJson['qrcode_y4'] = int(
        qrCodeCoords[3][0][0] * originalSizePer[0]), int(
          qrCodeCoords[3][0][1] * originalSizePer[1])

    mkv_logo_sample_Str = None
    if (TicketFormat == 1 or TicketFormat == 2):
      mkv_logo_sample_Str = str(
        pathlib.Path().resolve()) + '/classes/logo/urbtix5.jpg'
    elif (TicketFormat == 3):
      mkv_logo_sample_Str = str(
        pathlib.Path().resolve()) + '/classes/logo/cityline5.jpg'
    if (mkv_logo_sample_Str is not None):
      mkv_logo_coords = detectLogo(
        forCheckimg,
        mkv_logo_sample_Str,
      )
      if (concertLogoCoords is None):
        toJson['have_mkv_logo'] = False
      else:
        toJson['have_mkv_logo'] = True

    retStr = ocrProccess(img_crop, TicketType, TicketFormat)

    #retStr = re.split('\n+', retStr)
    toJson = getInfo(retStr, TicketFormat, toJson)

    oriJson = {
      'location1': '',
      'location2': '',
      'date': '',
      'weekday': '',
      'time': '',
      'price': '',
      'gate': '',
      'section': '',
      'row': '',
      'seat': ''
    }
    result = True

    # extracting value to compare
    test_val = list(oriJson.values())[0]

    for ele in oriJson:
      if toJson[ele] == test_val:
        result = False
        toJson['quality_issue'] = 'Not all data captured, try a better angle'
        break

  else:
    #img = cv.imread(imgStr)
    img = image_resize(img, 600)
    imgMsgStr = checkHist(img)
    if (imgMsgStr != ''):
      toJson['is_too_dark'] = True
      toJson['quality_issue'] = imgMsgStr
      return toJson
    if (checkA4(img)):
      toJson['is_a4_document'] = True
    elif (len(detectFace(img)) != 0):
      toJson['face_detected'] = True
      toJson['quality_issue'] = 'Found a face, Selfie?'
    else:
      toJson['quality_issue'] = 'Please try a better angle to capture'

    return toJson

  return toJson  #, retStr
  #toJson['location1'], toJson['location2'] = getLocation(retStr)
  #For extracting date time, gate and seats information

  #toJson['date'], toJson['weekday'], toJson['time'], toJson['gate'], toJson[
  # 'section'], toJson['row'], toJson['seat'] = getInfo(retStr, TicketFormat
