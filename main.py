from flask import Flask, request, jsonify, send_file
from classes.vouchers import Vouchers
import pathlib, json, os

app = Flask('app')  
vounchers = Vouchers()

# Dummy Endpoints for Testing purpose
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
  return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
  return jsonify({'msg': 'TESTING ONLY'}), 200

@app.route('/daolpuElif', methods=['POST'])
def upload_file():
  headerKey = request.headers.get('key')
  if headerKey == 'Vm91Y2hlclZhbGlkYXRpb25BUElpbkRldmVsb3BtZW50RW52aXJvbm1lbnQ=':
    if 'file' not in request.files:
      return jsonify({'msg': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
      return jsonify({'msg' : 'No file selected for uploading'}), 400
    if file and allowed_file(file.filename):
      file.save(os.path.join(vounchers.retrieveImgPath(), file.filename))
      data = vounchers.regonize(file.filename)
      if(data):
        return jsonify({'msg' : 'File successfully uploaded', 'reg': data }),201
      else:
        return jsonify({'msg' : 'File successfully uploaded', 'err': 'no data return from CV'}), 500
    else:
      return jsonify({'msg' : 'Allowed file types are png, jpg, jpeg, gif'}), 400 
  else:
    return jsonify({'msg': 'None of your business'}), 401
# End of Dummy


# API Endpoints
@app.route('/docs')
def apiDoc():
  path = str(pathlib.Path().resolve()) + '/documentation.json'
  try:
    with open(path, 'r') as docFile:
      data = json.load(docFile)
    return jsonify(data), 200
  except:
    return jsonify({'msg': 'Something wrong'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
  headerKey = request.headers.get('key')
  if headerKey=='Vm91Y2hlclZhbGlkYXRpb25BUElpbkRldmVsb3BtZW50RW52aXJvbm1lbnQ=':
    return send_file(vounchers.retrieveImgPath()+'/'+filename, as_attachment=True)
  return jsonify({'msg': 'Nope'}), 404

@app.route('/config', methods=['GET', 'PUT'])
def apiConfig():
  headerKey = request.headers.get('key')
  if headerKey=='Vm91Y2hlclZhbGlkYXRpb25BUElpbkRldmVsb3BtZW50RW52aXJvbm1lbnQ=':
    if request.method == 'GET':
      return jsonify({'img': vounchers.retrieveImgPath()}), 200
  
    if request.method == 'PUT':
      body = request.get_json()
      if body is not None:
        if body['path'] is not None:
          vounchers.changeImgPath(body['path'])
          return jsonify({'msg': 'Path update successfully'}), 201
        return jsonify({'msg': 'Illedgal Data to be updated'}), 401
    
  return jsonify({'msg': 'Invalid request'}), 400
  
@app.route('/voucher/<name>', methods=['GET'])
def getVoucher(name):
  # request
  headerKey = request.headers.get('key')
  if headerKey=='Vm91Y2hlclZhbGlkYXRpb25BUElpbkRldmVsb3BtZW50RW52aXJvbm1lbnQ=':
    # response to client
    data = vounchers.regonize(name)
    # successful (possible valid or invalid)
    if(data):
      return jsonify(data),200
    # failed
    else:
      return jsonify({'msg': 'Voucher is not found in the record'}), 404
  else: 
    return jsonify({'msg': 'Illedgal request'}), 400

@app.errorhandler(400)
def badRequest(e):
  return jsonify({'msg': 'Illedgal request.. NOnonono'}), 400

@app.errorhandler(404)
def notFound(e):
  return jsonify({'msg': 'Endpoint is not found'}), 404

@app.errorhandler(405)
def methodNotAllowed(e):
  return jsonify({'msg': 'Invalid method request'}), 405

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8080)

