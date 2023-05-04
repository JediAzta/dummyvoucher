# Voucher Recognization API

This API aims to recognize the images with the filename provided and return the data.

## Installation Guide
This program can be executed under any OS environment.  The following guide is based on the RedHat environment with **root** rights, and please remind to update the OS before starting the process.

> yum update -y

### Programming Environment
This program requires to install with Python 3.7+

(A) Installation of Python

> yum install python3 python3-pip

(B) Install the frameworks and packages for the program
The API requires Flask, OpenCV and the OCR model to support.  In addition, it is required to have the API deployment service to support the program.  Please remind to follow the steps below one by one to install suitable frameworks:

(Depends on your Linux, pip may map to other version, you need to check by pip -V)

> pip install matplotlib imutils paddleocr paddlepaddle flask gunicorn -y
> pip uninstall opencv-python opencv-contrib-python -y
> pip install opencv-python-headless opencv-contrib-python-headless -y

**Warning**
If there is any step above not correctly execute (Normally OpenCV crashed).  You need to uninstall everything by entering the following command then restart the installation above:

> pip uninstall paddleocr paddlepaddle opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless

Once you have completed the installation, remind to exit **root** right before the next steps

## Environment config
The config file is located under classes/config.json the content as below
```JSON
{
  "path": "/home/ec2-user/API/samples", 
  "wwwroot": "/home/ec2-user/API"
}
```
| key | Description |
|---|---|
| path | The absolute path of the voucher image that should be upload (for API) |
| wwwroot | The absolute path of the root location of the webserver (for console) |


## Testing if the API is working
Under the API project, execute the command
> python3 main.py

Now, the API is listening at port 8080.  You may press Ctrl-c to stop the service.

Noted that this is **NOT** a production deployment.  The cloud environment will automatically shut it down once you leave the console. 

## Deploy API
Gunicorn is the Flask deployment application that allows deploying any web services implemented by Flask. To deploy the API, you can execute the following command:
> gunicorn -w 1 -b 0.0.0.0:8080 main:app

-w (number) is the number of processes available for request and response.  If the RAM is enough for the job, it can be higher.

## Shutdown the API
If there is any reason that you wish to shut down the API service. You can execute the command at the terminal:
> pkill gunicorn

## Console Scanner Program
In the project, there is a Python script called scan.py.  In this stage, this file is NOT executable. To make it executable, you must grant the right to the file.

> chmod +x scan.py

You can check the right of this file with x, and that means you can execute the program now by:

> ./scan.py [filename]

[filename] is the voucher filename which is located in the upload folder from the webserver

# Questions?

If you have any questions please let us know.