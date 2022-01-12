from flask import Flask, render_template, Response
import cv2
import cvlib as cv
import detect_CNN_7_classes as detectCNN_7
import detect_CNN_4_classes as detectCNN_4
import detect_RF_4_classes as detectRF
import detect_SVM_4_classes as detectSVM

app = Flask(__name__)

camera = cv2.VideoCapture('./static/img/img.jpg')

def gen_frames_detect_CNN_4(): 
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', detectCNN_4.showWebcam(camera))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_detect_CNN_7(): 
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', detectCNN_7.showWebcam(camera))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_detect_RF(): 
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', detectRF.showWebcam(camera))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_detect_SVM(): 
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', detectSVM.showWebcam(camera))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_normal():  
    while True:
        success, frame = camera.read() 
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_normal_stream')
def video_normal_stream():
    return Response(gen_frames_normal(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_detect_stream_CNN_4')
def video_detect_stream_CNN_4():
    return Response(gen_frames_detect_CNN_4(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_detect_stream_CNN_7')
def video_detect_stream_CNN_7():
    return Response(gen_frames_detect_CNN_7(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_detect_stream_SVM')
def video_detect_stream_SVM():
    return Response(gen_frames_detect_SVM(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_detect_stream_RF')
def video_detect_stream_RF():
    return Response(gen_frames_detect_RF(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)