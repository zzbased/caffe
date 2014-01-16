import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import socket
import time
import struct
import image_interface_pb2

# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

# Server address
host = '10.1.152.71'
port = 5571
bufsiz = 10240
ADDR = (host,port)



# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify')
def classify():
    return render_template('classify_index.html')

@app.route('/similarity')
def similarity():
    return render_template('similarity_index.html')

@app.route('/search')
def search():
    return render_template('search_index.html')

# Route that will process the file upload
@app.route('/upload/similarity', methods=['POST'])
def upload_similarity():
    # Get the name of the uploaded file
    file = request.files['file']
    if len(file.filename) < 3:
        return "No Picture!!!"
    file2 = request.files['file2']
    if len(file2.filename) < 3:
        return "No Compare Picture!!!"

    feature_layer = int(request.form['feature_layer'])
    print feature_layer
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        filename2 = secure_filename(file2.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        # connect server 
        tcpCliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcpCliSock.connect(ADDR)
        request1 = image_interface_pb2.ClassifyRequest()
        request1.file_name =  "/data/home/vincentyao/image_classification/plain_board/image_analysis/uploads/" + filename
        request1.compare_file_name = "/data/home/vincentyao/image_classification/plain_board/image_analysis/uploads/" + filename2
        #request1.top_n_result = 5
        request1.feature_layer = feature_layer
        request1.request_type = image_interface_pb2.ClassifyRequest.SIMILARITY

        data = request1.SerializeToString()
        tcpCliSock.send(data)
        data = tcpCliSock.recv(bufsiz)
        proto_dat = image_interface_pb2.ClassifyResponse()
        succ = proto_dat.ParseFromString(data)
        tcpCliSock.close()
        
        returnbuf = "<!DOCTYPE html><html lang=\"en\"><head><link href=\"//netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css\" rel=\"stylesheet\"></head>"
        returnbuf += "<p><img src=\"/uploads/" + filename + "\"/></p>"
        returnbuf += "<p><img src=\"/uploads/" + filename2 + "\"/></p>"
        returnbuf += "<p>[" + filename + "]"  +  " vs [" + filename2 + "]" +  " Similarity : " + str(proto_dat.similarity) + "</p>"
        
        return returnbuf


# Route that will process the file upload
@app.route('/upload/classify', methods=['POST'])
def upload_classify():
    # Get the name of the uploaded file
    file = request.files['file']
    if len(file.filename) < 3:
        return "No Picture!!!"
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        #return redirect(url_for('uploaded_file', filename=filename))
        tcpCliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcpCliSock.connect(ADDR)
        request1 = image_interface_pb2.ClassifyRequest()
        request1.file_name =  "/data/home/vincentyao/image_classification/plain_board/image_analysis/uploads/" + filename
        request1.top_n_result = 5
        request1.request_type = image_interface_pb2.ClassifyRequest.CLASSIFY
        data = request1.SerializeToString()
        tcpCliSock.send(data)
        data = tcpCliSock.recv(bufsiz)
        proto_dat = image_interface_pb2.ClassifyResponse()
        succ = proto_dat.ParseFromString(data)

        #package result page
        returnbuf = "<!DOCTYPE html><html lang=\"en\"><head><link href=\"//netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css\" rel=\"stylesheet\"></head>"
        returnbuf += "<p><img src=\"/uploads/" + filename + "\"/></p>"
        returnbuf += "<p>[" + filename + "]"  + " Classify result: (" + str(len(proto_dat.rsp_res)) + ")</p>"
        returnbuf += "<br><table class=\"table table-bordered\"><thead><tr><th>category_name</th> <th>category_weight</th></tr></thead><tbody>"
        #print len(proto_dat.rsp_res)
        for result in proto_dat.rsp_res:
           returnbuf+="<tr><td>"
           returnbuf+= (result.category_name + "</td><td>" + str(result.category_weight) + "</td></tr>")
        returnbuf += "</tbody></table></html>"
        tcpCliSock.close()
        return returnbuf 
        #return render_template('result.html', returnbuf=returnbuf)

@app.route('/upload/search', methods=['POST'])
def upload_search():
    # Get the name of the uploaded file
    file = request.files['file']
    if len(file.filename) < 3:
        return "No Picture!!!"
    top_n_res = request.form['top_n_res']
    same_pic = request.form['detele_same_pic']
    print top_n_res,same_pic
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        #return redirect(url_for('uploaded_file', filename=filename))
        tcpCliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcpCliSock.connect(ADDR)
        request1 = image_interface_pb2.ClassifyRequest()
        request1.file_name =  "/data/home/vincentyao/image_classification/plain_board/image_analysis/uploads/" + filename
        request1.top_n_result = int(top_n_res)
        request1.request_type = image_interface_pb2.ClassifyRequest.SEARCH
        if int(same_pic) == 0:
            request1.max_sim_thres = 0.99
        #return "Test"
        data = request1.SerializeToString()
        tcpCliSock.send(data)
        data = tcpCliSock.recv(bufsiz)
        proto_dat = image_interface_pb2.ClassifyResponse()
        succ = proto_dat.ParseFromString(data)
        tcpCliSock.close()
        

        #package result page
        returnbuf = "<!DOCTYPE html><html lang=\"en\"><head><link href=\"//netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css\" rel=\"stylesheet\"></head>"
        returnbuf += "<p><img src=\"/uploads/" + filename + "\"/></p>"
        returnbuf += "<p>Search Result : [" + str(request1.top_n_result) + "]</p>"
        returnbuf += "<br><table class=\"table table-bordered\"><thead><tr><th>Image</th> <th>similarity_weight</th></tr></thead><tbody>"
        
        for result in proto_dat.search_res:
           returnbuf+="<tr><td>"
           returnbuf+= ("<img src=\"/site_media/" + result.search_file_name.split("/")[-1] + "\"/>" + "</td><td>" + str(result.search_similarity) + "</td></tr>")
        returnbuf += "</tbody></table></html>"
        
        return returnbuf 


# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/site_media/<filename>')
def site_media(filename):
    return send_from_directory("site_media/", filename)

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("5002"),
        debug=True
    )