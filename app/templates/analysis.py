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
port = 51078 #5571
bufsiz = 10000
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

@app.route('/semantic')
def semantic():
    return render_template('semantic_index.html')

def classify_base(filename, model_choose):
    #construct request
    request1 = image_interface_pb2.ClassifyRequest()
    request1.file_name = "/data/home/vincentyao/image_classification/plain_board/image_analysis/uploads/" + filename
    request1.top_n_result = 5
    if model_choose == 0:
        request1.request_type = image_interface_pb2.ClassifyRequest.CLASSIFY
    elif model_choose == 1:
        request1.request_type = image_interface_pb2.ClassifyRequest.CLASSIFY_PAIPAI
    data = request1.SerializeToString()

    #tcp connect
    tcpCliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpCliSock.connect(ADDR)
    tcpCliSock.send(data)
    data = tcpCliSock.recv(bufsiz)
    proto_dat = image_interface_pb2.ClassifyResponse()
    succ = proto_dat.ParseFromString(data)
    tcpCliSock.close()

    #package result page
    returnbuf = "<!DOCTYPE html><html lang=\"en\"><head><link href=\"//netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css\" rel=\"stylesheet\"></head>"
    returnbuf += "<p><img src=\"/uploads/" + filename + "\"/></p>"
    returnbuf += "<p>[" + filename + "]"  + " Classify result: (" + str(len(proto_dat.rsp_res)) + ")</p>"
    returnbuf += "<br><table class=\"table table-bordered\"><thead><tr><th>category_name</th> <th>category_weight</th></tr></thead><tbody>"
    for result in proto_dat.rsp_res:
       returnbuf+="<tr><td>"
       returnbuf+= (result.category_name + "</td><td>" + str(result.category_weight) + "</td></tr>")
    returnbuf += "</tbody></table></html>"
    
    return returnbuf


def similarity_base(filename, filename2, feature_layer):
    #construct request
    request1 = image_interface_pb2.ClassifyRequest()
    request1.file_name =  "/data/home/vincentyao/image_classification/plain_board/image_analysis/uploads/" + filename
    request1.compare_file_name = "/data/home/vincentyao/image_classification/plain_board/image_analysis/uploads/" + filename2
    request1.feature_layer = feature_layer
    request1.request_type = image_interface_pb2.ClassifyRequest.SIMILARITY
    data = request1.SerializeToString()

    # connect server 
    tcpCliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpCliSock.connect(ADDR)    
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

def search_base(filename, top_n_res, same_pic):
    request1 = image_interface_pb2.ClassifyRequest()
    request1.file_name =  "/data/home/vincentyao/image_classification/plain_board/image_analysis/uploads/" + filename
    request1.top_n_result = top_n_res
    request1.request_type = image_interface_pb2.ClassifyRequest.SEARCH
    if same_pic == 0:
        request1.max_sim_thres = 0.99
    request1.min_sim_thres = 0.01
    data = request1.SerializeToString()

    tcpCliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpCliSock.connect(ADDR)
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
       returnbuf+= ("<img src=\"/paipai_media/" + result.search_file_name.split("/")[-1] + "\"/>" + "</td><td>" + str(result.search_similarity) + "</td></tr>")
    returnbuf += "</tbody></table></html>"

    return returnbuf

def semantic_base(filename, top_n_res):
    request1 = image_interface_pb2.ClassifyRequest()
    request1.file_name =  "/data/home/vincentyao/image_classification/plain_board/image_analysis/uploads/" + filename
    request1.top_n_result = top_n_res
    request1.request_type = image_interface_pb2.ClassifyRequest.SEMANTIC
    data = request1.SerializeToString()

    tcpCliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpCliSock.connect(ADDR)
    tcpCliSock.send(data)
    data = tcpCliSock.recv(bufsiz)
    proto_dat = image_interface_pb2.ClassifyResponse()
    succ = proto_dat.ParseFromString(data)
    tcpCliSock.close()

    #package result page
    returnbuf = "<!DOCTYPE html><html lang=\"en\"><head><link href=\"//netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css\" rel=\"stylesheet\"></head>"
    returnbuf += "<p><img src=\"/uploads/" + filename + "\"/></p>"
    returnbuf += "<p>Search Result : [" + str(request1.top_n_result) + "]</p>"
    returnbuf += "<br><table class=\"table table-bordered\"><thead><tr><th>token</th> <th>weight</th><th>token</th> <th>weight</th></tr></thead><tbody>"
    index=0
    for result in proto_dat.search_feature:
        if index%2 == 0:
            returnbuf+="<tr>"    
        returnbuf+= "<td>" + str(result.feature_token) + "</td> \
            <td>" + str(result.feature_weight) + "</td>"
        if index%2 == 1:
            returnbuf+="</tr>"
        index=index+1
    returnbuf += "</tbody></table></html>"

    return returnbuf

# Route that will process the file upload
@app.route('/upload/classify', methods=['POST'])
def upload_classify():
    # Get the name of the uploaded file
    file = request.files['file']
    if len(file.filename) < 3:
        return "No Picture!!!"
    model_choose = request.form['model_choose']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which will basicaly show on the browser the uploaded file
        #return redirect(url_for('uploaded_file', filename=filename))
        return classify_base(filename, int(model_choose))

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
        # Move the file form the temporal folder to the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        
        return similarity_base(filename, filename2, feature_layer)

@app.route('/upload/search', methods=['POST'])
def upload_search():
    # Get the name of the uploaded file
    file = request.files['file']
    if len(file.filename) < 3:
        return "No Picture!!!"
    top_n_res = request.form['top_n_res']
    same_pic = request.form['detele_same_pic']
    #print top_n_res,same_pic
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which will basicaly show on the browser the uploaded file
        #return redirect(url_for('uploaded_file', filename=filename))
        return search_base(filename, int(top_n_res), int(same_pic)) 

@app.route('/upload/semantic', methods=['POST'])
def upload_semantic():
    # Get the name of the uploaded file
    file = request.files['file']
    if len(file.filename) < 3:
        return "No Picture!!!"
    top_n_res = request.form['top_n_res']

    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which will basicaly show on the browser the uploaded file
        #return redirect(url_for('uploaded_file', filename=filename))
        return semantic_base(filename, int(top_n_res)) 

@app.route('/testcase/<int:test_id>', methods=['POST'])
def testcase(test_id):
    #if testid < 10: classify task
    if test_id == 1:
        return classify_base("011CC50766A5748A.jpg", 0);
    elif test_id == 2:
        return classify_base("010AE6018D9EA335.jpg", 1);
    elif test_id == 11:
        return similarity_base("010A3F6069657A9B.jpg", "010A5A6069657A9B.jpg", 23);
    elif test_id == 12:
        return similarity_base("010A4A018D9EA335.jpg", "010A5A018D9EA335.jpg", 23); 
    elif test_id == 21:
        return search_base("010A5A6069657A9B.jpg", 8, 0);
    elif test_id == 22:
        return search_base("010A06054CD8462F.jpg", 8, 1); 
    elif test_id == 23:
        return search_base("010397038D9D9BD0.jpg", 8, 1);
    elif test_id == 31:
        return semantic_base("item-00A686B1-907BB62400000000040100001AE6247F.0.jpg", 20);
    elif test_id == 32:
        return semantic_base("item-0FBF2D89-068DE92300000000040100000F4372F2.0.jpg", 20); 
    elif test_id == 33:
        return semantic_base("item-002BCB57-E020D61B000000000401000014164099.0.jpg", 20);                                       
    else:
        return "Hello World"

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
    #return send_from_directory("../../../../nickyhe/image_retrieve/allpic/", filename)

@app.route('/paipai_media/<filename>')
def paipai_media(filename):
    return send_from_directory("paipai_media/", filename)

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("5002"),
        debug=True
    )
