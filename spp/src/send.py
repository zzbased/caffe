#!/usr/bin/python
import socket
import time
import struct
import image_interface_pb2

host = '10.1.152.71'
port = 5571
bufsiz = 1024
ADDR = (host,port)

tcpCliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpCliSock.connect(ADDR)

while True:
    raw_input('> ')

    request = image_interface_pb2.ClassifyRequest()
    request.file_name="/data/home/vincentyao/image_classification/plain_board/imageclassify/app/uploads/basketball.png"
    request.top_n_result = 5
    data = request.SerializeToString()
    print "send data len : %d" % len(data)
    tcpCliSock.send(data)
    time.sleep(1)
    data = tcpCliSock.recv(bufsiz)
    print "receive data len : %d" % len(data)
    if not data:
        break
    #print data
    #a,b,c=struct.unpack("3i", data)
    #print a,b,c
    proto_dat = image_interface_pb2.ClassifyResponse()
    succ = proto_dat.ParseFromString(data)
    print "succ:",succ
    print len(proto_dat.rsp_res)
    for result in proto_dat.rsp_res:
         print result.category_name
         print result.category_weight

tcpCliSock.close()
