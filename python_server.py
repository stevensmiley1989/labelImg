from multiprocessing import Queue
xy=Queue()
ready=Queue()
response=Queue()
import time
ready_status=True
global ThreadCount
ThreadCount=0
global data
PORT=8765
from _thread import *
def threaded_client(conn,xy,response,ready):
    global ready_status
    myresponse=''
    with conn:
        while True:
            myresponse=response.get()
            if myresponse!='':
                conn.sendall(myresponse.encode())
            try:
                ready_status=ready.get()
                print('waiting for response')
                #myresponse=response.get()
                #print('sending response')
                #if ready_status==True:
                #   conn.sendall(myresponse.encode())
                    
            except:
                pass
            data=conn.recv(1024)
            if not data:
                break;
            print(data.decode());
            xy.put(data.decode().strip(' '));


        conn.sendall(data);print('Received',repr(data))
def init(xy,ready,response,PORT=PORT):
    import socket
    global ThreadCount
    global ready_status
    HOST=socket.gethostname()
    print(HOST)
    
    #PORT=8888
    #while True:
    #    try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        #s.settimeout(0.5) #settimeout
        s.bind((HOST,PORT))
        s.listen(500)
        while True:
            conn,addr=s.accept()
            start_new_thread(threaded_client,(conn,xy,response,ready,))
            ThreadCount+=1
            print('Thread Number: '+str(ThreadCount))
        #except:
        #    pass

def get_timed_out(q, timeout):
    stoptime = time.monotonic() + timeout - 1
    while time.monotonic() < stoptime:
        try:
            return q.get(timeout=1)  
        except queue.Empty:
            pass
    timed_out=q.get(timeout=max(0, stoptime + 1 - time.monotonic()))  
    return 'timed_out'

def convert_boxes(full_boxes_i):
    print(type(full_boxes_i))
    bbox_list=[]
    if full_boxes_i.find(';')==-1:
        print('NO ; in these full_boxes_i')
        return bbox_list
    if full_boxes_i.find('&')==-1:
        full_boxes_i=full_boxes_i+'&'

    for bbxy_i in full_boxes_i.split('&'):
        if len(bbxy_i)>0: 
            print('valid bbxy_i')         
            boundingboxes_i={}
            obj_found=bbxy_i.split('_')[1].split(';')[0]
            obj_found=obj_found.replace("b'",'').replace("'","").replace('"',"")
            bbxy_i=bbxy_i.split(obj_found)[1]

            if bbxy_i.count(";")>3:
                xmin=(int(bbxy_i.split(';')[1]));
                ymin=(int(bbxy_i.split(';')[2]));
                xmax=(int(bbxy_i.split(';')[3]));
                ymax=(int(bbxy_i.split(';')[4]));
                print("x1= {}; y1= {}; x2= {}; y2={}".format(xmin,ymin,xmax,ymax))
            if bbxy_i.count(";")>4:
                confidence=(float(bbxy_i.split(';')[5]));
                print("confidence= ",confidence)
            if bbxy_i.count(";")>5:
                W=(int(bbxy_i.split(';')[6]));
                print("W= ",W)
            if bbxy_i.count(";")>6:
                H=(int(bbxy_i.split(';')[7]));
                print("H=",H)
            boundingboxes_i['obj_found']=obj_found
            boundingboxes_i['xmin']=xmin
            boundingboxes_i['ymin']=ymin
            boundingboxes_i['xmax']=xmax
            boundingboxes_i['ymax']=ymax
            boundingboxes_i['confidence']=confidence
            boundingboxes_i['W']=W
            boundingboxes_i['H']=H
            bbox_list.append(boundingboxes_i)
    return bbox_list          
if __name__=='__main__':
    init(xy,ready,response,PORT)
