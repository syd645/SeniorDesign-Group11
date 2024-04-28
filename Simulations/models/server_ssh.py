import paramiko
from scp import SCPClient
import socket
import time
import threading


def SendToClient(client,clientsocket, file = "",filepath = "",message = ""):
    try:
        with SCPClient(client.get_transport()) as scp_Client:
            scp_Client.put(file, filepath)

        clientsocket.send(bytes(message, "utf-8"))
        msg = clientsocket.recv(64)
        msg_decoded = msg.decode("utf-8")
    # print(msg_decoded)
    except:
        print("SentToClient() Failed.")


def Connection_handling(clientsocket, address):
    #time.sleep(5)
    username = 'pi'   # username of raspberry pi 4
    password = 'raspberry'   # pasword of raspberry pi 4

    # set up paramiko ssh client for scp file sending
    SSH_client = paramiko.client.SSHClient()
    SSH_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    SSH_client.connect(address[0], username=username, password=password)


    SendToClient(client=SSH_client,clientsocket=clientsocket,file="C:\\Users\\winte\\Desktop\\LAST_SEMESTER\\CPE496\\github\\federated-learning\\models\\main_server_fed_1.pt", filepath="/home/pi/Desktop/test_fromSyd.pt",message="Server:Sent file to client")
    #time.sleep(3)

   # msg = clientsocket.recv(64)
  #  msg_decoded = msg.decode("utf-8")
   # print(msg_decoded)
    SSH_client.close()

def serverSetup(): 
    # basic connection definitions
    # Update the next lines with raspberry pi and host computer information
    #SSH_IP = "10.4.158.232" # IP of the system to be SSHed into
                            # SSHport = 22  # Default SSH port (doesnt need to be hardcoded, but this is the port SSH uses)

    host = "10.4.129.192"   # this the address of server computer (not client!!)
    port = 4045

    # set up TCP socket connection for server 
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host,port))
    server.listen(1)      

    # Create threads
    threads = []
    for i in range(1):          #change range to num clients
        # Wait for 
        clientsocket, address = server.accept() 
        print("connection from " + address[0] + " accepted.")
        thread = threading.Thread(target = Connection_handling, args=(clientsocket, address))
        threads.append(thread)

    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    # close server
    server.close()


'''
Server/client port settings for windows
Control Panel
\->System and Security
    \->Windows Defender Firewall
        \->Advanced Settings
            \->Inbound Rules
                \->New Rule...
                    |ruletype == Port
                    |TCP and Specific local ports: 4045 (or any port you want to use over 1000ish and not reserved for any other communication)
                    |Allow the Connection
                    |Domain Private Public
                    |Name = TCP Port 4045 opening
            \->Outbound Rules
                \->New Rule...
                    |ruletype == Port
                    |TCP and Specific local ports: 4045 (or any port you want to use over 1000ish and not reserved for any other communication)
                    |Allow the Connection
                    |Domain Private Public
                    |Name = TCP Port 4045 opening


'''