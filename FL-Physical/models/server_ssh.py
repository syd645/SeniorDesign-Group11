import paramiko
from scp import SCPClient
import socket
import time
import threading



def SendToClient(client,clientsocket, file = "",filepath = "",message = ""):
    #try:
    with SCPClient(client.get_transport()) as scp_Client:
        scp_Client.put(file, filepath)

    clientsocket.send(bytes(message, "utf-8"))
    msg = clientsocket.recv(64)
    msg_decoded = msg.decode("utf-8")
    print(msg_decoded)
    #except:
    #    print("SentToClient() Failed.")


def Connection_handling(clientsocket, address):
    #time.sleep(5)
    f = open("config_server.txt", "r")
    lineCount = 0
    for line in f:
        currentLine = line.strip('\n').split("=")

        if currentLine[0] == 'CLIENT_USRNM':
            CLIENT_USRNM = currentLine[1]       

        if currentLine[0] == 'CLIENT_PSWD':
            CLIENT_PSWD = currentLine[1]  
        
        lineCount += 1

    f.close()
    username = CLIENT_USRNM   # username of raspberry pi 4
    password = CLIENT_PSWD   # pasword of raspberry pi 4

    # set up paramiko ssh client for scp file sending
    SSH_client = paramiko.client.SSHClient()
    SSH_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    SSH_client.connect(address[0], username=username, password=password)


    SendToClient(client=SSH_client,clientsocket=clientsocket,file="models/main_server_fed_overall.pt", 
                 filepath="/home/pi/Desktop/main_server_fed.pt",
                 message="Server:Sent file to client")
    #time.sleep(3)

    
    SSH_client.close()

'''  
while !flag
    try
        Connectio:
        flag = true
    except:
        flag = false
'''



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