## Federated Leanring Physical System Implementation 
Here we have the code and resources required to run the server and client code on a laptop and raspberry pi. 
Below are the required libraries as well as how to run the code on each of the systems. 

# Dataset
In order to run the physical implemetation with the HARdataset, it needs to be downloaded from the following linK:
     https://github.com/xmouyang/FL-Datasets-for-HAR/tree/main/datasets/HARBox

The data_pre.py file within the dataset then needs to be replaced by the one in this repo. The resulting .pt file then needs to be placed in the directory thait contains this repo. 

We were not able to include the .pt file we used due to the large size. 
 

# Server 
The Server is comprised of all files and directories in the FL-Physical directory other than the staging.py and config.txt files. To run the server code, navigate to the FL-Physical directory and run the command python main_server_fed.py. 

Before running, navigate to the config_server.txt file and set the following peramiters:
NUM_CLIENTS=(Number of clients)
NUM_gl_EPOCHS=(Number of global epohs)
SERVER_PORT= (Specifies the port on the server to connect to)
SERVER_IP= (Speifies the IP of the server to connect to )
CLIENT_USRNM=(Client Username)
CLIENT_PSWD=(Client password)
CLIENT_FILEPATH=(Filepath to staging.py in the client)
MODELFOLDER=(complete filepath to Pi_models)

# Client 
Each client needs the staging.py file as well as a config.txt file. 
These need to be placed together in any directory on the raspberry pi. The config file also needs to be updated to the parameters listed below

CLIENT_ID= (This will be the ID for the client, should be 1, 2, 3...)
SERVER_PORT= (Specifies the port on the server to connect to)
SERVER_IP= (Speifies the IP of the server to connect to )
SERVER_NAME= (Specifies the username for the server)
SERVER_PASS= (Specifies the password for the server)
