import socket
import time
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.settimeout(1)
try:
    client.connect(('192.168.143.4', 1285))
except socket.timeout as e:
    print("Nope !!!")
#message = 0x00FF0A0027020000204D0100.to_bytes(12, 'big')

# msg = "0"
# msg = hex(msg)
# print(msg)

# message = 0x03FF0A009B0000002044020034127698.to_bytes(16, 'big')
# client.send(message)
# from_server = client.recv(5050)
#
# print(from_server.hex())


while True:
    message = 0x00FF0A000900000020580100.to_bytes(12, 'big')

    #print(message.hex())
    client.send(message)
    from_server = client.recv(5050)
    #print(from_server)
    #print(message)

    print(from_server.hex())
    time.sleep(2)

# message = 0x00FF0A000000000020580000.to_bytes(12, 'big')
#
# #print(message.hex())
# client.send(message)
# from_server = client.recv(5050)
# #print(from_server)
# #print(message)
#
# print(from_server.hex())
# message = 0x02FF0A00A2020000204D0C00000000000000.to_bytes(18, 'big')
# client.send(message)
# from_server = client.recv(5050)
#
# print(from_server.hex())
# message = 0x00FF0A00A2020000204D0C00.to_bytes(12, 'big')
#
# #print(message.hex())
# client.send(message)
# from_server = client.recv(5050)
# print(from_server.hex())
client.close()
