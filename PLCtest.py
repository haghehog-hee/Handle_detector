import socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('192.168.143.4', 1285))
#client.send("00FF0A006400000024D0C00".encode())
#message = "00FF000A4D20000000640C00".encode()
#message = 0x00FF000A4D20000000640800.to_bytes(12, 'big')
message = 0x00FF0A006600000020590200.to_bytes(12, 'big')
print(message.hex())
client.send(message)
from_server = client.recv(5050)
print(from_server)
print(message)

print(from_server.hex())

client.close()
