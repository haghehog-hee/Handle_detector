import os
import re

images_path = "C:\\Tensorflow\\Dataset\\april\\test\\img\\"
annotations_path = "C:\\Tensorflow\\Dataset\\april\\test\\ann\\"
pathing_flag = True
naming_flag = False
ann_paths = os.listdir(annotations_path)

if pathing_flag:
    for PATH in ann_paths:
        ann_path = annotations_path + PATH
        file = open(ann_path, "r")
        lines = file.readlines()
        filename = lines[2]
        filename = filename.replace("\t<filename>","")
        filename = filename.replace("</filename>\n","")
        lines[3] = "    <path>" + images_path + filename + "</path>\n"
        file = open(ann_path, "w")
        file.writelines(lines)
if naming_flag:
    for PATH in ann_paths:
        ann_path = annotations_path + PATH
        file = open(ann_path, "r")
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            line = re.sub("handle+\d", "handle", line)
            line = re.sub("handle+\d", "handle", line)
            line = re.sub("handle+\d", "handle", line)
            lines[i] = line
        file = open(ann_path, "w")
        file.writelines(lines)