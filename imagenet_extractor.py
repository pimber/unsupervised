import tarfile as tar
import os
import shutil as sh

root = r"E:\imagenet\ILSVRC2012_img_train"
class_path = "data/imagenet_subsets/imagenet_50.txt"

if not os.path.exists(root):
    os.makedirs(root)

classes, subdirs = [], []
with open(class_path, 'r') as f:
    result = f.read().splitlines()
for line in result:
    subdir, class_name = line.split(' ', 1)
    subdirs.append(subdir)
    classes.append(class_name)

folder = tar.open(root + ".tar")
for subdir in folder.getmembers():
    if subdir.name.replace(".tar", "") not in subdirs:
        continue
    subdir_path = os.path.join(root, subdir.name.replace(".tar", ""))
    os.makedirs(subdir_path)
    subfolder = tar.open(fileobj=folder.extractfile(subdir))
    for img in subfolder.getmembers():
        file_path = os.path.join(subdir_path, "1")
        subfolder.extract(img, path=file_path)
        src = os.path.join(file_path, img.name)
        dst = os.path.join(subdir_path, img.name)
        sh.copyfile(src=src, dst=dst)
        sh.rmtree(file_path)
