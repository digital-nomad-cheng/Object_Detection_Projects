from dataset import WiderFaceDetection


dataset = WiderFaceDetection("/home/idealabs/data/opensource_dataset/WIDER/WIDER_train/",
                             "label.txt")

for data in dataset:
    print(data)
