import os

#train images
os.system("curl http://images.cocodataset.org/zips/train2017.zip -o train_2017.zip")
os.system("unzip train_2017.zip")

#validation images
os.system("curl http://images.cocodataset.org/zips/val2017.zip -o val2017.zip")
os.system("unzip val2017.zip")

# #test images
# os.system("curl http://images.cocodataset.org/zips/test2017.zip -o test2017.zip")
# os.system("unzip test2017.zip")

#train and val annotations
os.system("curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o ann_train_2017.zip")
os.system("unzip ann_train_2017.zip")

# #test annotations
# os.system("curl http://images.cocodataset.org/annotations/image_info_test2017.zip -o ann_test2017.zip")
# os.system("unzip ann_test2017.zip")