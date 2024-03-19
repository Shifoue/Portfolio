

from data_processing.COCO.COCO_processing import create_dataloader



if __name__ == "__main__":
    train_informations = ("data_processing/COCO/train2017", "data_processing/COCO/annotations/captions_train2017.json")
    test_informations = ("data_processing/COCO/val2017", "data_processing/COCO/annotations/captions_val2017.json")

    train_dataloader, validation_dataloader, test_dataloader = create_dataloader(train_informations, test_informations)