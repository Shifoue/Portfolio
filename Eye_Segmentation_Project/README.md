The goal here is to implement a UNET model and train him on 178x218 pixels face images

Here is what the model looks like :

![image](https://github.com/Shifoue/Portfolio/assets/69169567/8eb8587b-0d9d-498c-9255-be0b518b7f7e)

First we have to refine our Data and create masks. I chose to use mediapipe inside my **Eye_dataset_segmentation.py** script which help me create mesh for each faces that i use to approximate a mask for the eyes.

Once the masks are created i split my data in two parts :
  - Training data and masks
  - Validation data and masks

Finally i emplement my UNET architecture, refine the Data using basic data augmentation and train the model.

For the loss function of my model i choose to implement a dice score (IoU) which forces my model to predict as close as possible to the masks instead of going for an all black or white strategy that would give him a good accuracy but would be pretty bad for what we want.

As validation metric i use the IoU and it works fine :

![image](https://github.com/Shifoue/Portfolio/assets/69169567/67614430-a71b-46a7-b3ad-2127fe637bfc)
