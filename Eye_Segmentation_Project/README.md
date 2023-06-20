The goald here is to implement a UNET model and train him on 178x218 pixels face images:

Here is what the model looks like :

![image](https://github.com/Shifoue/Portfolio/assets/69169567/8eb8587b-0d9d-498c-9255-be0b518b7f7e)

First we have to refine our Data and create masks. I chose to use mediapipe inside my **Eye_dataset_segmentation.py** script which help me create mesh for each faces that i use to approximate a mask for the eyes.

Once the mask are created i split my data in two part :
  - Training data and mask
  - Validation data and mask

Finally i emplement my UNET architecture, i refine the Data using basic Data Augmentation and train the model.

For the loss function of my model i choose to implement a dice score (IoU) which force my model to predict as close as possible to the mask instead of opting for an all black or white strategy that would give him a good accuracy but would be pretty bad for what we want.

As a validation metrics i use IoU and it works fine.
