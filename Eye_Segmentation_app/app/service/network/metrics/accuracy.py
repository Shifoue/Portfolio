import torch

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0 #did i predict correctly
    num_pixels = 0 #how many pixel are correct Area accuracy

    #In order to force the algorithm on finding a good solution instead of only putting black pixel to get a good accuracy due to the small size of the target
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8) #calculate the intersection between ground truth and prediction

    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100}:.2f")
    print(f"Dice score {dice_score/len(loader)}")

    model.train()