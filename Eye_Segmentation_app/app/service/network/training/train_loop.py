import torch
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for i_batch, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            preds = model(data)
            loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())