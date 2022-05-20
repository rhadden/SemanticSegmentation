from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import utils
from src.semantic_segmentation.loaders.dataloader import CityScapesDataset, labels_classes

def show_pred(model, dataset, idx, use_gpu=False):
    img, target, label = dataset.__getitem__(idx)
    img = img.unsqueeze(0)

    if use_gpu:
        device = torch.device("cuda:0")
        model = torch.nn.DataParallel(model)
        model.to(device)
        img.to(device)

    softmax = torch.nn.Softmax(dim = 1)
    prediction = softmax(model(img))
    prediction = torch.argmax(prediction, dim=1)

    colorized = np.zeros([prediction.shape[1], prediction.shape[2], 3])
    gt = np.zeros([prediction.shape[1], prediction.shape[2], 3])
    for i in range(colorized.shape[0]):
        for j in range(colorized.shape[1]):
            pred = prediction[0, i, j]
            colorized[i, j, :] = labels_classes[pred][7]
            gt[i, j, :] = labels_classes[label[i, j]][7]

    colorized = colorized / 255
    gt = gt / 255
    img_name = dataset.data.iloc[idx, 0]
    img = np.asarray(Image.open(img_name))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(colorized)
    plt.subplot(1, 3, 3)
    plt.imshow(gt)
    plt.show()

def main():
    import random
    dataset = CityScapesDataset(csv_file='../assets/datasets/val.csv', transforms=None)
    idx = random.randint(1, len(dataset) - 1)
    for i in range(0, 30, 5):
        model = torch.load(f"../assets/weights/Unet/epoch-{i}")
        show_pred(model, dataset, 0, True)
        model.cpu()

if __name__=="__main__":
    main()
