from src.semantic_segmentation.loaders.dataloader import CityScapesDataset
from src.semantic_segmentation.models.unet import Unet
from src.semantic_segmentation.utils.fit_metrics import dice_loss, iou, pixel_acc
import logging
import numpy as np
from matplotlib import pyplot as plt
import sys
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.autograd import Variable
import time
from tqdm import tqdm
import gc

    
def train(model: torch.nn.Module, criterion: torch.nn.modules.loss._Loss, epochs: int, train_loader: CityScapesDataset,
          val_loader: CityScapesDataset, test_loader: CityScapesDataset, use_gpu: bool, name: str, debug: bool = False,
          start_epoch: int = 0):
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    if use_gpu:
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        model = torch.nn.DataParallel(model)
        model.to(device)

    train_loss_set = [0] * (epochs - start_epoch)
    val_loss_set = []
    val_acc_set = []
    val_iou_set = []
    
    # Early Stop criteria
    min_loss = 1e6
    min_loss_idx = 0
    earliest_stop_epoch = 20
    early_stop_delta = 7
    for epoch in range(start_epoch, epochs):
        ts = time.perf_counter()
        for iter, (inputs, tar, labels) in enumerate(train_loader):
            del labels
            optimizer.zero_grad()
            if use_gpu:
                inputs = inputs.to(device)
                tar = tar.to(device)

            outputs = model(inputs)
            del inputs
            loss = criterion(outputs, Variable(tar))
            del tar
            del outputs

            loss.backward()
            loss = loss.item()
            train_loss_set[epoch - start_epoch] += loss
            optimizer.step()

            if (iter % 10) == 0:
                logging.info(f"epoch: {epoch}\titer: {iter}\tloss: {loss}")

        # calculate val loss each epoch
        val_loss, val_acc, val_iou = val(model, val_loader, criterion, use_gpu)
        val_loss_set.append(val_loss)
        val_acc_set.append(val_acc)
        val_iou_set.append(val_iou)
        
        logging.info(f"epoch: {epoch}\ttime: {time.perf_counter() - ts}\ttrain loss: {loss}\tval loss: {val_loss}\tval acc: {val_acc}\tval iou: {val_iou}")
        
        torch.save(model, f"../assets/weights/{name}/epoch-{epoch}")
        
        # Early stopping
        if val_loss < min_loss:
            min_loss = val_loss
            min_loss_idx = epoch
            
        # If passed min threshold, and no new min has been reached for delta epochs
        elif epoch > earliest_stop_epoch and (epoch - min_loss_idx) > early_stop_delta:
            logging.info(f"Stopping early at {min_loss_idx}")
            break
        
    return train_loss_set, val_loss_set, val_acc_set, val_iou_set


def val(model: torch.nn.Module, val_loader: CityScapesDataset, criterion, use_gpu:bool = False):
    # set to evaluation mode 
    model.eval()
    softmax = torch.nn.Softmax(dim = 1)
    
    loss = []
    pred = []
    acc = []
    
    IOU_init = False
    if use_gpu:
        device = torch.device("cuda:0")
        
    for iter, (inputs, tar, labels) in tqdm(enumerate(val_loader)):
        
        if not IOU_init:
            IOU_init = True
            IOU = np.zeros((1,19))
        del tar
        
        if use_gpu:
            inputs = inputs.to(device)
            labels = labels.to(device)
            

            
        with torch.no_grad():   
            outputs = model(inputs)  
            del inputs
            loss.append(criterion(outputs, labels.long()).item())
            prediction = softmax(outputs) 
            del outputs
            acc.append(pixel_acc(prediction, labels))
            IOU = IOU + np.array(iou(prediction, labels))
            del prediction
            del labels
        
    
    acc = sum(acc)/len(acc)
    avg_loss = sum(loss)/len(loss) 
    IOU = IOU/iter  
    
    return avg_loss, acc, IOU      

    


def checkM(prev, q=False):
    out = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda and not q:
                    name = str(obj.size())
                    if name in out:
                        out[name] += 1
                    else:
                        out[name] = 1
                    
        except:
            pass
        
    for key in out:
        if key not in prev:
            logging.info("new: " + key + " : " + str(out[key]))
        elif prev[key] != out[key]:
            logging.info("diff (new - old): " + key + " : " + str(out[key])+ " - " +str(prev[key]))
            
    for key in prev:
        if key not in out:
            logging.info("dropped: " + key + " : " + str(prev[key]))
    return out

def memDiff(prev, out):
    for key in out:
        if key not in prev:
            logging.info("new: " + key + " : " + str(out[key]))
        elif prev[key] != out[key]:
            logging.info("diff (new - old): " + key + " : " + str(out[key] - prev[key]))
            
    for key in prev:
        if key not in out:
            logging.info("dropped: " + key + " : " + str(prev[key]))


def main():
    batch_size = 3
    train_dataset = CityScapesDataset(csv_file='../assets/datasets/train.csv', resize=True)
    val_dataset = CityScapesDataset(csv_file='../assets/datasets/val.csv', resize=True)
    test_dataset = CityScapesDataset(csv_file='../assets/datasets/test.csv')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=8,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=4,
                             shuffle=True)

    epochs = 100
    criterion = torch.nn.CrossEntropyLoss()
    # Fix magic number
    model = Unet(34)
    use_gpu = torch.cuda.is_available()
    name = "Unet"
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"train_{name}.log", mode='w+'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    train_loss_set, val_loss_set, val_acc_set, val_iou_set = train(model, criterion, epochs, train_loader, val_loader, test_loader, use_gpu, "Unet")

    plt.subplot(1, 2, 1)
    plt.plot(list(range(len(train_loss_set))), train_loss_set, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training')
    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(val_loss_set))), val_loss_set, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation')
    plt.show()



if __name__ == "__main__":
    main()
