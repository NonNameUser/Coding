

from Model import SPPNet
import torch
import logging
from dataset import TrainDataSet, TestDataSet
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
receiver1_path = '.../data/receiver1'
receiver2_path = '.../data/receiver2'
receiver1_path2 = '.../test_data/receiver1'
receiver2_path2 = '.../test_data/receiver2'
dir_checkpoint = Path('./checkpoints/')
def confusion_matrix(ture_label,pre_label,n_class):
    ret=np.zeros((n_class,n_class))
    for i in range(len(ture_label)):
        ret[ture_label[i]][pre_label[i]]+=1
    return ret

def my_collate(batch):
    receiver1_data = [item[0] for item in batch]
    receiver2_data = [item[1] for item in batch]
    label_data = [item[2] for item in batch]
    label_data=torch.stack(label_data)
    return [receiver1_data, receiver2_data, label_data]
def preprocess(batch):
    list_receiver1_data=batch[0]
    list_receiver2_data = batch[1]
    tensor_label_data=batch[-1]
    # Original data size
    receiver1_data_sizes_orig, receiver2_data_sizes_orig = [], []
    for i in list_receiver1_data:
        receiver1_data_sizes_orig.append(i.shape[-1])
    for i in list_receiver2_data:
        receiver2_data_sizes_orig.append(i.shape[-1])
    receiver1_data_sizes_orig_max_size = max(size for size in receiver1_data_sizes_orig)
    receiver2_data_sizes_orig_max_size = max(size for size in receiver2_data_sizes_orig)
    padded_batch = [
        torch.stack([torch.cat((receiver1_data, torch.tensor(np.zeros((receiver1_data.shape[0],receiver1_data.shape[1],receiver1_data_sizes_orig_max_size-receiver1_data.shape[-1])))), 2) for receiver1_data in batch[0]]),
        torch.stack([torch.cat((receiver2_data, torch.tensor(np.zeros((receiver2_data.shape[0],receiver2_data.shape[1],receiver2_data_sizes_orig_max_size-receiver2_data.shape[-1])))), 2) for receiver2_data in batch[1]]),
        tensor_label_data]

    return padded_batch
def train_net(model,
              device,
              n_class,
              epochs: int = 100,
              batch_size: int = 53,
              learning_rate: float = 1e-5,
              save_checkpoint: bool = True,
              amp: bool = False):

    # Create dataset
    try:
        dataset = TrainDataSet(receiver1_dir=receiver1_path, receiver2_dir=receiver2_path, n_class=n_class)
    except (AssertionError, RuntimeError):
        dataset = TrainDataSet(receiver1_dir=receiver1_path, receiver2_dir=receiver2_path, n_class=n_class)
    train_percent = 0.7
    n_train = int(len(dataset) * train_percent)
    n_test = len(dataset) - n_train
    train_dataset, test_dataset = random_split(dataset=dataset,
                                               lengths=[n_train, n_test],
                                               generator=torch.Generator().manual_seed(0))
    # Load Dataset
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, collate_fn=my_collate)

    # Initialize Log
    logging.info(f'''Starting training:            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Mixed Precision: {amp}
        ''')
    # Setting loss function
    loss_fun = torch.nn.CrossEntropyLoss()
    # Setting optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    # Setting learning decline strategy
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 250, gamma=0.1, last_epoch=-1)
    global_step = 0
    # Training
    total_test_step = 0
    test_loss = []
    test_accuracy=[]
    RESUME=1
    satart_epoch=1
    if RESUME:
        path_checkpoint = ".../gait_identification_model.pth"  # Breakpoint path
        checkpoint = torch.load(path_checkpoint)  # Loading breakpoints
        model.load_state_dict(checkpoint)  # Loading model learnable parameters
        satart_epoch =1
    for epoch in range(satart_epoch, epochs+1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='gait cycle img') as pbar:
            for batch in train_loader:
                batch = preprocess(batch)
                receiver1_data = batch[0]
                receiver2_data = batch[1]
                label=batch[2]
                receiver1_data = receiver1_data.to(device=device, dtype=torch.float32)
                receiver2_data = receiver2_data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)

                optimizer.zero_grad()
                output = model(receiver1_data, receiver2_data)  # 放入模型训练

                loss = loss_fun(output, label)

                loss.backward()
                optimizer.step()
                pbar.update(receiver1_data.shape[0])

                epoch_loss += loss.item()
                global_step += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        test_data_size = len(test_dataset)
        total_test_loss = 0
        total_accuracy = 0
        conf_matrix = np.zeros((n_class, n_class))
        model.eval()
        with tqdm(total=n_test, desc=f'Epoch {epoch}/{epochs}', unit='gait cycle img') as pbar1:
            with torch.no_grad():
                for batch in test_loader:
                    batch = preprocess(batch)
                    receiver1_data = batch[0]
                    receiver2_data = batch[1]
                    label = batch[2]
                    receiver1_data = receiver1_data.to(device=device, dtype=torch.float32)
                    receiver2_data = receiver2_data.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.long)
                    output = model(receiver1_data, receiver2_data)
                    pbar1.update(receiver1_data.shape[0])
                    loss = loss_fun(output, label)
                    predictions = output.argmax(dim=1)
                    conf_matrix += confusion_matrix(label.data.cpu().numpy(), predictions.data.cpu().numpy(), n_class)
                    total_test_loss = total_test_loss + loss.item()
                    accuracy = torch.eq(predictions, label).float().sum().item()
                    total_accuracy = total_accuracy + accuracy
                    pbar1.set_postfix(**{'loss (batch)': loss.item()})
        test_loss.append(total_test_loss)
        test_accuracy.append(total_accuracy / test_data_size)
        print('Loss on the overall test set:{}'.format(total_test_loss))
        print('The accuracy rate on the overall test set is:{}'.format(total_accuracy/ test_data_size))

        s = 'epoch:{}, train_acc:{}, train_loss:{}'.format(epoch, total_accuracy/ test_data_size, total_test_loss)
        with open('user_user2_1.txt', 'a') as f:
            f.write(s)
            f.write('\n')
        total_test_step += 1
        c_m='epoch:{}, mix:\n{}'.format(epoch , conf_matrix)
        with open('conf_matrix_user2_1.txt', 'a') as f:
            f.write(c_m)
            f.write('\n')
        print('train finish!')
        # Store weights
        torch.save(model.state_dict(), './gait_identification_model_user2_1.pth')

        if save_checkpoint & epoch%100==0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
    plt.figure()
    plt.plot(range(len(test_loss)), test_loss, 'b-')
    plt.show()
    plt.figure()
    plt.plot(range(len(test_accuracy)), test_accuracy, 'r-')
    plt.show()
def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    #number of categories
    classes=2
    MODEL = SPPNet(backbone=18, num_classes=classes, pool_size=(1, 2, 6), weights=None)
    logging.info(f'Network:\n'
                 f'\t{MODEL.backbone} input backbonen\n'
                 f'\t{MODEL.n_classes} output channels (classes)\n'
                 f'\t{"Weights" if MODEL.weights else "Weights=None"}')
    device = torch.device('cuda')
    MODEL.to(device=device)
    logging.info(f'Using device {device}')
    train_net(model=MODEL, device=device, n_class=classes)

def test():
    for i in range(20):
        torch.cuda.synchronize()
        begin = time.time()
        device = torch.device('cuda')
        path1 = '.../receiver1'
        path2 = '.../receiver2'
        dataset = TestDataSet(receiver1_dir=path1, receiver2_dir=path2)
        train_percent = 0.3
        n_train = int(len(dataset) * train_percent)
        n_test = len(dataset) - n_train
        train_dataset, test_dataset = random_split(dataset=dataset,
                                                   lengths=[n_train, n_test],
                                                   generator=torch.Generator().manual_seed(i))
        n_test = len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False, collate_fn=my_collate)
        net = SPPNet(backbone=18, num_classes=2, pool_size=(1, 2, 6))
        net.load_state_dict(torch.load(".../gait_identification_model.pth"))
        net.to(device)
        net.eval()
        total_true = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = preprocess(batch)
                receiver1_data = batch[0]
                receiver2_data = batch[1]
                label = batch[2]
                receiver1_data = receiver1_data.to(device=device, dtype=torch.float32)
                receiver2_data = receiver2_data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                output = net(receiver1_data, receiver2_data)
                predictions = output.argmax(dim=1)
                total_true += torch.eq(predictions, label).float().sum().item()
        accuracy = total_true / n_test
        print(accuracy)
        torch.cuda.synchronize()
        end = time.time()


if __name__ == '__main__':

    main()
    #test()


