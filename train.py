"""
Algorithm to Train, Save and Continue Training of Segmentation Models

This is a training algorithm for segmentation tasks, that can be used to
perform five tasks (training, saving, and testing models, but alose continue a
training and saving image examples comparing prediction and label) in a diverse
range of models. Next we will see which are the specific models applied using
these algorithms, but we can adapt them to train a range of other segmenta-
tion models, with just small changes (for classification models, please refers
to the algorithms in the folder 'Classification\train.py').

This program runs together with the files 'utils.py', 'model.py' and 'dataset.
py' to perform the training of the UResNet models (from 18 to 152 layers, as
specified at 'model.py'), which are based in the encoder-decoder architecture,
from the Raabin Dataset of Nucleus and Cytoplasm Ground Truths (available for
download at https://raabindata.com/free-data/). To train other models, just
specify your model at 'model.py', or simply load your model in the variable
'model' inside this algorithm. To use other datasets, you need to change the
file 'dataset.py' to load the new dataset as a 'torch.utils.data.Dataset'
instance (see torch documentation for more information at
https://pytorch.org/tutorials/beginner/basics/data_tutorial).

Standard Training: To train a model from zero (or from the first epoch), just
define the hyperparameters, as needed, and choose 'continue_training = False',
and 'last_epoch = 0' (parameter only used for continue a training). In order to
save your model, change 'save_model' to True, and to save images resultesd from
the segmentaiton (from the validation dataset), in the folder 'saved_images'
inside the 'root_folder', change 'save_images' to True. Variables 'test_model'
and 'load_model' are for other popouses (see options below), and can set to
False during the first training.

Continue a Training: set 'continue_training = True' in the hyperparameters to
continue a training, also setting 'last_epoch' with the number of epochs
already trained (e.g. if you trained 10 epochs, and want to continue, set
'last_epoch = 10'. Also the name of the pre-trained model has to exactly match
chekpoint_dir in the 'root_folder' directory, and the 'csv' file with
previous results, 'dictionary.csv', also has to be in 'root_folder'. The varia-
ble 'laod_model' does not need to be 'True' (it is just to test, see below).

Testing models: If you only want to test one or more models, just set
'test_models = True', and specify the directory where the models to be tested
are as a string in the variable 'test_models_dir'. If other options are also
chosen, the test will take place in the and, after the other options finish.

Loading and Testing One Model: if you want to test a model before continue a
training, or just wants to load and test one model, choose 'load_model = True'.
This will test the model chekpoint_dir stored in the 'root_folder'.


Find more on the GitHub Repository:
https://github.com/MarlonGarcia/attacking-white-blood-cells


@author: Marlon Rodrigues Garcia
@instit: University of São Paulo
"""

### Program  Header

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import numpy as np
import pandas as pd
import time
# If running on Colabs, mounting drive
run_on_colabs = True
if run_on_colabs:
    # Importing Drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    # To import add current folder to path (import py files):
    import sys
    root_folder =     '//content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Desenvolvido pelos Pesquisadores/Thales Pimentel Zuanazzi/Segmentação/teste_dice_optimal'
    test_models_dir = '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Desenvolvido pelos Pesquisadores/Thales Pimentel Zuanazzi/Segmentação/teste_dice_optimal'
    chekpoint_dir = 'my_checkpoint6.pth.tar'
else:
    root_folder = 'D:/Users/Thales/Documents/TCC/IA/main/attacking-white-blood-cells/Segmentation'
    test_models_dir = 'D:/Users/Thales/Documents/TCC/IA/main/attacking-white-blood-cells/Segmentation/results'
    chekpoint_dir = 'D:/Users/Thales/Documents/TCC/IA/main/attacking-white-blood-cells/Segmentation/results/my_checkpoint33.pth.tar'

# defining where to save results
if run_on_colabs:
    save_results_dir = '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Desenvolvido pelos Pesquisadores/Thales Pimentel Zuanazzi/Segmentação/teste_dice_optimal'
else:
    save_results_dir = 'D:/Users/Thales/Documents/TCC/IA/main/attacking-white-blood-cells/Segmentation/results'
import os
os.chdir(root_folder)

from model import *
from utils import *
from model import *
from utils import *


#%% Defining Parameters and Path

# defining hyperparameters
learning_rate = 1e-4    # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 6          # batch size
num_epochs = 30         # number of epochs
num_workers = 3         # number of workers (smaller or = n° processing units)
clip_train = 1.00       # percentage to clip the train dataset (for tests)
clip_valid = 1.00       # percentage to clip the valid dataset (for tests)
valid_percent = 0.15    # use a percent of train dataset as validation dataset
test_percent = 0.15     # a percent from training dataset (but do not excluded)
start_save = 0          # epoch to start saving
image_height = 512      # height to crop the image
image_width = 640       # width to crop the image
pin_memory = True
load_model = True       # 'true' to load a model and test it, or use it
save_model = True       # 'true' to save model trained after epoches
continue_training = True # 'true' to load and continue training a model
save_images = True      # saving example from predicted and original
test_models = False     # true: test all the models saved in 'save_results_dir'
last_epoch = 6        # when 'continue_training', it has to be the last epoch

# defining the paths to datasets
train_image_dir = ['/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/01',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/02',
                      #  '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/03',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/04',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/05',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/abdominal_wall/06',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/abdominal_wall/07',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/pancreas/08',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/abdominal_wall/09',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/abdominal_wall/10',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/11',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/12',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/13',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/abdominal_wall/14',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/15',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/abdominal_wall/16',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/abdominal_wall/17',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/18',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/abdominal_wall/19',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/20',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/21',
                      #  '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/22',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/abdominal_wall/23',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/24',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/25',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/26',
                       #'/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/pancreas/27',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/28',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/29',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/30',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/31'
                       ]

val_image_dir = ['/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/03',
                       '/content/gdrive/Shareddrives/Lab. de Óptica Biomédica/Datasets/DSAD/liver/22'
                       ]


#%% Training Function

# defining the training function
def train_fn(loader, model, optimizer, loss_fn, scaler, schedule, epoch, last_lr):
    loop = tqdm(loader, desc='Epoch '+str(epoch+1))

    for batch_idx, (dictionary) in enumerate(loop):
        image, label = dictionary
        x, y = dictionary[image], dictionary[label]
        x, y = x.to(device=device), y.to(device=device)
        y = y.float()
        # forward
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.autocast('cpu'):
            pred = model(x)
            # cropping 'pred' for when the model changes the image dimensions
            y = tf.center_crop(y, pred.shape[2:])
            # calculating loss
            loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        if device == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
        # if device='cpu', we cannot use 'scaler=torch.cuda.amp.GradScaler()':
        else:
            loss.backward()
            optimizer.step()
        # freeing space by deliting variables
        loss_item = loss.item()
        del loss, pred, y, x, image, label, dictionary
        # updating tgdm loop
        loop.set_postfix(loss=loss_item)
    # deliting loader and loop
    del loader, loop
    # scheduling the learning rate and saving its last value
    if scaler:
        if scale >= scaler.get_scale():
            schedule.step()
            last_lr = schedule.get_last_lr()
    else:
        schedule.step()
        last_lr = schedule.get_last_lr()

    return loss_item, last_lr


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # Standard cross-entropy loss
        loss = self.ce_loss(predictions, targets)

        # Check if there are positive targets
        positive_targets = (targets > 0).any()

        # If no positive targets are present, return 0 loss
        if not positive_targets:
            return torch.tensor(0.0, requires_grad=True).to(predictions.device)
        else:
            return loss



#%% Defining The main() Function
def main():
        # defining the model and casting to device
    model = UResNet34(in_channels=3, num_classes=2).to(device)
    # if binary classification, use BCEWithLogitsLoss and do not use logistic
    # function inside the model (this loss has logistic already).
    # loss_fn = nn.BCEWithLogitsLoss()
    # for multiclass segmentation, use e.g. CrossEntropyLoss, and a logistic
    # function inside the model (in its output), also changing the number of
    # classes as desired in the model defined above (e.g. num_classes=3).
    loss_fn = nn.L1Loss()
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = CustomCrossEntropyLoss()
    # pass 'lr=learning_rate' to Adam optim. to consider it, but it ahs its own
    # way to schedule learning rate, so here it is not considered.
    optimizer = optim.Adam(model.parameters())
    # SGD it is more stable, but has a lower accuracy in this segmentation:
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # if schedule is not used, please refer it as 'None'
    # schedule = None

    # loading dataLoaders
    train_loader, test_loader, valid_loader = get_loaders(
        train_image_dir=train_image_dir,
        valid_percent=valid_percent,
        test_percent=test_percent,
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width,
        num_workers=num_workers,
        pin_memory=pin_memory,
        val_image_dir=val_image_dir,
        clip_valid=clip_valid,
        clip_train=clip_train
    )

    # if this program is just to load and test a model, next it loads a model
    if load_model:
        # loading checkpoint
        os.chdir(root_folder)
        if device == 'cuda':
            load_checkpoint(torch.load(chekpoint_dir), model)
        # if 'cpu', we need to pass 'map_location'
        else:
            load_checkpoint(torch.load(chekpoint_dir,
                                       map_location=torch.device('cpu')), model)
        check_accuracy(valid_loader, model, loss_fn, device=device)

    if not load_model or continue_training:
        # changing folder to save dictionary
        os.chdir(save_results_dir)
        # if 'continue_training==True', we load the model and continue training
        if continue_training:
            print('\n- Continue Training...\n')
            start = time.time()
            if device == 'cuda':
                load_checkpoint(torch.load(chekpoint_dir), model,
                                optimizer=optimizer)
            else:
                load_checkpoint(torch.load(chekpoint_dir,
                                           map_location=torch.device('cpu')),
                                           model, optimizer=optimizer)
            # reading the csv 'dictionary.csv' as a dictionary
            df = pd.read_csv('dictionary.csv')
            temp = df.to_dict('split')
            temp = temp['data']
            dictionary = {'acc-valid':[], 'acc-test':[], 'loss':[], 'dice score-valid':[], 'dice score-test':[], 'time taken':[]}
            for acc_valid, acc_test, loss, dice_score_valid, dice_score_test, time_item in temp:
                dictionary['acc-valid'].append(acc_valid)
                dictionary['acc-test'].append(acc_test)
                dictionary['loss'].append(loss)
                dictionary['dice score-valid'].append(dice_score_valid)
                dictionary['dice score-test'].append(dice_score_test)
                dictionary['time taken'].append(time_item)
            # adding a last time to continue conting from here
            last_time = time_item
        # if it is the first epoch
        elif not continue_training:
            print('\n- Start Training...\n')
            start = time.time()
            # opening a 'loss' and 'acc' list, to save the data
            dictionary = {'acc-valid':[], 'acc-test':[], 'loss':[], 'dice score-valid':[], 'dice score-test':[], 'time taken':[]}
            acc_item_valid, loss_item, dice_score_valid = check_accuracy(valid_loader, model, loss_fn, device=device, title='Validating')
            acc_item_test, _, dice_score_test = check_accuracy(test_loader, model, loss_fn, device=device, title='Testing')
            print('\n')
            dictionary['acc-valid'].append(acc_item_valid)
            dictionary['acc-test'].append(acc_item_test)
            dictionary['loss'].append(loss_item)
            dictionary['dice score-valid'].append(dice_score_valid)
            dictionary['dice score-test'].append(dice_score_test)
            # we added last_time here to sum it to the 'time taken' in the
            # dictionary. it is done because if training is continued, we can
            # sum the actual 'last_time' taken in previous training.
            last_time = (time.time()-start)/60
            dictionary['time taken'].append(last_time)

        # with 'cpu' we can't use 'torch.cuda.amp.GradScaler()'
        if device == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None
        # to use 'last_lr' in 'train_fn', we have to define it first
        last_lr = schedule.get_last_lr()
        # begining image printing
        fig, ax = plt.subplots()
        # Criating a new start time (we have to sum this to 'last_time')
        start = time.time()

        # running epochs
        for epoch in range(last_epoch, num_epochs):
            # calling training function
            loss_item, last_lr = train_fn(train_loader, model, optimizer,
                                          loss_fn, scaler, schedule, epoch,
                                          last_lr)
            # appending resulted loss from training
            dictionary['loss'].append(loss_item)
            # saveing model
            if save_model and epoch >= start_save -1:
                # changing folder to save dictionary
                os.chdir(save_results_dir)
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename='my_checkpoint'+str(epoch+1)+'.pth.tar')
            # check accuracy
            print('\nValidating:')
            acc_item_valid, _, dice_score_valid = check_accuracy(valid_loader, model, loss_fn, device=device)
            print('Testing:')
            acc_item_test, _, dice_score_test = check_accuracy(test_loader, model, loss_fn, device=device)
            stop = time.time()
            dictionary['acc-valid'].append(acc_item_valid)
            dictionary['acc-test'].append(acc_item_test)
            dictionary['dice score-valid'].append(dice_score_valid)
            dictionary['dice score-test'].append(dice_score_test)
            dictionary['time taken'].append((stop-start)/60+last_time)
            # saving some image examples to specified folder
            if save_images:
                # criating directory, if it does not exist
                os.chdir(root_folder)
                try: os.mkdir('saved_images')
                except: pass
                save_predictions_as_imgs(
                    valid_loader, model, folder=os.path.join(root_folder,'saved_images'),
                    device=device
                )
            # saving dictionary to a csv file
            if save_model:
                # changing folder to save dictionary
                os.chdir(save_results_dir)
                df = pd.DataFrame(dictionary, columns = ['acc-valid', 'acc-test',
                                                         'loss', 'dice score-valid',
                                                         'dice score-test', 'time taken'])
                df.to_csv('dictionary.csv', index = False)

            print('\n- Time taken:',round((stop-start)/60+last_time,3),'min')
            print('\n- Last Learning rate:', round(last_lr[0],8),'\n\n')
            # deleting variables for freeing space
            del dice_score_test, dice_score_valid, acc_item_test, acc_item_valid,
            loss_item, stop
            try: del checkpoint
            except: pass

            # continue image printing
            if epoch == last_epoch:
                ax.plot(np.asarray(dictionary['acc-valid']), 'C1', label ='accuracy-validation')
                ax.plot(np.asarray(dictionary['acc-test']), 'C2', label ='accuracy-test')
                ax.plot(np.asarray(dictionary['dice score-valid']), 'C4', label = 'dice score-validation')
                ax.plot(np.asarray(dictionary['dice score-test']), 'C5', label = 'dice score-test')
                ax.plot(np.asarray(dictionary['loss']), 'C3', label = 'loss')
                plt.legend()
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Accuracy, Loss, and Dice score')
                plt.pause(0.5)
            else:
                ax.plot(np.asarray(dictionary['acc-valid']), 'C1')
                ax.plot(np.asarray(dictionary['acc-test']), 'C2')
                ax.plot(np.asarray(dictionary['dice score-valid']), 'C4')
                ax.plot(np.asarray(dictionary['dice score-test']), 'C5')
                ax.plot(np.asarray(dictionary['loss']), 'C3')
            plt.show()
            plt.pause(0.5)


if __name__ == '__main__':
    main()

#%% Defining test function

def testing_models():

    model = UResNet50(in_channels=3, num_classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # getting the 'valid_loader'
    e, train_loader, valid_loader = get_loaders(
        train_image_dir=train_image_dir,
        # csv_file_train=csv_file_train,
        valid_percent=valid_percent,
        test_percent=test_percent,
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width,
        num_workers=num_workers,
        pin_memory=pin_memory,
        clip_valid=clip_valid,
        clip_train=clip_train
    )

    loss_fn = nn.CrossEntropyLoss()

    os.chdir(test_models_dir)
    for file in os.listdir(test_models_dir):
        print('\n\n')
        if 'my_checkpoint' in file:
            # checking accuracy
            if device == 'cuda':
                load_checkpoint(torch.load(file), model)
            else:
                load_checkpoint(torch.load(file,
                                           map_location=torch.device('cpu')),
                                           model)
            print('\n- Model:', file)
            acc, loss,dice = check_accuracy(valid_loader, model, loss_fn, device=device)
            # acc_item_test, _, dice_score_test = check_accuracy(test_loader, model, loss_fn, device=device)
            print('\n- Acc:',round(acc,3),'; loss:',round(loss,3),'; dice:',round(dice,3))

# if (__name__ == '__main__') and (test_models == True):
#     testing_models()
