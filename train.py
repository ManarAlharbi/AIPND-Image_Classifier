import torch
import torch.utils.data as data
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, models
import time
import copy 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str,help='Path to dataset')
parser.add_argument('--epochs',type=int,help='Number of epochs')
parser.add_argument('--arch',type=str,help='Model architecture')
parser.add_argument('--learning_rate',type=float,help='Learning rate')
parser.add_argument('--hidden_units',type=int,help='Number of hidden units')
parser.add_argument('--checkpoint',type=str,help='Save trained model checkpoint to file')
parser.add_argument('--gpu',action='store_true',help='Use GPU if available')

args, _ = parser.parse_known_args()



def load_model(arch='vgg19', num_labels=102, hidden_units=4096):
    if arch=='vgg19':
        model = models.vgg16(pretrained=True)
    elif arch=='densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', arch)
    
    
    for param in model.parameters():
        param.requires_grad = False
        
    
    features=list(model.classifier.children())[:-1]
    
   
    num_features= model.classifier[len(features)].in_features
    
    
    features.extend([
        nn.Dropout(), 
        nn.Linear(num_features, hidden_units),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(True),
        nn.Linear(hidden_units, num_labels),
        nn.Softmax(dim=1)
    ])
    
    model.classifier =  nn.Sequential(*features)
    
    return model

def train_model(image_datasets, arch='vgg19', hidden_units=4096, epochs=20, learning_rate=0.001, gpu=False, checkpoint=''):
    
    if args.arch:
        arch=args.arch
        
    if args.hidden_units:
        hidden_units=args.hidden_units
    
    if args.epochs:
        epochs=args.epochs
        
    if args.learning_rate:
       learning_rate=args.learning_rate  
    
    if args.gpu:
        gpu=args.gpu
        
    if args.checkpoint:
        checkpoint=args.checkpoint
        
   
    dataloaders= {
        x: data.Dataloader(image_datasets[x], batch_size=4, shuffle=True, num_workers=2)
        for x in list(image_datasets.keys())
		         }
    
    
    dataset_sizes={
        x: len(dataloaders[x].dataset)
        for x in list(image_datasets.keys())
        }    
        
    num_labels= len(image_datasets['train'].classes)
    model= load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

 
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")
        model.cuda()
    
    else:
        print('Using CPU for training')
        device = torch.device("cpu")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler= lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    since=time.time()

    best_model_weights= copy.deepcopy(model.state_dict())
    best_accuracy=0.0
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1 , epochs))
        print('_' * 10)
    
    for phase in ['train','valid']:
        if phase == 'train' :
            scheduler.step()
            model.train()
        else:
            model.eval()
        
        running_loss= 0.0
        running_correct = 0
        
        
        for inputs, labels in dataloaders[phase]:
            inputs= inputs.to(device)
            labels= labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs= model(inputs)
                _, preds = torch.max(outputs, 1)
                loss= criterion(outputs, labels)
                
              
                if phase == 'train':
                    loss.backward()
                    optimizer.stop()
            
            
            running_loss+=loss.item() * input.size(0)
            running_correct += torch.sum(preds == labels.data)
            
        epoch_loss=running_loss / dataset_sizes[phase]
        epoch_acc=running_correct.double() / dataset_sizes[phase]
        
        print('{} Loss:{:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
    
    if phase == 'valid' and epoch_acc > best_accuracy:
        best_accuracy=epoch_acc
        best_model_weights= copy.deepcopy(model.state_dict())
        
    print()
    time_elapsed= time.time() - since
    print('Training complete in {:0f}m {0f}s'.format(time_elapsed // 60, time_elapsed %60))
    print('Best val Acc: {:4f}'.format(nest_acc))
    
   
    model.load_state_dict(best_model_weights)
    
   
    model.class_to_idx= image_datasets['train'].class_to_idx
    
  
    if checkpoint:
        print('Saving checkpoint to:', checkpoint)
        checkpoint_dict= {
            'arch': arch, 
            'class_to_idx': model.class_to_idx, 
            'state_dict':model.state_dict(), 
            'hidden_units':hidden_units
        }
        
        torch.save(checkpoint_dict, checkpoint)
    
        return model

if args.data_dir:
    train_model(args.data_dir)