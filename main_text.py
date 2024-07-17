import argparse
import os
import torch
import random
import numpy as np
import wandb
from config_text import config
import time
from tqdm import tqdm
import sys
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
#from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm

import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from train_utils import evaluate_model, train_epoch
from data_utils_text import data_prep_text

# Define the neural network architecture with additional features
class NN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # to seed the script globally

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("CUDA is available. 🚀")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: Name: {torch.cuda.get_device_name(i)}")
        print(torch.cuda.current_device()) 
    else:
        print("CUDA is not available.")

def create_log_dict(prefix, loss, accuracy, precision, recall, f1):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    values = [loss, accuracy, precision, recall, f1]
    return {prefix: {metric: value for metric, value in zip(metrics, values)}}
  
def get_new_model_path(base_path):
    dir_name, file_name = os.path.split(base_path)
    name, ext = os.path.splitext(file_name)
    i = 1
    while os.path.exists(os.path.join(dir_name, f"{name}_{i}{ext}")):
        i += 1
    return os.path.join(dir_name, f"{name}_{i}{ext}")

def run_training(num_epochs, is_sweep=False):
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if is_sweep:
        wandb.init(config=config.CONFIG_DEFAULTS, resume=False)
    else:
        wandb.init(project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS)# resume=False)

    # Prep data
    print('\n###### Preparing Dataset...')
    train_loader, val_loader, test_loader=data_prep_text()
    
    # Instantiate the model, loss function, optimizer, and scheduler
    print('\n###### Preparing Model...')
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # model = NeuralNetClassifier(module=NN(config.input_size, config.hidden_size1, config.hidden_size2, config.num_classes))# .to(device)
    model=NN(wandb.config.input_size, 
             wandb.config.hidden_size1, 
             wandb.config.hidden_size2, 
             wandb.config.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)
    criterion = nn.CrossEntropyLoss()
        
    # Training 
    initial_epoch = 1
    best_val_accuracy = 0.0
    best_val_loss = 0.0
    
    wandb.watch(model, log='all')

    print(f'\n##### Training starts.\nInitial epoch:{initial_epoch}\nTotal number of epoch: {num_epochs}')
    for epoch in tqdm(range(initial_epoch, initial_epoch + num_epochs), desc="Epochs"):
        epoch_start_time = time.time()
        print('Training...')
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, _, _=evaluate_model(model, val_loader, criterion,  device)
        #print(f"Epoch [{epoch}/]")
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch}/{initial_epoch}~{initial_epoch+num_epochs-1}] - Time: {epoch_duration:.2f}s")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"\nNew Best val accuracy found. Model saved to {config.MODEL_SAVE_PATH}")

        log_dict = {}
        log_dict.update(create_log_dict('train', train_loss, train_accuracy, train_precision, train_recall, train_f1))
        log_dict.update(create_log_dict('val', val_loss, val_accuracy, val_precision, val_recall, val_f1))
        
        wandb.log(log_dict, step=epoch)

    test_loss, test_accuracy, test_precision, test_recall, test_f1, _, _ = evaluate_model(model, test_loader, criterion, device)
    print(f"\n##### Training finished.\nTest Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}\nBest Val accuracy: {best_val_accuracy:.4f}, Best Val loss: {best_val_loss:.4f}")

    log_dict_test={}
    log_dict.update(create_log_dict('test', test_loss, test_accuracy, test_precision, test_recall, test_f1))
    
    wandb.log(log_dict_test)
    wandb.finish()

def main():
    print('\n#############\n')
    parser = argparse.ArgumentParser(description="Run emotion recognition model training")
    parser.add_argument("epochs", type=int, nargs='?', default=10, help="Number of epochs to train")
    parser.add_argument("--sweeps", type=int, help="Number of sweeps for hyperparameter search", default=0)
    args = parser.parse_args()
    if len(sys.argv) == 1:
        print(f"\nNo arguments provided. Using default config.")
    config.num_epochs=args.epochs    
  
    if args.sweeps > 0:

        sweep_id = wandb.sweep(sweep=config.CONFIG_SWEEP, project=config.WANDB_PROJECT)
        print(f'\nSweep starts. {config.num_epochs} epochs per sweep.\nSweep id: {sweep_id}\nTotal number of sweep: {config.num_epochs}\n')
        wandb.agent(sweep_id, function=lambda: run_training(args.epochs, is_sweep=True), count=args.sweeps)
    else:
        run_training(args.epochs, is_sweep=False)
        print(f"\nTraining for {config.num_epochs} epochs.")
        
def second():
    config.num_epochs=2
    num_sweep=2
    if num_sweep>0:
        sweep_id = wandb.sweep(sweep=config.CONFIG_SWEEP, project=config.WANDB_PROJECT)
        print(f'\nSweep starts. Sweep id: {sweep_id}\nTotal number of sweep: {config.num_epochs}\n')
        wandb.agent(sweep_id, function=lambda: run_training(config.num_epochs, is_sweep=True), count=num_sweep)
    else:
        run_training(config.num_epochs, is_sweep=False)
        print(f"\nTraining for {config.num_epochs} epochs.")
if __name__ == "__main__":
    main()