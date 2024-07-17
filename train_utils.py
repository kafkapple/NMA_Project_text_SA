import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
from config_text import config
import os
from tqdm import tqdm
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for features, batch_labels in progress_bar:
        features, batch_labels = features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, accuracy, precision, recall, f1

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, batch_labels in tqdm(dataloader, desc="Evaluating"):
            features, batch_labels = features.to(device), batch_labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, batch_labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, accuracy, precision, recall, f1, all_labels, all_preds


def train_model(model, train_loader, val_loader, initial_epoch, num_epochs, criterion, optimizer, device, id_wandb):
    best_val_accuracy = 0
    
    for epoch in range(initial_epoch, initial_epoch + num_epochs):
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{initial_epoch}-{initial_epoch + num_epochs - 1}]")
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Model saved to {config.MODEL_SAVE_PATH}")
            
        # Checkpoint 저장
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'id_wandb': id_wandb
        }
        torch.save(ckpt, config.CKPT_SAVE_PATH)
        print(f"Checkpoint saved to {config.CKPT_SAVE_PATH}", ckpt['epoch'])
        
        wandb.log({
            "train": {
                "loss": train_loss,
                "accuracy": train_accuracy,
                "precision": train_precision,
                "recall": train_recall,
                "f1": train_f1
            },
            "validation": {
                "loss": val_loss,
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1
            }
        })
    
    return model


def load_checkpoint(ckpt_path, model, optimizer, device):
    print(ckpt_path)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        best_val_accuracy = checkpoint['best_val_accuracy']
        id_wandb=checkpoint['id_wandb']
        print(f"\n##### Checkpoint loaded: start epoch {start_epoch}, best validation accuracy {best_val_accuracy:.4f}")
        return model, optimizer, start_epoch, best_val_accuracy, id_wandb
    else:
        print("No checkpoint found.")
        return model, optimizer, 0, 0, wandb.util.generate_id()
    