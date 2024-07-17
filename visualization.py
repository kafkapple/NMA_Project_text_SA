import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import  confusion_matrix
from sklearn.manifold import TSNE
import wandb
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('agg')

def plot_confusion_matrix(labels, preds, labels_emotion, normalize=True):
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=labels_emotion.values(), yticklabels=labels_emotion.values(), ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # wandb.log({"confusion_matrix": wandb.Image(fig)})
    #plt.close(fig)
    return fig

def visualize_embeddings(embeddings, labels, method='tsne'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Use 'pca' or 'tsne'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette="deep", legend="full", ax=ax)
    ax.set_title(f"{method.upper()} of Emotion Recognition Embeddings")
    
    # wandb.log({"embeddings": wandb.Image(fig)})
    # plt.close(fig)
    return fig
    

def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    labels_list = []
    
    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc="Extracting Embeddings"):
            features = features.to(device)
            outputs = model(features)
            embeddings.extend(outputs.cpu().numpy())
            labels_list.extend(labels.numpy())
    
    return np.array(embeddings), np.array(labels_list)

# In progress

# import shap
# import numpy as np

# def explain_predictions(model, data_loader, device):
#     # preparing background samples
#     background = next(iter(data_loader))[0][:100].to(device)  # 

#     # SHAP 
#     explainer = shap.DeepExplainer(model, background)

#     # Preparing dataset to be explained
#     test_data = next(iter(data_loader))[0][:10].to(device)  

#     # SHAP value 
#     shap_values = explainer.shap_values(test_data)

#     shap.summary_plot(shap_values, test_data.cpu().numpy(), plot_type="bar")

def perform_rsa(model, data_loader, device):
    model.eval()
    representations = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, batch_labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            representations.append(outputs.cpu().numpy())
            labels.extend(batch_labels.numpy())

    representations = np.vstack(representations)
    labels = np.array(labels)

    corr_matrix = np.corrcoef(representations)

    # Calculate label correlation matrix
    label_matrix = np.equal.outer(labels, labels).astype(int)

    # Calculate RSA correlation
    rsa_corr, _ = spearmanr(corr_matrix.flatten(), label_matrix.flatten())

    # Plot the representation correlation matrix and the label correlation matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(corr_matrix, cmap='coolwarm', ax=ax1)
    ax1.set_title("Representation Correlation Matrix")

    sns.heatmap(label_matrix, cmap='coolwarm', ax=ax2)
    ax2.set_title("Label Correlation Matrix")

    plt.suptitle(f"RSA Correlation: {rsa_corr:.2f}", fontsize=16)
    return fig