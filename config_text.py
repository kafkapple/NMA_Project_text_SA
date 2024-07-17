import os

class Config:
  
    SEED = 2024 # random seed
    ### Test
    input_size = 384  # size of word vectors
    hidden_size1 = 128
    hidden_size2 = 64
    num_classes = 6
    num_epochs = 10
    lr = 0.001
    ratio_train = 0.7
    ratio_test = 0.15

    ### Hyperparam - Initial default value
    BATCH_SIZE = 32
    MODEL = "text_model"

    #self.relu = nn.LeakyReLU(negative_slope=0.01)
    
    N_STEP_FIG = 2 # log visualization per step
    
    PROJECT_DIR="NMA_Project_text_SA"
    parent_dir = os.path.dirname(os.getcwd()) # assume that start from git folder of high level
    BASE_DIR = os.path.join(parent_dir, PROJECT_DIR)
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_NAME= "text"
    MODEL_NAME=f"text_model_{MODEL}_v4"#'wav2vec_v3_1'
    MODEL_DIR = os.path.join(BASE_DIR, 'model', MODEL_NAME)
    
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f'best_model_{MODEL_NAME}.pth')
    CKPT_SAVE_PATH = os.path.join(MODEL_DIR, f'checkpoint_{MODEL_NAME}.pth')
    
    LABELS_EMOTION = {
        0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
        4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
    }
    
    ####### Wandb config
    WANDB_PROJECT = f"{PROJECT_DIR}_test_v1"
    ENTITY="biasdrive-neuromatch"
    WANDB_NAME = MODEL_NAME
    # 2: Define the search space
    
    CONFIG_SWEEP = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "val.loss"},
        "parameters": {
            'BATCH_SIZE' : {"values":[16,32,64]},
        #  'input_size': {"values":[384, 512, 768]},  # Adjust based on your embedding dimensions
          'hidden_size1':{"values": [64, 128, 256]},
          'hidden_size2': {"values":[32, 64, 128]},
          'lr': {"values":[0.001, 0.01, 0.1]},
        }
    }
    CONFIG_DEFAULTS = {
    "resume":"allow",
    "architecture": f"{MODEL_NAME}",
    "dataset": f"{DATA_NAME}",
    "num_epochs": 2,
    "MODEL": MODEL,
    "lr": lr,
    "BATCH_SIZE":BATCH_SIZE,
    'input_size':input_size,
    'hidden_size1':hidden_size1,
    'hidden_size2':hidden_size2,
    'num_classes': num_classes  # Number of emotion classes
    }  
    def __init__(self):
        os.makedirs(self.BASE_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

config = Config()