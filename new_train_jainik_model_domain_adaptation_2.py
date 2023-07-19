import json
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from dataloader import BatchDataloader
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset, ConcatDataset
import torch.optim as optim
import numpy as np
import wfdb
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
import torch.nn.functional as F
import random
import pandas as pd
import argparse
import torch.optim as optim
from warnings import warn
from new_resnet_Jainik_model_domain_adaptation_2 import ResNet1d 


    ###############################################################################################################
    # train_Val_test_split
        
def train_val_test_split(dataset, train_ratio, val_ratio):
    """
    :param dataset: PyTorch dataset to split.
    :param train_ratio: Split ratio for the training data.
    :param val_ratio: Split ratio for the validation data.
    :return: train, validation, and test DataLoaders.
    """
    length = len(dataset)
    N_train = int(round(length * train_ratio))
    N_val = int(round(length * val_ratio))
    
    # Create random order of indices
    random_order = torch.randperm(length)
    
    # Split the dataset into train, validation, and test subsets
    train_indices = random_order[:N_train]
    val_indices = random_order[N_train:N_train + N_val]
    test_indices = random_order[N_train + N_val:]
    
    # Create Subset objects for each subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, val_dataset, test_dataset
        

    ###############################################################################################################
    # train_test_split    

def train_test_split(dataset, ratio):
    """
    :param dataset: PyTorch dataset to split.
    :param ratio: Split ratio for the data - e.g., 0.8 ==> 80% train and 20% test
    :return: train and test datasets.
    """

    length = len(dataset)
    random_order = torch.randperm(length)
    N_test = int(round(length * ratio))
    train_dataset = Subset(dataset, random_order[:N_test])
    test_dataset = Subset(dataset, random_order[N_test:])
    return train_dataset, test_dataset


    ###############################################################################################################
    # compute_loss_domain

def compute_loss_domain(domain_predictions, domain_labels):
    criterion_domain = nn.L1Loss()  
    domain_loss = criterion_domain(domain_predictions.squeeze(), domain_labels)
    return domain_loss

    ###############################################################################################################
    # compute_loss_age

def compute_loss_age(ages, pred_ages):
    ages = ages.view(-1)
    loss = nn.L1Loss()
    mae = 0.0
    count = 0
    for i in range(len(ages)):
        if ages[i]>= 18 and ages[i] <= 89 and not torch.isnan(pred_ages[i]).cpu().numpy():
            mae += loss(pred_ages[i], ages[i])
            count += 1
    if count > 0:
        mae /= count
    else:
        mae = 0
    return mae


    ###############################################################################################################
    # train

def train(epoch, dataload):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(epoch, 0, 0), position=0)

    for batch_idx, (combined_data, y, domain_labels) in enumerate(dataload):
        model.zero_grad()
        combined_data = combined_data.to(device)
        y = y.to(device)
        domain_labels = domain_labels.to(device)

        # Forward pass
        age_predictions, domain_predictions = model(combined_data)
        age_predictions = age_predictions.squeeze()
        if age_predictions.dim() == 0:
            age_predictions = age_predictions.view(1)
        y = y.float()
        
        age_predictions = age_predictions.squeeze()
        domain_predictions = domain_predictions.squeeze()
        # Calculate losses
        age_loss = compute_loss_age(y, age_predictions)
        domain_loss = compute_loss_domain(domain_predictions, domain_labels)

        # Update loss weights
        age_weight = 1.0  # Adjust the weight for the age loss
        domain_weight = 1.0  # Adjust the weight for the domain loss

        # Calculate total loss
        loss = age_weight * age_loss - domain_weight * (domain_loss)

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().cpu().numpy()
        n_entries += 1
        train_bar.desc = train_desc.format(epoch, total_loss / n_entries)
        train_bar.update(1)

        # Update train bar with batch_idx
        train_bar.set_postfix(batch_idx=batch_idx)

    total_loss = total_loss / n_entries
    train_bar.close()
    return total_loss


    ###############################################################################################################
    # eval

def eval(dataloader):
    model.eval()
    total_loss = 0
    n_entries = 0

    with torch.no_grad():
        for batch_idx, (combined_data, y, domain_labels) in enumerate(dataloader):
            combined_data = combined_data.to(device)
            y = y.to(device)
            domain_labels = domain_labels.to(device)

            # Forward pass
            logits, domain_predictions = model(combined_data)
            logits = logits.squeeze()
            if logits.dim() == 0:
                logits = logits.view(1)
            y = y.float()

            # Calculate losses
            age_loss = compute_loss_age(y, logits)
            domain_loss = compute_loss_domain(domain_predictions, domain_labels)

            # Update loss weights
            age_weight = 1.0  # Adjust the weight for the age loss
            domain_weight = 1.0  # Adjust the weight for the domain loss

            # Calculate total loss
            loss = age_weight * age_loss - domain_weight * (domain_loss)
            total_loss += loss.detach().cpu().numpy()
            n_entries += 1

    total_loss = total_loss / n_entries
    return total_loss


    ###############################################################################################################
    # ECGTransform

class ECGTransform(object):
        """
        This will transform the ECG signal into a PyTorch tensor. This is the place to apply other transformations as well, e.g., normalization, etc.
        """
        def __call__(self, signal):
            # Transform the data type from double (float64) to single (float32) to match the later network weights.
            t_signal = signal.astype(np.single)
            # We transpose the signal to later use the lead dim as the channel... (C,L).
            t_signal = torch.transpose(torch.tensor(t_signal), 0, 1)
            return t_signal  # Make sure I am a PyTorch Tensor
        

    ###############################################################################################################
    # ECGDataset

class ECGDataset(Dataset):

        def __init__(self, DB_path, table_path, Transform, num_examples = None):
            super().__init__()  # When using a subclass, remember to inherit its properties.
            # Define self.DB_path, self.table (with pandas reading csv) and self.transform (create an object from the transform we implemented):
            self.DB_path = DB_path
            self.table = pd.read_csv(table_path)
            self.transform = Transform
            self.num_examples = num_examples

        def get_wfdb_path(self, index):
            # Get the wfdb path as given in the database table:
            wfdb_path = self.DB_path + table['filename_lr'][int(index)]
            return wfdb_path

        def get_label(self, index):
            # A method to decide the label:
            age_str = self.table["age"][int(index)]
            age_float = float(age_str)
            return age_float

        def __getitem__(self, index):
            # Read the record with wfdb (use get_wfdb_path) and transform its signal. Assign a label by using get_label.
            record = wfdb.rdrecord(self.get_wfdb_path(index))
            signal = self.transform(record.p_signal)  # get tensor with the right dimensions and type.
            label = self.get_label(index)
            return signal, label

        def __len__(self):
            if self.num_examples is not None:
                return min(len(self.table), self.num_examples)
            else:
                return len(self.table)
    
    
    ###############################################################################################################
    # ECGDataset_Domain

class ECGDataset_Target_Domain(Dataset):
    def __init__(self, data_dir, Transform):
        self.transform = Transform
        self.data_dir = data_dir
        self.records = self._load_records()

    def _load_records(self):
        records_path = os.path.join(self.data_dir, 'RECORDS')
        with open(records_path, 'r') as f:
            records = [line.strip() for line in f]
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        
        hea_path = os.path.join(self.data_dir, record + '.hea')
        mat_path = os.path.join(self.data_dir, record)

        age = self._extract_age_from_hea(hea_path)
        ecg_data = self._load_ecg_data(mat_path)
        return ecg_data, age

    def _extract_age_from_hea(self, hea_path):
        age = None
        with open(hea_path, 'r') as f:
            for line in f:
                if line.startswith('# Age:'):
                    age_string = line.split(':')[1].strip()
                    if age_string != "NaN":
                        age = int(line.split(':')[1].strip())
                    else:
                        age = 20
                    break      
        return age
    
    def _load_ecg_data(self, mat_path):
        record = wfdb.rdrecord(mat_path)
        signals = record.p_signal
        num_leads = signals.shape[1]
        processed_signals = []

        for lead in range(num_leads):
            signal = signals[:, lead]
            # Ensure the signal has a length of 1000
            if len(signal) < 1000:
                # Pad the signal with zeros if it's shorter than 1000
                padding = np.zeros(1000 - len(signal))
                signal = np.concatenate((signal, padding))
            elif len(signal) > 1000:
                # Truncate the signal if it's longer than 1000
                signal = signal[:1000]
            processed_signals.append(signal)

        processed_signals = self.transform(np.array(processed_signals))
        return  torch.transpose(processed_signals, 0, 1)


    ###############################################################################################################
    #main

if __name__ == "__main__":


    ###############################################################################################################
    # Arguments that will be saved in config file

    # Define the number of random search iterations
    epochs = 50
    
    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                    description='Train model to predict age from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=100,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=1000,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=1, #default war 10 NB
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 450],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[1000, 500,250, 125,25],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='model/',
                        help='output folder (default: ./out)')
    parser.add_argument('--cuda', action='store_true',
                        help='use cuda for computations. (default: False)')
    parser.add_argument('--n_valid', type=int, default=100,
                        help='the first `n_valid` exams in the hdf will be for validation.'
                            'The rest is for training')

    args, unk = parser.parse_known_args()
        # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    torch.manual_seed(args.seed)
    device = torch.device('cuda:1')
    print(device)
    folder = args.folder
    # Generate output folder if needed
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    # Save config file
    with open(os.path.join(args.folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')
    tqdm.write("Building data loaders...")


###############################################################################################################
# loading data from source domain

    DB_path = '/home/stu15/MachineLearning_ageEstimation/Code/ptb-xl-a-large/'
    table_path = DB_path + 'ptbxl_database_new.csv'
    table = pd.read_csv(table_path)
    ECG_path = DB_path+'records100/'  # we will use the 100Hz recording version.
    list_of_folders = os.listdir(ECG_path)
    list_of_files = os.listdir(ECG_path+list_of_folders[0])
    record = wfdb.rdrecord(ECG_path + list_of_folders[0] + '/' + list_of_files[0][:-4])
    fs = record.fs
    lead_names = record.sig_name
    signal = record.p_signal
    print("Information about source domain:\n")
    print("There are total of %d folders in this source domain dataset"%len(list_of_folders))
    print("There are total of %d files in each folder of source domain\n"%len(list_of_files))
    print("We will use frequency of %d hertz throught this model" %fs)
    print("Each file has %d lead data\n"%len(lead_names))


################################################################################################################
# transforming data and doing test train validation split of source domain

    ecg_source_domain =  ECGDataset(DB_path, table_path, ECGTransform())
    print('The source domain dataset length is ' + str(len(ecg_source_domain)))
    # Reduce the number of samples in ecg_source_domain by specifying the indices to keep
    num_samples_to_keep = 21000  # Set the desired number of samples to keep
    indices_to_keep = range(num_samples_to_keep)
    ecg_source_domain = Subset(ecg_source_domain, indices_to_keep)
    print('The source domain dataset new length is ' + str(len(ecg_source_domain)))
    source_domain_train, source_domain_val, source_domain_test = train_val_test_split(ecg_source_domain, train_ratio=0.80, val_ratio = 0.1) 


################################################################################################################
# loading and transforming data and test train split of target domain

    target_transform = ECGTransform()
    ecg_target_domain1 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g1",Transform=target_transform)
    ecg_target_domain2 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g2",Transform=target_transform)
    ecg_target_domain3 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g3",Transform=target_transform)
    ecg_target_domain4 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g4",Transform=target_transform)
    ecg_target_domain5 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g5",Transform=target_transform)
    ecg_target_domain6 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g1",Transform=target_transform)
    ecg_target_domain7 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g2",Transform=target_transform)
    ecg_target_domain8 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g3",Transform=target_transform)
    ecg_target_domain9 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g4",Transform=target_transform)
    ecg_target_domain10 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g5",Transform=target_transform)
    ecg_target_domain11 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g6",Transform=target_transform)
    ecg_target_domain12 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g7",Transform=target_transform)
    ecg_target_domain13 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g8",Transform=target_transform)
    ecg_target_domain14 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/georgia/g9",Transform=target_transform)
    ecg_target_domain15 = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/ptb/g1",Transform=target_transform) 

    print('The domain 1 dataset length is ' + str(len(ecg_target_domain1)))
    print('The domain 2 dataset length is ' + str(len(ecg_target_domain2)))
    print('The domain 3 dataset length is ' + str(len(ecg_target_domain3)))
    print('The domain 4 dataset length is ' + str(len(ecg_target_domain4)))
    print('The domain 5 dataset length is ' + str(len(ecg_target_domain5)))
    print('The domain 6 dataset length is ' + str(len(ecg_target_domain6)))
    print('The domain 7 dataset length is ' + str(len(ecg_target_domain7)))
    print('The domain 8 dataset length is ' + str(len(ecg_target_domain8)))
    print('The domain 9 dataset length is ' + str(len(ecg_target_domain9)))
    print('The domain 10 dataset length is ' + str(len(ecg_target_domain10)))
    print('The domain 11 dataset length is ' + str(len(ecg_target_domain11)))
    print('The domain 12 dataset length is ' + str(len(ecg_target_domain12)))
    print('The domain 13 dataset length is ' + str(len(ecg_target_domain13)))
    print('The domain 14 dataset length is ' + str(len(ecg_target_domain14)))
    print('The domain 15 dataset length is ' + str(len(ecg_target_domain15)))


    ecg_target_domain = ConcatDataset([ecg_target_domain1]) #ecg_target_domain2, ecg_target_domain3, ecg_target_domain5, ecg_target_domain6, ecg_target_domain7, ecg_target_domain8, ecg_target_domain9, ecg_target_domain11, ecg_target_domain14, ecg_target_domain15])
    indices = list(range(len(ecg_target_domain))) # Create a list of indices
    random.shuffle(indices) # Shuffle the indices
    ecg_target_domain = Subset(ecg_target_domain, indices) # Create a new shuffled dataset using the shuffled indices and concat source and target domain
    target_domain_train, target_domain_val = train_test_split(ecg_target_domain, ratio = 0.9) 
    tqdm.write("Done!")
    tqdm.write("Defining model...")
    N_LEADS = 12  # the 12 leads
    N_Output = 1 #len(labels_train_set)  # just the age
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                        blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                        n_classes=N_Output,
                        kernel_size=args.kernel_size,
                        dropout_rate=args.dropout_rate)
    model.to(device=device)
    tqdm.write("Done!")
    tqdm.write("Define optimizer...")
    optimizer = optim.Adam(model.parameters(), args.lr)
    tqdm.write("Done!")
    tqdm.write("Define scheduler...")
    print("Batch size ",args.batch_size)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                    min_lr=args.lr_factor * args.min_lr,
                                                    factor=args.lr_factor)
        
    # Perform training and evaluation with the current hyperparameters
    source_dataloader_train = DataLoader(source_domain_train, batch_size=args.batch_size, shuffle=True)
    target_dataloader_train = DataLoader(target_domain_train, batch_size=args.batch_size, shuffle=True)
    source_dataloader_val = DataLoader(source_domain_val, batch_size=args.batch_size, shuffle=True)
    target_dataloader_val = DataLoader(target_domain_val, batch_size=args.batch_size, shuffle=True)

    combined_dataloader_train = []
    for source_batch_train, target_batch_train in zip(source_dataloader_train, target_dataloader_train):
        X_source_train, y_source_train = source_batch_train
        X_target_train, _ = target_batch_train
        source_domain_train_labels = torch.zeros(X_source_train.shape[0])  # Source domain train labels
        target_domain_train_labels = torch.ones(X_target_train.shape[0])  # Target domain train labels
        combined_X_train = torch.cat((X_source_train, X_target_train), dim=0)
        combined_y_train = y_source_train
        combined_domain_train_labels = torch.cat((source_domain_train_labels, target_domain_train_labels), dim=0)
        combined_train_batch = (combined_X_train, combined_y_train, combined_domain_train_labels)
        combined_dataloader_train.append(combined_train_batch)

    combined_dataloader_val = []
    for source_batch_val, target_batch_val in zip(source_dataloader_val, target_dataloader_val):
        X_source_val, y_source_val = source_batch_val
        X_target_val, _ = target_batch_val
        source_domain_val_labels = torch.zeros(X_source_val.shape[0])  # Source domain validation labels
        target_domain_val_labels = torch.ones(X_target_val.shape[0])  # Target domain validation labels
        combined_X_val = torch.cat((X_source_val, X_target_val), dim=0)
        combined_y_val = y_source_val
        combined_domain_val_labels = torch.cat((source_domain_val_labels, target_domain_val_labels), dim=0)
        combined_val_batch = (combined_X_val, combined_y_val, combined_domain_val_labels)
        combined_dataloader_val.append(combined_val_batch)    

print("training started")
for epoch in range(epochs):
    train_loss = train(epoch, combined_dataloader_train)
    valid_MAE = eval(combined_dataloader_val)
    print("validation loss: ",valid_MAE )


   