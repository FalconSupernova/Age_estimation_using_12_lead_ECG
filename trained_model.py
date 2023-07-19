import json
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wfdb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from warnings import warn
from new_resnet_Jainik_model_domain_adaptation_4 import ResNet1d 

###############################################################################################################
# Plotting

def create_scatter_plot(true_labels, final_predicted_ages, folder):
    # Create a scatter plot
    plt.scatter(final_predicted_ages, true_labels, color='blue', label='Data Points')

    # Add a diagonal line
    x = np.linspace(0, np.max(true_labels), 100)
    plt.plot(x, x, color='red', label='perfect value')

    # Set labels and title
    plt.xlabel('Logits')
    plt.ylabel('True Age')
    plt.title('True age vs. predicted age')
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    # Save the scatter plot as an image
    plt.savefig(os.path.join(folder, 'scatter_plot_test_set.png'))
    plt.clf()

def create_confusion_matrix(true_labels, final_predicted_ages, folder):
    interval = 10
    age_intervals = range(10, 100, interval)

    predicted_age_intervals = np.digitize(np.round(final_predicted_ages), age_intervals)
    true_age_intervals = np.digitize(true_labels, age_intervals)
    # Calculate the confusion matrix
    cm = confusion_matrix(true_age_intervals, predicted_age_intervals)

    # Plot the confusion matrix with age interval labels
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    # Define the tick positions and labels for x-axis
    x_tick_positions = np.arange(len(age_intervals)) + 0.5
    x_tick_labels = [f'{age}-{age + interval}' for age in age_intervals]

    # Define the tick positions and labels for y-axis
    y_tick_positions = np.flip(np.arange(len(age_intervals))) + 0.5
    y_tick_labels = np.flip([f'{age}-{age + interval}' for age in age_intervals])

    # Set the tick positions and labels for x-axis and y-axis
    plt.xticks(x_tick_positions, x_tick_labels, rotation='vertical')
    plt.yticks(y_tick_positions, y_tick_labels)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted age')
    plt.ylabel('True age')
    plt.savefig(os.path.join(folder, 'confusionmatrix_test_set.png'))
    plt.clf()


###############################################################################################################
# ECGTransform

class ECGTransform(object):
    def __call__(self, signal):
        # Transform the data type from double (float64) to single (float32) to match the later network weights.
        t_signal = signal.astype(np.single)
        # We transpose the signal to later use the lead dim as the channel... (C,L).
        t_signal = torch.transpose(torch.tensor(t_signal), 0, 1)
        return t_signal  # Make sure I am a PyTorch Tensor


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
        return torch.transpose(processed_signals, 0, 1)

# Arguments that will be saved in config file
parser = argparse.ArgumentParser(add_help=True,
                                description='Train model to predict age from the raw ecg tracing.')
parser.add_argument('--seed', type=int, default=2,
                    help='random seed for number generator (default: 2)')
parser.add_argument('--seq_length', type=int, default=1000,
                    help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                'to fit into the given size. (default: 4096)')
parser.add_argument('--scale_multiplier', type=int, default=1, 
                    help='multiplicative factor used to rescale inputs.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size (default: 32).')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument("--patience", type=int, default=2,
                    help='maximum number of epochs without reducing the learning rate (default: 7)')
parser.add_argument("--min_lr", type=float, default=1e-7,
                    help='minimum learning rate (default: 1e-7)')
parser.add_argument("--lr_factor", type=float, default=0.1,
                    help='reducing factor for the lr in a plateu (default: 0.1)')
parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 450],
                    help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[1000, 500, 250, 125, 25],
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

N_LEADS = 12  # the 12 leads
N_Output = 1 #len(labels_train_set)  # just the age
model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                    blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                    n_classes=N_Output,
                    kernel_size=args.kernel_size,
                    dropout_rate=args.dropout_rate)
model.load_state_dict(torch.load('/home/stu15/MachineLearning_ageEstimation/Code/JAINIK/model_domain_adaptation/model_weights.pth'))
model.eval()
batch_size = 128

target_transform = ECGTransform()
test_dataset = ECGDataset_Target_Domain(data_dir="/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/g1", Transform=target_transform)

# Use the model for testing
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

predicted_labels = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        outputs, _ = model(inputs)  # Extract the tensor from the tuple
        predicted_labels.extend(outputs.tolist())
        true_labels.extend(labels.tolist())

# Convert the predicted_labels and true_labels to numpy arrays
predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)

folder = "/home/stu15/MachineLearning_ageEstimation/Code/JAINIK/model_domain_adaptation_trained/"
print("Reached here")
# Plot the confusion matrix
create_confusion_matrix(true_labels, predicted_labels, folder)

# Plot the scatter plot
create_scatter_plot(true_labels, predicted_labels, folder)