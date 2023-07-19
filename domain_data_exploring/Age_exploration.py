import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class ECGDataset_Domain(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.records = self._load_records()

    def _load_records(self):
        records = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.hea'):
                    record = os.path.splitext(file)[0]
                    records.append(os.path.join(root, record))
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        hea_path = record + '.hea'

        age = self._extract_age_from_hea(hea_path)

        return age

    def _extract_age_from_hea(self, hea_path):
        age = None
        with open(hea_path, 'r') as f:
            for line in f:
                if line.startswith('# Age:'):
                    age_string = line.split(':')[1].strip()
                    if age_string != "NaN":
                        age = int(line.split(':')[1].strip())
                    else:
                        age = None
                    break
        return age

# Create an instance of the ECGDataset_Domain class
data_dir = '/home/stu15/MachineLearning_ageEstimation/Code/physionet.org/files/challenge-2020/1.0.2/training/ptb/'
dataset = ECGDataset_Domain(data_dir)

# Extract ages from the dataset
ages = []
for index in range(len(dataset)):
    age = dataset[index]
    if age is not None:
        ages.append(age)

# Plot the histogram of ages
plt.hist(ages, bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.title('Histogram of Age in ptb Dataset')
output_dir = '/home/stu15/MachineLearning_ageEstimation/Code/JAINIK/domain_data_exploring/'
output_file = os.path.join(output_dir, 'ptb_age_histogram.png')
plt.savefig(output_file)
print("Histogram saved as", output_file)