import numpy as np

with open('output.txt', 'r') as file:
    lines = file.readlines()

# Separate lists for train and test
train_pehe = []
train_eate = []
test_pehe = []
test_eate = []

# Process the lines to extract values for train and test
for i in range(0, len(lines), 4):
    split_type = lines[i + 1].split(": ")[1].strip()
    pehe_value = float(lines[i + 2].split(": ")[1])
    eate_value = float(lines[i + 3].split(": ")[1])

    if split_type == 'train':
        train_pehe.append(np.sqrt(pehe_value))  # Square root of pehe for train
        train_eate.append(eate_value)
    elif split_type == 'test':
        test_pehe.append(np.sqrt(pehe_value))  # Square root of pehe for test
        test_eate.append(eate_value)
    else:
        raise Exception(f"unknown split: {split_type}.")

# Compute mean and std for train
train_pehe_mean = np.mean(train_pehe)
train_pehe_std = np.std(train_pehe)
train_eate_mean = np.mean(train_eate)
train_eate_std = np.std(train_eate)

# Compute mean and std for test
test_pehe_mean = np.mean(test_pehe)
test_pehe_std = np.std(test_pehe)
test_eate_mean = np.mean(test_eate)
test_eate_std = np.std(test_eate)

with open("result.txt", "w") as file:
    file.write(f"train: square root pehe: {train_pehe_mean:.2f} +- {train_pehe_std:.2f}\n"
               f"train: eate:{train_eate_mean:.2f} +- {train_eate_std:.2f}\n"
               f"test: square root pehe: {test_pehe_mean:.2f} +- {test_pehe_std:.2f}\n"
               f"test: eate:{test_eate_mean:.2f} +- {test_eate_std:.2f}\n")

