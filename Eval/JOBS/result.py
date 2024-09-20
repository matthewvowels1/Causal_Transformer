import numpy as np

with open('output.txt', 'r') as file:
    lines = file.readlines()

datas = {}


def parse(split, sort, value):
    if sort == "pehe":
        sort = "square root pehe"
        value = np.sqrt(value)
    index = f"{split}: {sort}"
    if index not in datas:
        datas[index] = []
    datas[index].append(value)


for i in range(0, len(lines), 4):
    split_type = lines[i + 1].split(": ")[1].strip()
    for j in range(2, 4):
        splits = lines[i + j].split(": ")
        parse(split=split_type,
              sort=splits[0].strip(),
              value=float(splits[1]))

with open("result.txt", "w") as file:
    for index, values in datas.items():
        file.write(f"{index}: {np.mean(values):.2f} +- {np.std(values):.2f}\n")
