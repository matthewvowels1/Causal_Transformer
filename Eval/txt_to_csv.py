import csv
import os


def txt_to_csv(directory_path='result.txt', output_path='table.csv'):
    files = os.listdir(directory_path)
    print(files)
    datas = []
    for file_path in files:
        with open(os.path.join(directory_path,file_path), 'r') as file:
            lines = file.readlines()

        data = [file_path]
        for line in lines:
            *infos, result = line.split(": ")
            result = result.split("n=")[0].strip()
            result.replace('+-', '±')
            data.append(result)
        datas.append(data)

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Writing each row to the CSV file
        for row in datas:
            writer.writerow(row)

if __name__ == "__main__":
    txt_to_csv("JOBS/results/")