'''
Data importer for Thermo Fisher DXR2 Raman spectrometer.
Provide functions to synthesize multiple scan files into one csv file suitable for data analysis.
'''

import os
import csv
import pandas as pd
def merge_files(folder_path, output_path, header = None):
    '''
    Merge multiple scan files into one dataset (save as csv). 
    If there are leading metadata lines. Skip them by passing header = N.

    Example
    -------
        merge_files("C:/Users/xx/白及 四川", 
        'C:/Users/xx/白及四川-merged_data.CSV')
    '''

    # 获取目标文件夹下的所有csv文件路径
    csv_files = [file for file in os.listdir(folder_path) if file.upper().endswith('.CSV') or file.upper().endswith('.TXT')]
    # print(csv_files)
    # 创建一个空的DataFrame用于存储整合后的数据
    merged_data = pd.DataFrame()

    # 逐个读取csv文件并整合到merged_data中
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        print(file_path)
        data = pd.read_csv(file_path, header=header)
        merged_data = pd.concat([data,merged_data,], axis=1, ignore_index=True)
    #print(merged_data)

    # 转置处理
    merged_data = merged_data.transpose()

    # 填充缺失值为0
    merged_data = merged_data.fillna(0)

    # 在第一行之前插入一行，值由原来的第一行的值赋
    merged_data.loc[-1] = merged_data.iloc[0]
    merged_data.index = merged_data.index + 1
    merged_data = merged_data.sort_index()
   
    # 删除偶数行，保留奇数行
    merged_data = merged_data.iloc[::2, :]
    
    # 在最左侧增加一列，第一个值为'label'，其余为0
    #若有其他标签，0进一步修改为1、2....
    merged_data.insert(0, 'label', ['label'] + [0] * (merged_data.shape[0] - 1))
    

    # 保存整合好的数据到输出文件
    merged_data.to_csv(output_path, index=False,header=None)


def merge_multiple_datasets(file_list, output_file):
    '''
    Further merge multiple datasets into one.

    Example
    -------
        file_list = [  'C:/Users/xx/白及四川-merged_data.CSV',
            'C:/Users/xx/白及云南-merged_data.CSV'
        ]
        output_file = 'C:/Users/xx/白及merged_data.CSV'
        merge_multiple_datasets(file_list, output_file)
    '''


    # 创建一个写入 CSV 文件的文件对象
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 读取第一个文件的所有数据
        with open(file_list[0], 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                writer.writerow(row)

        # 读取其他文件的除去第一行的数据
        for file_name in file_list[1:]:
            with open(file_name, 'r', newline='') as file:
                reader = csv.reader(file)
                next(reader)  # 跳过第一行标题
                for row in reader:
                    writer.writerow(row)