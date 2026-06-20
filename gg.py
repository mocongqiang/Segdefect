import csv
import os

def clean_filename(filename: str) -> str:
    """删除文件名中的所有 '.jpg' 子串（不区分大小写）"""
    return filename.replace('.jpg', '')

def process_csv(input_path: str, output_path: str = None):
    """
    读取 CSV，清洗第一列中的 '.jpg'，写入新文件。
    若 output_path 为 None，则覆盖原文件（先写临时文件再替换）。
    """
    if output_path is None:
        output_path = input_path + '.tmp'

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            if row:                     # 跳过空行
                row[0] = clean_filename(row[0])
            writer.writerow(row)

    # 若为覆盖模式，则替换原文件
    if output_path == input_path + '.tmp':
        os.replace(output_path, input_path)
        print(f"已覆盖原文件: {input_path}")
    else:
        print(f"已生成新文件: {output_path}")

if __name__ == '__main__':
    # 请将 'data.csv' 替换为实际文件名
    process_csv('submission.csv')