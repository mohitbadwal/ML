import os

import sys

if len(sys.argv) == 3:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
else:
    print("Format python filename train_dataset_path test_dataset_path")
    sys.exit(1)
import os
os.chdir(r"D:\backup\PycharmProjects\test\Image Batches-20171017T131547Z-001\python_files")
#os.system('python Page_classification_final_model.py ' + train_file + ' ' + test_file)
os.system('python ocr_line.py ' + train_file + ' ' + test_file)
os.system('python is_heading_combined_model.py ' + train_file + ' ' + test_file)
os.system('python final_is_remittance_model_4_files.py ' + train_file + ' ' + test_file)

print("Done , the predicitons were made for " + test_file)
