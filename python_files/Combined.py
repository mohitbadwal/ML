import os

import sys

if len(sys.argv) == 3:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
else:
    print("Format python filename train_dataset_path test_dataset_path")
    sys.exit(1)

os.system('python Page_classification_final_model.py ' + train_file + ' ' + test_file)
os.system('python ocr_line.py ' + train_file + ' ' + test_file)
os.system('python ocr_heading_harshit.py ' + train_file + ' ' + test_file)
os.system('python final_is_remittance_model_4_files.py ' + train_file + ' ' + test_file)

print("Done , the predicitons were made for " + test_file)
