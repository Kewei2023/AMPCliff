import pandas as pd
import os
import ipdb
sequences_to_extract = ["AFHHIFRGIVHVGKTIHRLVTG","FFHHAFRGIVHVGKTIHRLVTG","AGRGKQGGKVRAKAKTRSS","DGRGKQGGKVRAKAKTRSS","KGRGKQGGKVRAKAKTRSS","GRGKQGGKVRAKAKTRSS","AGYLLGKINLKALAALAKKIL","AGYLLGKINLKPLAALAKKIL"]
new_names = [f"#{i}" for i in range(1,9)]

if __name__=='__main__':
  
  folder_path = "/data/home/scv6872/AMPCliff/outputs/random/"
  
  for files in os.listdir(folder_path):
    csv_file_path = os.path.join(folder_path, files)
    # Read the CSV file
    data = pd.read_csv(csv_file_path)
  
    # Extract rows with specified sequences
    extracted_data = data[data['Sequence'].isin(sequences_to_extract)].copy()
    # ipdb.set_trace()
    # Rename the sequences
    sequence_name_mapping = dict(zip(sequences_to_extract, new_names))
    extracted_data['NewName'] = extracted_data['Sequence'].map(sequence_name_mapping)
  
    # Display the extracted and renamed data
    extracted_data.to_csv(csv_file_path.replace('.csv','_extract.csv'))
    