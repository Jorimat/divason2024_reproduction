# Download and unzip models
wget https://zenodo.org/records/10231845/files/models.zip
unzip models.zip
rm models.zip

# Download and unzip data
wget https://zenodo.org/records/10231845/files/df_dataset.csv
wget https://zenodo.org/records/10231845/files/dataset.zip
unzip dataset.zip
rm dataset.zip

# Organize data
mkdir data
mv dataset data
mv df_dataset.csv data
