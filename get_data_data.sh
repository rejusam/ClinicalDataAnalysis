# Create directory structure
mkdir -p data/mimic3wdb/p00

# Download just 10 specific patient records
cd data/mimic3wdb/p00

# Example patients
for patient in p000020 p000030 p000033 p000052 p000079 p000085 p000123 p000154 p000208 p000262
do
    wget -r -np -nH --cut-dirs=4 -R "index.html*" \
    "https://physionet.org/files/mimic3wdb-matched/1.0/p00/${patient}/"
done

cd ../../..
