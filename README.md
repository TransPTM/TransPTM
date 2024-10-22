### TransPTM
A Transformer-Based Model for Non-Histone Acetylation Site Prediction

*Briefings in Bioinformatics, Volume 25, Issue 3, May 2024, bbae219, https://doi.org/10.1093/bib/bbae219*

### Environment
python 3.8.0

pytorch 1.12.0

pytorch geometric 2.2.0

Cuda 11.6

### How to run
Run ./data_preprocess/generate_data.py to generate graph data.

Run ./model/main.py to train the TransPTM.

### Dataset
NHAC.csv (Non-Histone Acetylation Collection)

unique_id format: Uniprot ID; acetylation position; protein sequence full length; label


