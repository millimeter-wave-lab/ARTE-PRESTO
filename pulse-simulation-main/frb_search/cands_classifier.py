import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from your.candidate import Candidate

# script to adapt candidates csv generated by the pipeline to the classifier
# and to classify the candidates

def load_candidates(cands_csv):
    """
    Load candidates from csv file
    """
    cands = pd.read_csv(cands_csv)
    return cands


def build_new_csv(cands_csv, file):
    """
    Build a new csv file with the columns needed for the classifier
    """
    cands = load_candidates(cands_csv)
    new_csv_columns = ['file','snr','stime','width','dm','label','chan_mask_path','num_files']
    new_csv = pd.DataFrame(columns=new_csv_columns)
    new_csv['file'] = file
    new_csv['snr'] = cands['Sigma']
    new_csv['stime'] = cands['Time(s)']
    new_csv['width'] = 2  # width is fixed to 2 samples, this needs to be changed
    new_csv['dm'] = cands['DM']
    new_csv['label'] = 0
    new_csv['chan_mask_path'] = ''
    new_csv['num_files'] = 1
    return new_csv


def create_candidate_objects(cands_csv, file, downsample=1):
    """
    Create candidate objects from the new csv
    """
    for i in cands_csv:
        cand = Candidate(fp=file, snr=i['Sigma'], tcand=i['Time(s)'], width=2, dm=i['DM'], label=0)
        cand.dedisperse()
        if downsample > 1:
            cand.decimate(key="ft", axis=0, pad=True, decimate_factor=downsample, mode="median")
        fout = cand.save_h5()
        print(fout)
        

