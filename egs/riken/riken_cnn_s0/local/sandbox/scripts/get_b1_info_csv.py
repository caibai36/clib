import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

class Segment:
    """A segment structure in audacity format
    
    Parameters
    ----------
    begin_sec : float
        The begin time of a segment.
    end_sec : float
        The end time of a segment.
    label : str
        The label of the segment.
    low_freq : float, optional
        The low frequency of a segment.
    high_freq : float, optional
        The high frequency of a segment.

    Examples
    --------
    0.108084        0.153355        tr
    \       5468.574219     10856.178711
    0.675958        1.387807        ph
    \       7743.052246     10856.178711
    1.722795        2.404735        ph
    \       7372.833496     9630.715820
    """
    def __init__(self, begin_sec, end_sec, label, low_freq=None, high_freq=None):
        self.begin_sec = begin_sec
        self.end_sec = end_sec
        self.label = label
        self.low_freq = low_freq
        self.high_freq = high_freq

    def __repr__(self):
        if self.low_freq and self.high_freq:
            return f"{self.begin_sec}\t{self.end_sec}\t{self.label}\n\\\t{self.low_freq}\t{self.high_freq}"       
        return f"{self.begin_sec}\t{self.end_sec}\t{self.label}"

def read_audacity_segments(segment_file):
    """Read the audacity segment file.
    
    Parameters
    ----------
    segment_file : str
        The path to the segment file of audacity.

    Returns
    -------
    list of Segment
        A list of Segment objects representing the segments in the file.

    Examples
    --------
    # Audacity format of each line: (begin_sec end_sec label), or optionally (\ lowest_freq high_freq)
    $ cat test_audacity_segments.txt
        0.108084        0.153355        tr
        \       5468.574219     10856.178711
        0.675958        1.387807        ph
        \       7743.052246     10856.178711
        1.722795        2.404735        ph
        \       7372.833496     9630.715820
    """
    segments = []
    with open(segment_file, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            elem = re.split("\s+", line)
            
            if len(elem) != 3:
                print(f"Warning: Invalid format of a line in the segment file: {segment_file}\n" 
                      f"The line: '{line}' is not in the format of 'begin_sec end_sec label' or '\\ min_freq high_freq'")
            else:
                first, second, third = elem
                
                if first != "\\":
                    s = Segment(begin_sec=float(first), end_sec=float(second), label=str(third))
                    segments.append(s)
                else:
                    segments[-1].low_freq = float(second)
                    segments[-1].high_freq = float(third)

    return segments

def get_age_in_weeks_days_and_total_days(birth_date, file_date):
    """Calculate age in weeks and days between two dates"""
    print(file_date)  # Preserved debug print
    age = file_date - birth_date
    weeks = age.days // 7
    days = age.days % 7
    total_days = age.days
    return total_days, weeks, days

def get_family_birth_date(pathname):
    """Maps family identifiers in pathname to their birth dates"""
    family_birth_dates = {
        'b1_906F_1302M_3153M': datetime(2024, 3, 2),
        'b2_1305F_759M_3162F': datetime(2024, 3, 9),
        'b3_762F_763M_3121F': datetime(2024, 2, 2),
        'b4_1372F_1169M_3117F': datetime(2024, 1, 30),
        'familybooth_1594F_1449M_3010': datetime(2023, 7, 22),
        'jay_family': datetime(2023, 7, 22),
        'jay_individual': datetime(2023, 7, 22),
        'wara': datetime(2023, 6, 4),
        'akiko_kii': datetime(2023, 8, 1),
        'akiko_renga': datetime(2023, 7, 24),
    }
    
    for family, birth_date in family_birth_dates.items():
        if family in pathname:
            return birth_date
    return None

def get_age(segments_file, default_birth_date=None):
    """Extract age information from segment filename and family data.
    
    Parameters
    ----------
    segments_file : str
        Segment label file whose name starting with date such as 
        240310_008_ch1_cnn_model_b0family3010_best_dev
        or 20240310_008_ch1_cnn_model_b0family3010_best_dev
    default_birth_date : datetime, optional
        Default birth date if family not found
    """
    current_year = datetime.now().year
    pathname = os.path.dirname(segments_file)
    filename = os.path.basename(segments_file)
    
    print(filename)  # Preserved debug print
    
    # Extract date from filename - supports both YYYYMMDD and YYMMDD formats
    if re.match(r'^20\d{6}', filename):
        file_date_str = filename[:8]
        file_date = datetime.strptime(file_date_str, "%Y%m%d")
    elif re.match(r'^\d{6}', filename):
        file_date_str = filename[:6]
        file_date = datetime.strptime(file_date_str, "%y%m%d")
    else:
        warnings.warn(f"Filename '{filename}' does not start with a valid date format (YYMMDD or YYYYMMDD).")
        return None, None, None
    
    print(file_date_str)  # Preserved debug print
    
    # Adjust the year if necessary for two-digit dates
    if file_date.year > current_year:
        file_date = file_date.replace(year=file_date.year - 100)
    
    birth_date = get_family_birth_date(pathname) or default_birth_date
    if not birth_date:
        return None, None, None
    
    return get_age_in_weeks_days_and_total_days(birth_date, file_date)

# Main data processing
root = "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sel"

# Configure dataset paths with preserved original paths in comments
all_seg_paths = {}
# Original paths preserved for reference:
# root="/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sandbox/202407_cnn/cnn_model_b0family3010_best_dev"
# root="/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sandbox/202407_cnn/cnn_model_b0family3010_best_dev_high_res_0.01sec"
# dataid ='b2_1305F_759M_3162F_cnn_model_b0family3010_best_dev_most_vocalized'
# all_seg_paths[dataid] = os.path.join(root, "20240906/seg/b2_1305F_759M_3162F")
# dataid ='b2_1305F_759M_3162F_cnn_model_b0family3010_best_dev_most_vocalized_every_3rd_days'
# all_seg_paths[dataid] = os.path.join(root, "20240906_b2_every_3rd_days/seg/b2_1305F_759M_3162F")

dataid = 'b1_906F_1302M_3153M_cnn_model_b0family3010_best_dev'
all_seg_paths[dataid] = os.path.join(root, "20240906/seg/b1_906F_1302M_3153M")

# Get label (segment) files for each dataset
all_seg_files = {}
for dataid in all_seg_paths.keys():
    all_seg_files[dataid] = []
    for dirpath, dirnames, filenames in os.walk(all_seg_paths[dataid]):
        for filename in filenames:
            all_seg_files[dataid].append(os.path.join(dirpath, filename))

# Create tables for all data
table = []
table_header = ['dataid', 'audioid', 'age_days', 'age_weeks', 'begin_sec', 'end_sec', 'label', 'low_freq', 'high_freq']

for dataid in all_seg_paths.keys():
    seg_files = all_seg_files[dataid]

    for path in sorted(seg_files):
        file_base = os.path.basename(path)
        audioid = file_base[:file_base.find("ch1") + len("ch1")] if "ch1" in file_base else None

        segments = read_audacity_segments(path)
        total_days, weeks, days = get_age(path)
        print(len(segments))  # Preserved debug print
        
        for seg in segments:
            table.append([dataid, audioid, total_days, weeks, seg.begin_sec, seg.end_sec, seg.label, seg.low_freq, seg.high_freq])

# Create and filter DataFrame
df_all = pd.DataFrame(data=table, columns=table_header)
df_b1 = df_all[df_all.age_days >= 0]  # Filter for b1 family with valid ages

print(len(df_b1.age_days.unique()))  # Print number of unique age days

df_b1.to_csv("/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sandbox/data/csv/sel_b1.csv")
