import argparse
from datetime import datetime
import os
import sys

def get_age_in_weeks_days_and_total_days(birth_date, file_date):
    age = file_date - birth_date
    weeks = age.days // 7
    days = age.days % 7
    total_days = age.days
    return weeks, days, total_days

def get_family_birth_date(pathname):
    family_birth_dates = {
        'b1_906F_1302M_3153M': datetime(2024, 3, 2),
        'b2_1305F_759M_3162F': datetime(2024, 3, 9),
        'b3_762F_763M_3121F': datetime(2024, 2, 2),
        'b4_1372F_1169M_3117F': datetime(2024, 1, 30),
        'familybooth_1594F_1449M_3010': datetime(2023, 7, 22)
    }
    
    for family, birth_date in family_birth_dates.items():
        if family in pathname:
            return birth_date
    
    return None

def process_files(default_birth_date):
    current_year = datetime.now().year
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        # Extract the filename and pathname
        pathname = os.path.dirname(line)
        filename = os.path.basename(line)
        
        # Extract the date from the filename
        file_date_str = filename[:6]
        file_date = datetime.strptime(file_date_str, "%y%m%d")
        
        # Adjust the year if necessary
        if file_date.year > current_year:
            file_date = file_date.replace(year=file_date.year - 100)
        
        # Get the appropriate birth date
        birth_date = get_family_birth_date(pathname) or default_birth_date
        
        weeks, days, total_days = get_age_in_weeks_days_and_total_days(birth_date, file_date)
        print(f"#W{weeks}# #P{total_days}# #W{weeks}D{days}# {line}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files and calculate age in weeks, days, and total days.")
    parser.add_argument("--birth_date", default="240202", help="Default birth date in YYMMDD format")
    args = parser.parse_args()
    
    default_birth_date = datetime.strptime(args.birth_date, "%y%m%d")
    process_files(default_birth_date)
