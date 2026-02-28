import pandas as pd
import numpy as np
import datetime
import os

def generate_synthetic_icu_data(num_patients=1000, days_per_patient=3, measurements_per_day=4):
    """
    Generates a synthetic dataset of ICU vital signs and lab results, 
    including a target variable ('sepsis_onset') for training.
    """
    print(f"Generating data for {num_patients} patients...")
    data = []
    
    # Simulate a realistic patient mix
    # Approx 15% of patients develop sepsis
    np.random.seed(42)
    sepsis_patients = np.random.choice(range(num_patients), size=int(num_patients * 0.15), replace=False)
    
    for patient_id in range(num_patients):
        pid = f"P_{patient_id:04d}"
        
        # Static patient features
        base_age = np.random.normal(loc=65, scale=15) # average age 65
        base_age = np.clip(base_age, 18, 95)
        base_weight = np.random.normal(loc=80, scale=20) # average weight 80kg
        base_weight = np.clip(base_weight, 40, 150)
        
        is_sepsis_case = patient_id in sepsis_patients
        
        # When does sepsis happen? (Random measurement index during their stay)
        total_measurements = days_per_patient * measurements_per_day
        if is_sepsis_case:
            # Sepsis usually happens midway through stay
            sepsis_onset_idx = np.random.randint(total_measurements // 4, total_measurements - 2)
        else:
            sepsis_onset_idx = -1
            
        # Start time of ICU admission (sometime in 2023)
        start_time = datetime.datetime(2023, 1, 1) + datetime.timedelta(days=np.random.randint(0, 365))
        
        for i in range(total_measurements):
            timestamp = start_time + datetime.timedelta(hours=i * (24 / measurements_per_day))
            
            # Base healthy vitals
            hr = np.random.normal(loc=75, scale=10)
            sbp = np.random.normal(loc=120, scale=15)
            dbp = np.random.normal(loc=80, scale=10)
            map_val = (sbp + 2 * dbp) / 3
            o2sat = np.clip(np.random.normal(loc=98, scale=2), 50, 100)
            temp = np.random.normal(loc=37.0, scale=0.4)
            resp_rate = np.random.normal(loc=16, scale=3)
            
            # Base healthy labs
            lactate = np.random.lognormal(mean=0.0, sigma=0.5) # approx 1.0
            wbc = np.random.normal(loc=7.0, scale=2.0)
            creatinine = np.random.normal(loc=0.9, scale=0.2)
            bilirubin = np.random.normal(loc=0.6, scale=0.2)
            platelets = np.random.normal(loc=250, scale=50)
            
            sepsis_label = 0
            
            # Introduce Sepsis pathology if applicable
            if is_sepsis_case:
                # 1. The trajectory leading *up* to sepsis (deterioration)
                if i >= (sepsis_onset_idx - 4) and i < sepsis_onset_idx:
                    hr += np.random.uniform(10, 25) # Tachycardia
                    sbp -= np.random.uniform(5, 15) # Mild hypotension
                    temp += np.random.uniform(0.5, 1.5) # Fever
                    wbc += np.random.uniform(2, 5) # Elevated WBC
                
                # 2. Sepsis onset and after (Shock)
                elif i >= sepsis_onset_idx:
                    hr += np.random.uniform(30, 50) # Severe tachycardia
                    sbp -= np.random.uniform(20, 40) # Hypotension (Shock)
                    lactate += np.random.uniform(2.0, 6.0) # High lactate (tissue hypoxia)
                    resp_rate += np.random.uniform(6, 12) # Tachypnea
                    temp += np.random.uniform(1.0, 2.5) # High Fever
                    sepsis_label = 1
            
            # 10% chance a sensor disconnects / lab isn't drawn (missing data)
            if np.random.rand() < 0.1: lactate = np.nan
            if np.random.rand() < 0.1: wbc = np.nan
            if np.random.rand() < 0.05: hr = np.nan
            if np.random.rand() < 0.05: sbp = np.nan

            row = {
                'patient_id': pid,
                'timestamp': timestamp,
                'age': round(base_age, 1),
                'weight': round(base_weight, 1),
                'hr': round(hr, 1) if not pd.isna(hr) else np.nan,
                'sbp': round(sbp, 1) if not pd.isna(sbp) else np.nan,
                'dbp': round(dbp, 1),
                'map': round(map_val, 1),
                'o2sat': round(o2sat, 1),
                'temp': round(temp, 2),
                'resp_rate': round(resp_rate, 1),
                'lactate': round(lactate, 2) if not pd.isna(lactate) else np.nan,
                'wbc': round(wbc, 1) if not pd.isna(wbc) else np.nan,
                'creatinine': round(creatinine, 2),
                'bilirubin': round(bilirubin, 2),
                'platelets': round(platelets, 0),
                'sepsis_label': sepsis_label
            }
            data.append(row)
            
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = "synthetic_sepsis_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated {len(df)} records for {num_patients} patients.")
    print(f"Data saved to: {os.path.abspath(output_path)}")
    
    print("\nDataset Summary:")
    print(f"Total Sepsis Episodes (label=1): {df['sepsis_label'].sum()} ({df['sepsis_label'].mean():.1%} of records)")
    print(df[['hr', 'sbp', 'lactate', 'age']].describe().round(1))
    
    return df

if __name__ == "__main__":
    # Generate a lightweight dataset of 1,000 patients
    generate_synthetic_icu_data(num_patients=1000)
