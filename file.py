# step0_merge_data.py
import pandas as pd
import numpy as np
import os

# --------- Config ----------
HEART_CSV = "heartrate_seconds_merged.csv"
STEPS_CSV = "hourlySteps_merged.csv"
CAL_CSV = "hourlyCalories_merged.csv"

OUTPUT_CSV = "fitbit.csv"

# If True, only keep rows where heart_rate, steps and calories are all present.
# Set to False if you prefer to keep partial records.
DROP_PARTIAL_ROWS = True
# ---------------------------

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {path} -> {df.shape[0]} rows, columns: {list(df.columns)[:8]}{'...' if len(df.columns)>8 else ''}")
    return df

def detect_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    # Load files
    heart = load_csv(HEART_CSV)
    steps = load_csv(STEPS_CSV)
    cal = load_csv(CAL_CSV)

    # ---------- Heart rate ----------
    # Expecting columns like ['Id','Time','Value'] where Value is bpm
    id_col_h = detect_col(heart, ['Id', 'id', 'user_id', 'UserID'])
    time_col_h = detect_col(heart, ['Time', 'time', 'datetime', 'DateTime'])
    val_col_h = detect_col(heart, ['Value', 'value', 'HeartRate', 'heart_rate', 'BPM'])

    if not (id_col_h and time_col_h and val_col_h):
        raise ValueError("Could not detect required columns in heart CSV. Found: "
                         f"{heart.columns.tolist()}")

    heart = heart[[id_col_h, time_col_h, val_col_h]].rename(columns={
        id_col_h: 'Id', time_col_h: 'Time', val_col_h: 'heart_rate'
    })

    # parse datetime and floor to hour for alignment
    heart['Time'] = pd.to_datetime(heart['Time'], errors='coerce')
    before_drop = heart.shape[0]
    heart = heart.dropna(subset=['Time'])
    if heart.shape[0] < before_drop:
        print(f"Dropped {before_drop - heart.shape[0]} heart rows with invalid Time")
    # Average heart rate per hour per Id
    heart['Hour'] = heart['Time'].dt.floor('H')
    heart_hourly = heart.groupby(['Id', 'Hour'], as_index=False)['heart_rate'].mean()
    heart_hourly.rename(columns={'Hour': 'Time'}, inplace=True)
    print(f"Heart aggregated hourly -> {heart_hourly.shape[0]} rows")

    # ---------- Steps ----------
    # Expecting ['Id','ActivityHour','StepTotal'] or similar
    id_col_s = detect_col(steps, ['Id', 'id', 'user_id'])
    time_col_s = detect_col(steps, ['ActivityHour', 'Activity Hour', 'activity_hour', 'ActivityHour.1', 'ActivityHour'])
    # possible step column names
    steps_col_candidates = ['StepTotal', 'Step Total', 'Steps', 'steps', 'StepTotal_','TotalSteps']
    steps_val_col = detect_col(steps, steps_col_candidates)

    if not (id_col_s and time_col_s and steps_val_col):
        raise ValueError("Could not detect required columns in steps CSV. Found: "
                         f"{steps.columns.tolist()}")

    steps = steps[[id_col_s, time_col_s, steps_val_col]].rename(columns={
        id_col_s: 'Id', time_col_s: 'Time', steps_val_col: 'steps'
    })
    steps['Time'] = pd.to_datetime(steps['Time'], errors='coerce')
    steps = steps.dropna(subset=['Time'])
    # floor to hour (should already be hourly, but ensures same key)
    steps['Time'] = steps['Time'].dt.floor('H')
    # If steps are cumulative or not, we keep the provided value.
    print(f"Steps prepared -> {steps.shape[0]} rows")

    # ---------- Calories ----------
    id_col_c = detect_col(cal, ['Id', 'id', 'user_id'])
    time_col_c = detect_col(cal, ['ActivityHour', 'Activity Hour', 'activity_hour', 'ActivityHour'])
    cal_col_candidates = ['Calories', 'calories', 'Calorie', 'CaloriesBurned']
    cal_val_col = detect_col(cal, cal_col_candidates)

    if not (id_col_c and time_col_c and cal_val_col):
        raise ValueError("Could not detect required columns in calories CSV. Found: "
                         f"{cal.columns.tolist()}")

    cal = cal[[id_col_c, time_col_c, cal_val_col]].rename(columns={
        id_col_c: 'Id', time_col_c: 'Time', cal_val_col: 'calories'
    })
    cal['Time'] = pd.to_datetime(cal['Time'], errors='coerce')
    cal = cal.dropna(subset=['Time'])
    cal['Time'] = cal['Time'].dt.floor('H')
    print(f"Calories prepared -> {cal.shape[0]} rows")

    # ---------- Merge ----------
    # Merge on Id + Time (hour buckets). Use outer merge to preserve records, then optionally drop partials.
    merged = pd.merge(heart_hourly, steps, on=['Id', 'Time'], how='outer', suffixes=('_heart','_steps'))
    merged = pd.merge(merged, cal, on=['Id', 'Time'], how='outer')

    print(f"Merged shape (outer): {merged.shape}")
    # Sort and reset index
    merged = merged.sort_values(['Id','Time']).reset_index(drop=True)

    # Optionally drop rows missing any of the key signals
    if DROP_PARTIAL_ROWS:
        before = merged.shape[0]
        merged = merged.dropna(subset=['heart_rate', 'steps', 'calories'])
        after = merged.shape[0]
        print(f"Dropped {before - after} rows missing heart_rate/steps/calories (kept only full rows).")

    # Fill remaining tiny gaps if any (not recommended to blindly fill; here we keep them NaN)
    # merged['heart_rate'] = merged['heart_rate'].fillna(method='ffill')  # example if needed

    # Save cleaned file
    # Rename Time to a consistent column name if needed
    merged = merged.rename(columns={'Time': 'Time', 'Id': 'Id'})
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved merged dataset to {OUTPUT_CSV} with shape: {merged.shape}")
    print(merged.head(10))

if __name__ == "__main__":
    main()
