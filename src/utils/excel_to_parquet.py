import os
import pandas as pd
import re
import datetime

# --- CONFIGURATION ---
# Using relative paths
RAW_BASE_DIR = "data/data_raw/Battery Degradation Dataset"
README_PATH = os.path.join(RAW_BASE_DIR, "Readme.txt")
OUTPUT_DIR = "data/data_processed/standardized_cells"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_readme_metadata(readme_path):
    """
    Reads the Readme.txt to create a map: Cell_ID -> (Charge, Discharge)
    """
    metadata = {}
    print(f"Reading metadata from {readme_path}...")
    
    if not os.path.exists(readme_path):
        print(f"Error: Readme not found at {readme_path}")
        return {}

    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line.startswith("#"): continue
        
        # Parse line like: "#1  Random  3C"
        parts = re.split(r'\s+', line) 
        if len(parts) >= 3:
            cell_id = parts[0].replace("#", "")
            charge = parts[1]
            discharge = parts[2]
            
            if "Random" in charge: charge = "Rd"
            metadata[cell_id] = (charge, discharge)
            
    return metadata

def convert_time_to_seconds(time_val):
    """
    Converts 'HH:MM:SS' strings to total seconds (float).
    Handles cases where it might already be a number or a datetime object.
    """
    if pd.isna(time_val):
        return 0.0
        
    # If it's already a (int/float), just return it
    if isinstance(time_val, (int, float)):
        return float(time_val)
        
    # If it is a datetime.time object (Excel sometimes does this automatically)
    if isinstance(time_val, datetime.time):
        return time_val.hour * 3600 + time_val.minute * 60 + time_val.second + time_val.microsecond / 1e6

    # If it's a string like "00:00:10"
    if isinstance(time_val, str):
        try:
            # Check for HH:MM:SS format
            if ":" in time_val:
                parts = time_val.split(":")
                if len(parts) == 3:
                    h, m, s = map(float, parts)
                    return h * 3600 + m * 60 + s
                elif len(parts) == 2: # MM:SS
                    m, s = map(float, parts)
                    return m * 60 + s
        except ValueError:
            pass # Fallback to standard conversion if this fails
            
    # Fallback: force numeric, turn errors to 0
    try:
        return float(time_val)
    except:
        return 0.0

def standardize_dataset():
    meta_map = parse_readme_metadata(README_PATH)
    if not meta_map:
        print("No metadata found. Exiting.")
        return

    print("Scanning directories...")
    
    for root, dirs, files in os.walk(RAW_BASE_DIR):
        current_folder = os.path.basename(root)
        
        if current_folder.startswith("#") and current_folder.replace("#", "").isdigit():
            cell_id = current_folder.replace("#", "")
            
            if cell_id not in meta_map:
                continue
                
            charge, discharge = meta_map[cell_id]
            
            # Find the main aging file
            aging_files = [f for f in files if "cycle" in f.lower() and "first20" not in f.lower() and f.endswith(".xlsx") and not f.startswith("~$")]
            
            if not aging_files:
                continue
            
            source_file = os.path.join(root, aging_files[0])
            new_name = f"{int(cell_id):02d}_{charge}_{discharge}.parquet"
            dest_path = os.path.join(OUTPUT_DIR, new_name)
            
            if os.path.exists(dest_path):
                print(f"Skipping {new_name} (Already exists)")
                continue
                
            print(f"Processing Cell #{cell_id} -> {new_name}...")
            try:
                # Read Excel
                df = pd.read_excel(source_file, engine='openpyxl')
                
                # 1. Clean Headers (lowercase, remove spaces and parentheses content for safety if needed)
                # We strip spaces and lower case. "Test_Time(s)" -> "test_time(s)"
                df.columns = [str(c).strip().lower() for c in df.columns]
                
                # 2. Fix the Time Column
                # Use a flexible search for the time column name
                time_col = next((c for c in df.columns if "time" in c and "date" not in c), None)
                
                if time_col:
                    # Apply the conversion logic row-by-row
                    df[time_col] = df[time_col].apply(convert_time_to_seconds)
                    # Rename to standard 'time_s' for ML script clarity
                    df.rename(columns={time_col: 'time_s'}, inplace=True)
                
                # 3. Ensure numeric types for other key columns
                for col in df.columns:
                    if col != 'date_time' and col != 'time_s':
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Save
                df.to_parquet(dest_path, index=False)
                
            except Exception as e:
                print(f"Failed to convert Cell #{cell_id}: {e}")

    print(f"\nDone - Processed files are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    standardize_dataset()
