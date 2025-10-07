import pandas as pd
import re

def clean_translation_dataset(input_file, output_file):
    """Clean the translation dataset to have only catalan and chino columns"""
    
    print(f"Processing {input_file}...")
    
    # Read the original file
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Process each line
    clean_data = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Skip header
        if i == 0:
            continue
            
        # Split by ;;; separator
        parts = line.split(';;;')
        if len(parts) < 2:
            continue
            
        # Get the CSV data part (first part before ;;;)
        csv_data = parts[0]
        
        # Parse the CSV data
        if csv_data.startswith('"') and csv_data.endswith('"'):
            # Remove outer quotes
            csv_data = csv_data[1:-1]
            
            # Parse the CSV content
            # Format: "558,""text1"",""text2"""
            # We need to split by comma but handle quoted content
            
            # Use regex to find the pattern: number,""text1"",""text2""
            pattern = r'^(\d+),""([^"]*(?:""[^"]*)*)"",(.+)$'
            match = re.match(pattern, csv_data)
            
            if match:
                number = match.group(1)
                catalan_text = match.group(2).replace('""', '"')  # Convert escaped quotes
                chino_text = match.group(3)
                
                # Add to clean data
                clean_data.append([catalan_text, chino_text])
            else:
                # Try simple comma split as fallback
                parts_csv = csv_data.split(',')
                if len(parts_csv) >= 3:
                    catalan_text = parts_csv[1].strip('"')
                    chino_text = parts_csv[2].strip('"')
                    clean_data.append([catalan_text, chino_text])
        else:
            # Simple format: 31524,text1,text2
            parts_csv = csv_data.split(',')
            if len(parts_csv) >= 3:
                catalan_text = parts_csv[1]
                chino_text = parts_csv[2]
                clean_data.append([catalan_text, chino_text])
    
    # Create DataFrame
    df = pd.DataFrame(clean_data, columns=['catalan', 'chino'])
    
    # Remove empty rows
    df = df.dropna()
    df = df[df['catalan'].str.strip() != '']
    df = df[df['chino'].str.strip() != '']
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ Cleaned dataset saved to {output_file}")
    print(f"📊 Total rows: {len(df)}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🔍 Sample data:")
    print(df.head(3).to_string())
    
    return df

if __name__ == "__main__":
    # Clean the dataset
    df = clean_translation_dataset(
        'datasets/muestra_traducciones_10000.csv',
        'datasets/clean_translations.csv'
    )