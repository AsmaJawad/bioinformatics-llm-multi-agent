import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the VCF file
df = None
with open('variants.vcf', 'r') as file:
    if file.readline().startswith('>'):
        from Bio import SeqIO
        df = pd.DataFrame([record.description.split(' ')[0] for record in SeqIO.parse(file, 'fasta')])
    elif file.readline().startswith('#'):
        df = pd.read_csv(file, sep='\t', comment='#')
    else:
        df = pd.read_csv(file, sep=None, engine='python')

# Strip '#' from headers
df.columns = df.columns.str.strip('#')

# Check for columns (QUAL, DP, REF, ALT, etc.) before filtering
required_columns = ['QUAL', 'DP', 'REF', 'ALT']
if not all(column in df.columns for column in required_columns):
    print("Missing required columns in the VCF file", file=sys.stderr)
    exit(1)

# Filter the rows where the 'QUAL' column is greater than 30 and 'DP' column is greater than 10
filtered_df = df[(df['QUAL'] > 30) & (df['DP'] > 10)]

# Create a summary of variant types
variant_summary = filtered_df['ALT'].str.split(',').explode().value_counts().sort_values(ascending=False).head(10)

# Save the filtered data to a new VCF file
filtered_df.to_csv('filtered_variants.vcf', index=False)

# Save the summary of variant types to a new CSV file
variant_summary.to_csv('variant_summary.csv')

# Create a heatmap of the variant types
plt.figure(figsize=(10, 8))
sns.heatmap(variant_summary.unstack().sort_values(ascending=False).head(10), annot=True, fmt='d')
plt.savefig('heatmap.png')

# Deactivate and remove the virtual environment
deactivate
rm -rf venv
