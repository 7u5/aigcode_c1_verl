import pandas as pd
df = pd.read_parquet("/sharedata/hf/BytedTsinghua-SIA_AIME-2024/data/aime-2024.parquet")
print("Columns:", df.columns)
print("Sample rows:", df.head())
print("Missing prompt:", df[df['prompt'].isna() | (df['prompt'].apply(lambda x: not x or not x[0].get('content', '').strip()))])
print("Missing ground_truth:", df[df['reward_model'].apply(lambda x: not isinstance(x, dict) or not x.get('ground_truth', '').strip())])
