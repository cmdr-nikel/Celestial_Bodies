import pandas as pd
import json, os

df = pd.read_csv('../Data/star_classification.csv')

# Samples 500 of each class from dataset
parts = []
for cls in ['GALAXY', 'QSO', 'STAR']:
    parts.append(df[df['class'] == cls].sample(500, random_state=42))
sample = pd.concat(parts).reset_index(drop=True)

# Normalise u/g/r/i/z -- 0–1
for band in ['u', 'g', 'r', 'i', 'z']:
    mn = df[band].quantile(0.01)
    mx = df[band].quantile(0.99)
    sample[f'{band}_norm'] = ((sample[band] - mn) / (mx - mn)).clip(0, 1)

# Color index g-r
sample['color_gr'] = (sample['g'] - sample['r']).clip(-1, 3)

# Gathering objects into dict
objects = []
for i, row in sample.iterrows():
    objects.append({
        "id":       f"sdss_{i:04d}",
        "ra":       round(float(row['alpha']),   4),
        "dec":      round(float(row['delta']),   4),
        "u":        round(float(row['u']),        3),
        "g":        round(float(row['g']),        3),
        "r":        round(float(row['r']),        3),
        "i":        round(float(row['i']),        3),
        "z":        round(float(row['z']),        3),
        "u_norm":   round(float(row['u_norm']),   3),
        "g_norm":   round(float(row['g_norm']),   3),
        "r_norm":   round(float(row['r_norm']),   3),
        "i_norm":   round(float(row['i_norm']),   3),
        "z_norm":   round(float(row['z_norm']),   3),
        "redshift": round(float(row['redshift']), 5),
        "class":    str(row['class']),
        "color_gr": round(float(row['color_gr']), 3)
    })

# Creating folders
out_dir = '../FrontEnd'
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, 'objects.json')
with open(out_path, 'w') as f:
    json.dump(objects, f, separators=(',', ':'))

print(f"Successfully exported {len(objects)} objects - {out_path}")
print(sample['class'].value_counts())