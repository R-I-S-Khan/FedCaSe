import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main(directory):
    # Read all CSV files in the directory that match the pattern
    all_data = []

    for filename in os.listdir(directory):
        if filename.endswith('_samples.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            id_ = filename.split('_')[0]
            df['id'] = id_
            all_data.append(df)

    # Prepare data for the new CSV file
    combined_data = pd.concat(all_data)

    combined_data['hitratio'] = combined_data['cache'] / combined_data['total']
    combined_data['missratio'] = combined_data['ssd'] / combined_data['total']

    # Write data to a new CSV file
    combined_data.to_csv(os.path.join(directory, 'allclientiotrace.csv'), index=False)

    # Read the new CSV file and prepare CDF of hit ratio for the first 500 client IDs
    # new_df = pd.read_csv(os.path.join(directory, 'allclientiotrace.csv'))

    # Select data for the first 500 client IDs
    # first_500_ids = new_df['id'].unique()[:500]
    # selected_data = new_df[new_df['id'].isin(first_500_ids)]

    # Calculate CDF of hit ratio
    # hit_ratio_cdf = selected_data['hitratio'].value_counts(normalize=True).sort_index().cumsum()

    # Plot CDF
    # plt.plot(hit_ratio_cdf.index, hit_ratio_cdf.values)
    # plt.xlabel('Hit Ratio')
    # plt.ylabel('CDF')
    # plt.title('CDF of Hit Ratio for First 500 Client IDs')
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files in a directory.')
    parser.add_argument('--directory', '-d', default='.', help='Directory containing CSV files')
    args = parser.parse_args()
    main(args.directory)