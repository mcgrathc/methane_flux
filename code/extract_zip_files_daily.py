import os
import shutil
import zipfile


def extract_and_copy_fluxnet_data(path):
    """
  Extracts fluxnet data from zipped files and copies them to a designated directory.

  Parameters:
  - path (str): The root directory where the zipped data and the extracted files will be located.

  Note:
  The function assumes specific directory structures based on the given root directory.
  """

    # Set the directory containing the zip files
    zip_dir = f'{path}fluxnet_community_raw_data\\'

    # Set the directory for saving the extracted zipped files
    extract_dir = f'{path}flux_daily_raw_zipped\\'

    # Create directory for extracted files if it doesn't exist
    if not os.path.exists(extract_dir):
        os.mkdir(extract_dir)

    # Extract files that contain 'DD' in the filename from the zip archives
    for filename in os.listdir(zip_dir):
        if filename.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(zip_dir, filename)) as zip_ref:
                for zip_info in zip_ref.infolist():
                    if 'DD' in zip_info.filename:
                        zip_ref.extract(zip_info, extract_dir)

    # Set the directory for saving the extracted CSV files
    csv_dir = f'{path}flux_daily_raw\\'

    # Create directory for extracted CSV files if it doesn't exist
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    # Copy extracted CSV files to the specified directory
    for folder_name in os.listdir(extract_dir):
        if os.path.isdir(os.path.join(extract_dir, folder_name)):
            for filename in os.listdir(os.path.join(extract_dir, folder_name)):
                if filename.endswith('.csv'):
                    shutil.copy2(os.path.join(extract_dir, folder_name, filename),
                                 os.path.join(csv_dir, filename))