import requests
from simple_image_download import simple_image_download
import time

# Instantiate the downloader object
downloader = simple_image_download.Downloader

# Number of retry attempts
retry_attempts = 5

for attempt in range(retry_attempts):
    try:
        # Attempt to download images
        downloader().download('thanosMarvel', 60)
        print("Download completed successfully.")
        break  # Exit the loop if download is successful
    except requests.exceptions.ConnectionError as e:
        print(f"Attempt {attempt + 1} failed with error: {e}")
        time.sleep(5)  # Wait for 5 seconds before retrying
else:
    print("All retry attempts failed.")