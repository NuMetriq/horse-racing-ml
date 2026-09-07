import kagglehub

dataset = "deltaromeo/horse-racing-results-ukireland-2015-2025"

download_path = kagglehub.dataset_download(dataset)

print(f"Dataset downloaded to: {download_path}")