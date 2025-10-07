from LoRa_train.dataset_manager import OptimizedDatasetManager

dm = OptimizedDatasetManager('./datasets')
datasets = dm.scan_datasets(use_cache=False)
print('Available datasets:')
for i, ds in enumerate(datasets, 1):
    print(f'{i}. {ds["name"]} - {ds["file_path"]}')