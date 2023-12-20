# csv_file = pd.read_csv('../../ship_data/train_ship_segmentations_v2.csv')
# csv_file = csv_file.groupby('ImageId')['EncodedPixels'].apply(list).reset_index()
# image_ids, pixels = csv_file['ImageId'].values.tolist(), csv_file['EncodedPixels'].values.tolist()