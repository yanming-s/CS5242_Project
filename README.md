# Course Project of NUS CS5242

This is the group project of Group 29 NUS CS5242 (AY2024/2025).


## Model Testing

You can only run the testing script only for a specific checkpoint file, following the command below:

```bash
python main_vit.py --test_only --ckpt_path <absolute/path/to/your/checkpoint_file>
```


## Environment Setup

1. Python environment setup

   - Create a new conda environment with Python 3.12.9:

     ```bash
     conda create -n <env_name> python=3.12.9
     ```

   - Activate the environment:

     ```bash
     conda activate <env_name>
     ```

   - Install the required packages:

     ```bash
     pip install -r requirements.txt
     ```

2. Download the dataset from Kaggle

   ```bash
   python dataset/download.py
   ```

   The dataset will be downloaded to the `.cache/kagglehub` directory, with the specific path output in the terminal. You can manually move the dataset to the `./data` directory.

   If you have once set up the Kaggle API, you can download the dataset directly to the `./data` directory by running the following command:

   ```bash
   python dataset/download.py --use_kaggle_api
   ```

3. Convert the dataset into Tensor format for faster I/O (Optional)

   ```bash
   python dataset/convert_to_tensor.py
   ```

   This will automatically leverage the balanced sampling strategy to create a subset of the whole dataset. And the converted dataset will be saved in the `./data_tensor` directory. The original dataset will not be modified.

   For data augmentation with different splitting strategies, or binary splitting for generative models, you can run the following command:

   ```bash
   python dataset/convert_to_tensor.py --split_type <split_type>
   ```

   The `<split_type>` can be one of the following:
   - `balanced`: Balanced split based on the labels. (default)
   - `rare_first`: Sampling based on the rarest class first.
   - `original`: Sampling based on the original dataset distribution.
   - `binary`: Binary split for generative models.
