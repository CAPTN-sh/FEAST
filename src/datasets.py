import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from torch.utils.data import Dataset

def generate_ood_data(df, ood_mean=10, ood_std=5):

    ood_data = df.copy()
    
    for col in df.columns:
        # 数值列处理
        if pd.api.types.is_numeric_dtype(df[col]):
            #mean, std = df[col].mean(), df[col].std()
            # 在分布范围外生成数据
            ood_data[col] = np.random.normal(ood_mean, ood_std, len(df))
        
    # 转为 DataFrame
    df_ood = pd.DataFrame(ood_data)
    return df_ood[:500]
    
def linear_interpolate(sequence, target_length):
    original_length = len(sequence)
    if original_length >= target_length:
        return sequence[:target_length]

    original_indices = np.linspace(0, original_length - 1, num=original_length)
    target_indices = np.linspace(0, original_length - 1, num=target_length)

    if sequence.ndim > 1:
        interpolated_sequence = np.zeros((target_length, sequence.shape[1]))
        for i in range(sequence.shape[1]):
            interpolated_sequence[:, i] = np.interp(target_indices, original_indices, sequence[:, i])
    else:
        interpolated_sequence = np.interp(target_indices, original_indices, sequence)

    return interpolated_sequence

def zero_padding(sequence, target_length):
    original_length = len(sequence)
    if original_length >= target_length:
        return sequence[:target_length]
    else:
        padding_length = target_length - original_length
        padding = np.zeros((padding_length,) + sequence.shape[1:])
        padded_sequence = np.concatenate([sequence, padding], axis=0)
        return padded_sequence

def allocate_real_values(data_length, seq_len, pred_len):
    total_len = seq_len + pred_len
    real_values_for_input = int((seq_len / total_len) * data_length)
    real_values_for_target = data_length - real_values_for_input

    if real_values_for_target == 0 and real_values_for_input > 0:
        real_values_for_input -= 1
        real_values_for_target = 1

    return real_values_for_input, real_values_for_target

def sliding_window_sequences(features, seq_len, pred_len=None, mode='ae', target_values=None, interpolate_method=None, padding_method=zero_padding, labels_df=None):
    input_sequences, targets_sequences, input_masks, target_masks, labels_sequences = [], [], [], [], []
    
    step = seq_len + (pred_len if mode == 'prediction' and pred_len else 0)

    for start in range(0, len(features), step):
        end = start + seq_len
        if mode == 'prediction' and pred_len:
            real_values_for_input, real_values_for_target = allocate_real_values(len(features[start:end]), seq_len, pred_len)
        else:
            real_values_for_input = len(features[start:end])
            real_values_for_target = 0

        input_seq = features[start:start+real_values_for_input]
        valid_length = len(input_seq)
        
        if labels_df is not None:
            labels_seq = labels_df.iloc[start]
            #print(input_seq[0], labels_seq)
        else:
            labels_seq = None
        valid_length = len(input_seq)
        if valid_length < seq_len and interpolate_method is not None:
            input_seq = interpolate_method(input_seq, seq_len)
        else:
            input_seq = padding_method(input_seq, seq_len)

        input_mask = np.hstack((np.ones((valid_length,)), np.zeros((seq_len - valid_length,))))
        input_sequences.append(input_seq)
        labels_sequences.append(labels_seq)
        input_masks.append(input_mask)

        if mode == 'prediction' and pred_len:
            target_start = start + real_values_for_input
            target_seq = target_values[target_start:target_start+real_values_for_target] if target_values is not None else np.array([])
            target_valid_length = len(target_seq)
            if target_valid_length < pred_len and interpolate_method is not None:
                target_seq = interpolate_method(target_seq, pred_len)
            else:
                target_seq = padding_method(target_seq, pred_len)

            target_mask = np.hstack((np.ones((target_valid_length,)), np.zeros((pred_len - target_valid_length,))))
            targets_sequences.append(target_seq)
            target_masks.append(target_mask)

    inputs = np.array(input_sequences, dtype=np.float32)
    targets = np.array(targets_sequences, dtype=np.float32) if mode == 'prediction' else None
    input_masks = np.array(input_masks, dtype=np.float32)
    target_masks = np.array(target_masks, dtype=np.float32) if mode == 'prediction' else None

    return (inputs, targets, input_masks, target_masks, labels_sequences) if mode == 'prediction' else (inputs, input_masks, labels_sequences)

class TrajectoryDataset(Dataset):
    def __init__(self, dataframe, seq_len, pred_len=None, mode='ae', drop_features_list=None, pred_features=None, interpolate_method=None, filter_percent=None, scaler_method='MinMaxScaler', scaler=None, filter_less_seq_len=None):
        """
        Initialize the trajectory dataset.

        Parameters:
        - dataframe: DataFrame containing trajectory data.
        - seq_len: Length of input sequence.
        - pred_len: (Optional) Length of target sequence in prediction mode.
        - mode: Dataset mode, 'ae' for autoencoder training, 'prediction' for prediction mode.
        - drop_features_list: (Optional) List of features to drop in DAE mode.
        - pred_features: (Optional) Target feature column names in prediction mode.
        - interpolate_method: (Optional) Method for sequence interpolation.
        - filter_percent: e.g. 0.995 means data with quantiles greater than 99.5% and less than 5% will be filtered to reduce extreme values
        - scaler: If it is a Train set, the scaler is None. If it is a Val set, the scaler of the Train set needs to be used. 
        - filter_less_seq_len: Seqs smaller than this value will be filtered, if None, no filtering
		- scaler_methid: The scaling method to use. Options are 'MinMaxScaler', 'StandardScaler', 'RobustScaler'.
        """
        self.dataframe = dataframe.copy()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.filter_less_seq_len = filter_less_seq_len 
        self.mode = mode
        self.interpolate_method = interpolate_method
        self.scaler_name = scaler_method
        # Define feature columns
        if drop_features_list is not None:
            self.feature_columns = dataframe.columns.drop(drop_features_list)
        else:
            self.feature_columns = dataframe.columns
        self.n_features = len(self.feature_columns)
        
        if 'season' in self.feature_columns:
            self.feature_columns = self.feature_columns.drop('season')
            self.categorical_features = ['season']
        else:
            self.categorical_features = []

        self.n_features = len(self.feature_columns) + len(self.categorical_features)
        
        self.drop_features_list = drop_features_list
        
        # print("Feature columns (numerical):", self.feature_columns)
        # print("Categorical features:", self.categorical_features)

        # Target features in prediction mode
        self.pred_features = pred_features if pred_features is not None else self.feature_columns

        # Feature normalization
        self.normalize_features(scaler, filter_percent)

        # Prepare sequence data
        if mode == 'prediction':
            self.inputs, self.targets, self.input_masks, self.target_masks, self.labels = self.prepare_sequences()
        else:  # 'ae' mode
            self.inputs, _, self.input_masks, _, self.labels = self.prepare_sequences()
            self.targets = None
            self.target_masks = None

    def normalize_features(self, scaler, filter_value):
        if filter_value is not None:
            thresholds_right = self.dataframe[self.feature_columns].quantile(filter_value)
            thresholds_left = self.dataframe[self.feature_columns].quantile(1-filter_value)
            mask = ((self.dataframe[self.feature_columns] > thresholds_right) | (self.dataframe[self.feature_columns] < thresholds_left)).any(axis=1)
            self.dataframe = self.dataframe[~mask]
        
        if scaler is not None:
            self.scaler = scaler
            self.dataframe[self.feature_columns] = self.scaler.transform(self.dataframe[self.feature_columns])
        elif self.scaler_name is not 'No_Scaler': 
            if self.scaler_name == 'MinMaxScaler':
                self.scaler = MinMaxScaler()
            elif self.scaler_name == 'StandardScaler':
                self.scaler = StandardScaler()
            elif self.scaler_name == 'RobustScaler':
                self.scaler = RobustScaler()
            elif self.scaler_name == 'QuantileTransformer':
                self.scaler = QuantileTransformer()  
            else:
                raise ValueError(f"Unsupported scaler: {self.scaler_name}")
            self.dataframe[self.feature_columns] = self.scaler.fit_transform(self.dataframe[self.feature_columns])
            
    def get_sequence_lengths(self, mask):
        lengths = mask.sum(axis=1)  
        return lengths.long() 
        
    def prepare_sequences(self):
        inputs, targets, input_masks, target_masks, labels = [], [], [], [], []

        # Generate sequences by grouping by 'obj_id' and 'traj_id'
        count = 0
        for _, group in self.dataframe.groupby(['obj_id', 'traj_id']):
            count += 1
            features = group[self.feature_columns].values
            if self.categorical_features:
                # Concatenate categorical features
                categorical = group[self.categorical_features].values
                features = np.hstack((features, categorical))
            labels_df = group
            if self.mode == 'prediction' and self.pred_features:
                target_values = group[self.pred_features].values
                in_seqs, tgt_seqs, in_masks, tgt_masks, label_seqs = sliding_window_sequences(
                    features, self.seq_len, self.pred_len, self.mode, target_values, self.interpolate_method, labels_df=labels_df)
                inputs.extend(in_seqs)
                targets.extend(tgt_seqs)
                input_masks.extend(in_masks)
                target_masks.extend(tgt_masks)
                if label_seqs is not None:
                    labels.extend(label_seqs)
            else:  # Autoencoder mode
                in_seqs, in_masks, label_seqs = sliding_window_sequences(features, self.seq_len, None, self.mode, labels_df=labels_df)
                #print(in_masks)
                inputs.extend(in_seqs)
                input_masks.extend(in_masks)
                if label_seqs is not None:
                    labels.extend(label_seqs)
        print(count)
        inputs = np.array(inputs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32) if targets else None
        input_masks = np.array(input_masks, dtype=np.float32)
        target_masks = np.array(target_masks, dtype=np.float32) if target_masks else None
        label_seqs_df = pd.DataFrame(labels) if len(labels)>0 else None
        if self.filter_less_seq_len is not None:
            sequence_lengths = input_masks.sum(axis=1)  # (batch_size,)
            
            valid_indices = sequence_lengths >= self.filter_less_seq_len  # 布尔索引 (batch_size,)
            
            filtered_inputs = inputs[valid_indices]
            filtered_input_masks = input_masks[valid_indices]
            
            filtered_targets = targets[valid_indices] if targets is not None else None
            filtered_target_masks = target_masks[valid_indices] if target_masks is not None else None
            
            filtered_label_seqs_df = label_seqs_df[valid_indices] if label_seqs_df is not None else None
            
            print(f"Filtered inputs shape: {filtered_inputs.shape}")
            print(f"Filtered input_masks shape: {filtered_input_masks.shape}, before:{input_masks.shape}")
            if filtered_targets is not None:
                print(f"Filtered targets shape: {filtered_targets.shape}, before:{targets.shape}")
            if filtered_target_masks is not None:
                print(f"Filtered target_masks shape: {filtered_target_masks.shape}, before:{target_masks.shape}")
            if filtered_label_seqs_df is not None:
                print(f"Filtered labels shape: {filtered_label_seqs_df.shape}, before:{label_seqs_df.shape}")
                
            return filtered_inputs, filtered_targets, filtered_input_masks, filtered_target_masks, filtered_label_seqs_df
        else:
            return inputs, targets, input_masks, target_masks, label_seqs_df

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        data = {
            'inputs': torch.tensor(self.inputs[idx], dtype=torch.float),
            'input_masks': torch.tensor(self.input_masks[idx], dtype=torch.float)
        }

        if self.targets is not None:
            data['targets'] = torch.tensor(self.targets[idx], dtype=torch.float)
            data['target_masks'] = torch.tensor(self.target_masks[idx], dtype=torch.float) if self.target_masks is not None else None

        return data

def clean_outliers_by_quantile(dataframe, columns_to_clean, iqr_multiplier=1.5, remove_na=False):
    """
    Clean outliers in the specified columns of a DataFrame using the IQR (Interquartile Range) method.
    Optionally, remove rows with NA values.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to clean.
    columns_to_clean (list): List of column names to clean for outliers.
    iqr_multiplier (float): The multiplier for the IQR to define outlier limits (default is 1.5).
    remove_na (bool): Whether to remove rows with NA values (default is False).

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    # Copy the original DataFrame to avoid modifying it directly
    cleaned_data = dataframe.copy()

    # Optionally remove rows with NA values
    if remove_na:
        cleaned_data = cleaned_data.dropna()
        print(f"Rows with NA removed. Total rows after removing NA: {len(cleaned_data)}")

    # Apply IQR filtering for each column
    for col in columns_to_clean:
        Q1 = cleaned_data[col].quantile(0.25)  # 1st quartile (25th percentile)
        Q3 = cleaned_data[col].quantile(0.75)  # 3rd quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range

        # Calculate lower and upper limits
        lower_limit = Q1 - iqr_multiplier * IQR
        upper_limit = Q3 + iqr_multiplier * IQR

        # Print the calculated limits (optional)
        print(f"{col}: lower_limit = {lower_limit}, upper_limit = {upper_limit}")

        # Filter the DataFrame for the current column
        cleaned_data = cleaned_data[
            (cleaned_data[col] >= lower_limit) & (cleaned_data[col] <= upper_limit)
        ]

    # Print the number of rows before and after cleaning
    print(f"Total rows before cleaning: {len(dataframe)}")
    print(f"Total rows after cleaning: {len(cleaned_data)}")

    return cleaned_data
