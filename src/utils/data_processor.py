from sklearn.preprocessing import LabelEncoder


class DataProcessor:
    """Utility class for data processing operations"""

    @staticmethod
    def clean_column_names(df):
        """Clean and standardize column names"""
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
        return df

    @staticmethod
    def handle_missing_values(df):
        """Handle missing values in the dataset"""
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                )
            else:
                df[col] = df[col].fillna(df[col].median())
        return df

    @staticmethod
    def encode_categorical_features(df, columns, encoders=None):
        """Encode categorical features using LabelEncoder"""
        if encoders is None:
            encoders = {}

        for col in columns:
            if col not in encoders:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = encoders[col].transform(df[col].astype(str))

        return df, encoders
