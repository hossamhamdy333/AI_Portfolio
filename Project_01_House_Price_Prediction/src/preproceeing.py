import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def remove_outliers(train, y, threshold=4000):
    """
    Remove outlier houses with GrLivArea > threshold and suspiciously low price.
    
    Args:
        train: training dataframe
        y: target series (log-transformed SalePrice)
        threshold: GrLivArea cutoff (default 4000 sqft)
    
    Returns:
        train, y with outliers removed
    """
    outlier_idx = train[train['GrLivArea'] > threshold].index
    train = train.drop(outlier_idx)
    y = y.drop(outlier_idx)
    print(f"Removed {len(outlier_idx)} outliers. Train shape: {train.shape}")
    return train, y


def handle_missing_values(all_data):
    """
    Handle all missing values in the combined train+test dataset.
    
    Strategy:
        - Categorical NaN = 'None' (feature doesn't exist)
        - Numerical NaN = 0 (no area/count)
        - LotFrontage = neighborhood median
        - MSZoning = subclass mode
        - Others = global mode
    
    Args:
        all_data: combined train+test dataframe
    
    Returns:
        all_data with no missing values
    """
    # Categorical — NA means feature doesn't exist
    none_features = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'MasVnrType'
    ]
    for col in none_features:
        all_data[col] = all_data[col].fillna('None')

    # Numerical — NA means 0
    zero_features = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_features:
        all_data[col] = all_data[col].fillna(0)

    # LotFrontage — use neighborhood median (local context)
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage']\
        .transform(lambda x: x.fillna(x.median()))

    # MSZoning — use subclass mode
    all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning']\
        .transform(lambda x: x.fillna(x.mode()[0]))

    # Others — use global mode
    cat_cols_mode = ['Electrical', 'KitchenQual', 'Exterior1st',
                     'Exterior2nd', 'SaleType', 'Functional']
    for col in cat_cols_mode:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # Drop Utilities — near zero variance, no predictive power
    all_data.drop(['Utilities'], axis=1, inplace=True)

    # Verify
    remaining = all_data.isnull().sum().sum()
    print(f"Missing values remaining: {remaining}")
    
    return all_data


def fix_skewness(all_data, threshold=0.75):
    """
    Apply log1p transform to highly skewed numerical features.
    
    Args:
        all_data: combined train+test dataframe
        threshold: skewness threshold (default 0.75)
    
    Returns:
        all_data with skewness corrected
    """
    num_features = all_data.select_dtypes(include=[np.number]).columns
    skewed = all_data[num_features].apply(lambda x: x.skew())
    high_skew = skewed[skewed > threshold].index

    for col in high_skew:
        all_data[col] = np.log1p(all_data[col])

    print(f"Applied log1p to {len(high_skew)} skewed features")
    return all_data


def engineer_features(all_data):
    """
    Create new features from existing ones.
    
    New features:
        TotalSF           — total living area across all floors
        TotalBathrooms    — weighted bathroom count
        HouseAge          — age at time of sale
        YearsSinceRemodel — years since last remodel
        Remodeled         — binary flag
        TotalPorchSF      — combined porch area
        HasPool           — binary flag
        HasGarage         — binary flag
        HasBsmt           — binary flag
        HasFireplace      — binary flag
        OverallScore      — quality × condition interaction
        GarageScore       — garage cars × area interaction
    
    Args:
        all_data: combined train+test dataframe
    
    Returns:
        all_data with new features added
    """
    # Total living area
    all_data['TotalSF'] = (all_data['TotalBsmtSF'] +
                            all_data['1stFlrSF'] +
                            all_data['2ndFlrSF'])

    # Weighted bathroom count
    all_data['TotalBathrooms'] = (all_data['FullBath'] +
                                   0.5 * all_data['HalfBath'] +
                                   all_data['BsmtFullBath'] +
                                   0.5 * all_data['BsmtHalfBath'])

    # Age features
    all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
    all_data['YearsSinceRemodel'] = all_data['YrSold'] - all_data['YearRemodAdd']
    all_data['Remodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)

    # Porch area
    all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] +
                                 all_data['EnclosedPorch'] +
                                 all_data['3SsnPorch'] +
                                 all_data['ScreenPorch'])

    # Binary flags
    all_data['HasPool']      = (all_data['PoolArea'] > 0).astype(int)
    all_data['HasGarage']    = (all_data['GarageArea'] > 0).astype(int)
    all_data['HasBsmt']      = (all_data['TotalBsmtSF'] > 0).astype(int)
    all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)

    # Interaction features
    all_data['OverallScore'] = all_data['OverallQual'] * all_data['OverallCond']
    all_data['GarageScore']  = all_data['GarageCars'] * all_data['GarageArea']

    print(f"Feature engineering complete. Shape: {all_data.shape}")
    return all_data


def encode_features(all_data):
    """
    Encode categorical features.
    
    Strategy:
        - Ordinal encoding for quality features (order matters)
        - Label encoding for other ordinal features
        - One-hot encoding for nominal features
    
    Args:
        all_data: combined train+test dataframe
    
    Returns:
        all_data fully encoded, ready for modeling
    """
    # Ordinal — quality has meaningful order
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    ordinal_features = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
        'HeatingQC', 'KitchenQual', 'FireplaceQu',
        'GarageQual', 'GarageCond', 'PoolQC'
    ]
    for col in ordinal_features:
        all_data[col] = all_data[col].map(quality_map)

    # Label encoding for other ordinal features
    label_features = ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                      'GarageFinish', 'Fence', 'LotShape', 'LandContour',
                      'LandSlope', 'Alley']
    for col in label_features:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))

    # One-hot encoding for nominal features
    all_data = pd.get_dummies(all_data)

    print(f"Encoding complete. Final shape: {all_data.shape}")
    return all_data


def full_pipeline(train, test, y):
    """
    Run the complete preprocessing pipeline.
    
    Args:
        train: raw training dataframe
        test:  raw test dataframe
        y:     log-transformed target series
    
    Returns:
        X_train, X_test, y ready for modeling
    """
    # Step 1 — Remove outliers
    train, y = remove_outliers(train, y)

    # Step 2 — Combine for consistent transformations
    ntrain = train.shape[0]
    all_data = pd.concat([train, test], axis=0, ignore_index=True)

    # Step 3 — Handle missing values
    all_data = handle_missing_values(all_data)

    # Step 4 — Fix skewness
    all_data = fix_skewness(all_data)

    # Step 5 — Engineer features
    all_data = engineer_features(all_data)

    # Step 6 — Encode
    all_data = encode_features(all_data)

    # Step 7 — Split back
    X_train = all_data[:ntrain].astype(float)
    X_test  = all_data[ntrain:].astype(float)

    print(f"\nPipeline complete:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y:       {y.shape}")

    return X_train, X_test, y