from sklearn.model_selection import train_test_split
import pandas as pd

def split_subsets(df, n=2, stratify=None, verbose=False):
    subsets = [] 
    
    count = 1
    
    remaining_df = df
    
    for i in range(n):

        if count == n:
            subsets.append(remaining_df)
            
        else: 
            X = remaining_df.drop(stratify, axis=1)
            
            y = remaining_df[stratify]
            
            subset_size = (1 / n)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=subset_size, stratify=y)
    
            # Concatenate X_train and y_train to form the remain_df dataframe
            remain_df = pd.concat([X_train, y_train], axis=1)
            
            # Concatenate X_test and y_test to form the subset dataframe
            subset_df = pd.concat([X_test, y_test], axis=1)
    
            # Add subset_df into list
            subsets.append(subset_df)

            # Assign new remain_df to remaing_df
            remaining_df = remain_df
            
            
        if verbose:
            print(f"Subset: {count} is ready")
            
        count += 1
    
    return subsets