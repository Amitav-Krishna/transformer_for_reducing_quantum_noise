"""Extract train/test loss statistics for matched transformer models."""
import pandas as pd

def main():
    for loss_type in ['frob', 'physics']:
        # Training data - filter out validation and fidelity rows
        train_df = pd.read_csv(f'csvs_2/transformer_matched_{loss_type}.csv')
        train_df = train_df[train_df['chunk_id'] != 'val']
        train_df = train_df[train_df['chunk_id'] != 'fid']
        final_train = train_df.groupby('epoch')['train_loss'].mean().iloc[-1]

        # Get val loss
        val_df = pd.read_csv(f'csvs_2/transformer_matched_{loss_type}.csv')
        val_df = val_df[val_df['chunk_id'] == 'val']
        final_val = val_df['val_loss'].iloc[-1]

        print(f'Transformer-Matched {loss_type.capitalize()}:')
        print(f'  Final train loss: {final_train:.2f}')
        print(f'  Final val loss: {final_val:.2f}')
        print(f'  Gap: {final_val - final_train:.2f}')
        print()


if __name__ == "__main__":
    main()