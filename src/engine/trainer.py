"""
NAME:           train.py
VERSION:        2.0 (Link to main.py file instead of independant func.)
DESCRIPTION:    File to train ML algorithms on Q-V curve
"""

# ==========================================================================
# --------                        IMPORTS                      --------
# ==========================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time


# ==========================================================================
# --------                  TRAINING ENGINE                    --------
# ==========================================================================

def train_model(model, train_loader, val_loader,
                device, epochs, lr, save_dir, CONFIG):

    # Setup Optimiser
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (reduces LR when validation plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Early stopping parameters
    #     | If model stops improving before set epochs, just stop early.
    best_val_rmse = float('inf')
    patience = 20  # Stop if no improvement for 20 epochs
    patience_counter = 0
    
    # Training Loop
    print("\nStarting Training Loop...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            # Move to GPU/CPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward Pass
            optimiser.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward Pass
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)


        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)


                preds = model(X_val)
                batch_loss = criterion(preds, y_val)
                val_loss += batch_loss.item() * X_val.size(0)

        val_loss /= len(val_loader.dataset)
        val_rmse = np.sqrt(val_loss)
        
        # Update learning rate based on validation performance
        scheduler.step(val_rmse)

        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"\tEpoch {epoch+1:03d}/{epochs} | " \
                    f"Train Loss: {train_loss:.6f} | " \
                    f"Val RMSE: {val_rmse:.5f}")
        
        # Checkpointing + Early Stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0  # Reset patience
            save_path = os.path.join(save_dir, 
                                     f"{CONFIG['model']}.pth")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    total_time = time.time() - start_time
    print(f"\nDone!\n\n \tTotal Time: {total_time:.1f}s")
    print(f"\tBest RMSE: {best_val_rmse:.5f}")
    print(f"\tModel Saved: {os.path.abspath(save_path)}")
    
    # Load best weights before returning
    model.load_state_dict(torch.load(save_path))

    return model



if __name__ == "__main__":
    train()
