def main():
    print("=" * 80)
    print("FIRE SEGMENTATION - MULTI-SCALE FUSION U-NET")
    print("TRACKING BOTH F1 AND IoU OPTIMAL THRESHOLDS")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Checkpoint Directory: {CKPT_DIR}")

    # Load dataset files
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.tif')))
    print(f"\nTotal files found: {len(all_files)}")

    if len(all_files) == 0:
        raise FileNotFoundError(f"No .tif files found in {DATA_DIR}")

    # Train/Val split
    random.seed(42)
    random.shuffle(all_files)
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")

    # Create datasets
    train_dataset = SatelliteFireDataset(
        train_files, crop_size=CROP_SIZE, augment=True, mode='train'
    )
    val_dataset = SatelliteFireDataset(
        val_files, crop_size=CROP_SIZE, augment=False, mode='val'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False
    )

    # Initialize model
    model = MultiScaleFusionUNet(in_high=3, in_low=2, out_channels=1).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Initialize loss, optimizer, scheduler
    criterion = CombinedLoss(
        focal_alpha=FOCAL_ALPHA, focal_gamma=FOCAL_GAMMA,
        dice_weight=DICE_WEIGHT, edge_weight=EDGE_WEIGHT
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # History tracking - NOW WITH BOTH METRICS
    history = {
        'train_loss': [], 'val_loss': [], 'lr': [],
        'best_f1_value': [], 'best_iou_value': [],
        'best_f1_threshold': [], 'best_iou_threshold': [],
        'f1_at_f1_opt': [], 'f1_at_iou_opt': [],
        'iou_at_iou_opt': [], 'iou_at_f1_opt': [],
    }

    # Early stopping for BOTH metrics
    best_f1_value = 0.0
    best_iou_value = 0.0
    best_f1_threshold = 0.5
    best_iou_threshold = 0.5
    best_f1_epoch = 0
    best_iou_epoch = 0
    patience = 15
    patience_counter_f1 = 0
    patience_counter_iou = 0

    # Log file
    with open(LOG_FILE, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Model: Multi-Scale Fusion U-Net\n")
        f.write(f"Tracking: F1 and IoU with separate optimal thresholds\n")
        f.write("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (high_t, low_t, targ_t) in enumerate(train_loader):
            high_t = high_t.to(DEVICE)
            low_t = low_t.to(DEVICE)
            targ_t = targ_t.to(DEVICE)

            optimizer.zero_grad()

            logits = model(high_t, low_t)
            loss = criterion(logits, targ_t)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if batch_idx % 10 == 0:
                print(f"\rEpoch {epoch+1}/{EPOCHS} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}", end='')

        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)

        # Validation
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE, EVAL_THRESHOLDS)
        history['val_loss'].append(val_loss)

        # Find optimal thresholds for BOTH metrics
        best_thresholds, best_values, all_threshold_metrics = find_optimal_thresholds(
            model, val_loader, DEVICE, EVAL_THRESHOLDS
        )

        # Update history with BOTH metrics
        history['best_f1_value'].append(best_values['f1'])
        history['best_iou_value'].append(best_values['iou'])
        history['best_f1_threshold'].append(best_thresholds['f1'])
        history['best_iou_threshold'].append(best_thresholds['iou'])

        # Cross-metrics: F1 at IoU-opt threshold and vice versa
        f1_at_iou_opt = val_metrics[best_thresholds['iou']]['f1']
        iou_at_f1_opt = val_metrics[best_thresholds['f1']]['iou']
        history['f1_at_f1_opt'].append(best_values['f1'])
        history['f1_at_iou_opt'].append(f1_at_iou_opt)
        history['iou_at_iou_opt'].append(best_values['iou'])
        history['iou_at_f1_opt'].append(iou_at_f1_opt)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # Print epoch summary - NOW SHOWING BOTH METRICS
        print(f"\rEpoch {epoch+1}/{EPOCHS} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        print(f"  F1-Opt: τ={best_thresholds['f1']:.2f}, F1={best_values['f1']:.4f}, "
              f"IoU at this τ={iou_at_f1_opt:.4f}")
        print(f"  IoU-Opt: τ={best_thresholds['iou']:.2f}, IoU={best_values['iou']:.4f}, "
              f"F1 at this τ={f1_at_iou_opt:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Log to file
        with open(LOG_FILE, 'a') as f:
            f.write(f"Epoch {epoch+1}: "
                   f"Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}\n")
            f.write(f"  F1-Opt (τ={best_thresholds['f1']:.2f}): F1={best_values['f1']:.4f}, "
                   f"IoU={iou_at_f1_opt:.4f}\n")
            f.write(f"  IoU-Opt (τ={best_thresholds['iou']:.2f}): IoU={best_values['iou']:.4f}, "
                   f"F1={f1_at_iou_opt:.4f}\n")

        # Save best model for EACH metric
        improved = False

        if best_values['f1'] > best_f1_value:
            best_f1_value = best_values['f1']
            best_f1_threshold = best_thresholds['f1']
            best_f1_epoch = epoch
            patience_counter_f1 = 0
            improved = True

            # Save F1-best model
            checkpoint_f1 = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': 'f1',
                'best_metric_value': best_f1_value,
                'best_threshold': best_f1_threshold,
                'iou_at_this_threshold': iou_at_f1_opt,
                'history': history,
            }
            torch.save(checkpoint_f1, os.path.join(CKPT_DIR, 'best_model_f1.pth'))
            print(f"  ✓ New best F1 model! F1={best_f1_value:.4f} at τ={best_f1_threshold:.2f}")
        else:
            patience_counter_f1 += 1

        if best_values['iou'] > best_iou_value:
            best_iou_value = best_values['iou']
            best_iou_threshold = best_thresholds['iou']
            best_iou_epoch = epoch
            patience_counter_iou = 0
            improved = True

            # Save IoU-best model
            checkpoint_iou = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': 'iou',
                'best_metric_value': best_iou_value,
                'best_threshold': best_iou_threshold,
                'f1_at_this_threshold': f1_at_iou_opt,
                'history': history,
            }
            torch.save(checkpoint_iou, os.path.join(CKPT_DIR, 'best_model_iou.pth'))
            print(f"  ✓ New best IoU model! IoU={best_iou_value:.4f} at τ={best_iou_threshold:.2f}")
        else:
            patience_counter_iou += 1

        # Also save a combined "latest" checkpoint
        checkpoint_latest = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1_value': best_f1_value,
            'best_f1_threshold': best_f1_threshold,
            'best_iou_value': best_iou_value,
            'best_iou_threshold': best_iou_threshold,
            'history': history,
        }
        torch.save(checkpoint_latest, os.path.join(CKPT_DIR, 'latest_model.pth'))

        # Early stopping (only if BOTH haven't improved)
        if patience_counter_f1 >= patience and patience_counter_iou >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs!")
            break

        # Visualization every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize_predictions(
                model, val_loader, DEVICE, epoch, VIZ_DIR,
                best_f1_threshold, best_iou_threshold
            )
            plot_training_history(history, VIZ_DIR)

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n📊 F1-OPTIMIZED MODEL:")
    print(f"   Best F1: {best_f1_value:.4f} at τ={best_f1_threshold:.2f} (Epoch {best_f1_epoch+1})")
    print(f"   Saved to: {os.path.join(CKPT_DIR, 'best_model_f1.pth')}")

    print(f"\n📊 IoU-OPTIMIZED MODEL:")
    print(f"   Best IoU: {best_iou_value:.4f} at τ={best_iou_threshold:.2f} (Epoch {best_iou_epoch+1})")
    print(f"   Saved to: {os.path.join(CKPT_DIR, 'best_model_iou.pth')}")

    # Compare the two
    print(f"\n🔍 COMPARISON:")
    if best_f1_threshold != best_iou_threshold:
        print(f"   Different optimal thresholds!")
        print(f"   F1 model threshold: {best_f1_threshold:.2f} (likely more sensitive)")
        print(f"   IoU model threshold: {best_iou_threshold:.2f} (likely more conservative)")

    # Load both models and compare
    print(f"\n💡 DEPLOYMENT RECOMMENDATION:")
    if best_iou_value > 0.3:  # IoU is decent
        print(f"   ✅ Use IoU-optimized model for production (τ={best_iou_threshold:.2f})")
        print(f"   → More conservative, fewer false alarms")
    else:
        print(f"   ⚠️  IoU still low - model needs improvement")
        print(f"   → Consider data quality or architecture changes")

    plot_training_history(history, VIZ_DIR)

    return model, history
def init_weights(m):
    """Initialize weights to produce reasonable initial probabilities (~0.5)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)
