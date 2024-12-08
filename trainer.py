import torch


def main(ctx, device, dataloaders, model, avg_model, optimizer, criterion):
    torch.set_float32_matmul_precision("high")

    cfg = ctx.config
    logger = ctx.logger
    print(f"MLXP logging on with ID: {logger.log_id}")

    # Dataloader
    train_dataloader, val_dataloader = dataloaders
    val_iterator = iter(val_dataloader)

    # Switching to device
    model = model.to(device)
    if avg_model is not None:
        avg_model = avg_model.to(device)
    criterion = criterion.to(device)

    # Compiling
    if cfg.train.compile:
        model = torch.compile(model)
        if avg_model is not None:
            avg_model = torch.compile(avg_model)
        criterion = torch.compile(criterion)

    len_print_epoch = len(str(cfg.train.epochs))

    # Best val loss
    best_val_loss = float("inf")
    counter = 0

    # Training loop
    print("Starting training...")
    for epoch in range(cfg.train.epochs):
        # Reset metrics
        train_loss = 0.0
        train_acc = 0.0
        train_steps = 0
        val_loss = 0.0
        val_acc = 0.0
        val_steps = 0
        it_train_loss = 0.0
        it_val_loss = 0.0
        it_avg_train_loss = 0.0
        it_avg_val_loss = 0.0

        # Train one epoch
        for train_batch in train_dataloader:
            # Eval
            try:
                val_batch = next(val_iterator)
            except StopIteration:
                val_iterator = iter(val_dataloader)
                val_batch = next(val_iterator)

            # One step of validation
            with torch.no_grad():
                model.eval()
                inputs, targets = val_batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                preds = model(inputs)
                loss = criterion(preds, targets)
                it_val_loss = loss.item()
                val_loss += it_val_loss
                val_acc += torch.mean((preds.argmax(dim=-1) == targets).float()).item()
                val_steps += 1

                if avg_model is not None:
                    avg_model.eval()
                    preds = avg_model(inputs)
                    loss = criterion(preds, targets)
                    it_avg_val_loss = loss.item()

            # Train one step
            model.train()
            inputs, targets = train_batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = criterion(preds, targets)
            it_train_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            train_acc += torch.mean((preds.argmax(dim=-1) == targets).float()).item()
            train_steps += 1

            if avg_model is not None:
                avg_model.update_parameters(model)
                avg_model.train()
                with torch.no_grad():
                    preds = avg_model(inputs)
                    loss = criterion(preds, targets)
                    it_avg_train_loss = loss.item()

            logger.log_metrics(
                {
                    "train/loss": it_train_loss,
                    "val/loss": it_val_loss,
                    "avg_train/loss": it_avg_train_loss,
                    "avg_val/loss": it_avg_val_loss,
                },
                log_name="train_metrics",
            )

        # End of epoch
        print(f"[{epoch+1:>{len_print_epoch}} / {cfg.train.epochs}] ", end="")
        train_loss /= train_steps
        train_acc /= train_steps
        val_loss /= val_steps
        val_acc /= val_steps

        print(f"train/loss: {train_loss:>6.4e} ", end="")
        print(f"train/acc: {train_acc:>6.4e} ", end="")
        print(f"val/loss: {val_loss:>6.4e} ", end="")
        print(f"val/acc: {val_acc:>6.4e} ", end="")
        print("", end="\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
        if counter == cfg.train.early_stopping_patience:
            print("Early stopping: val loss did not improve for", counter, "epochs.")
            break
