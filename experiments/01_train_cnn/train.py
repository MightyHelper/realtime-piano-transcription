import torch
import wandb
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb as w
from model import model_of, ModelVersion
from src.common import MaestroSplitType
from src.maestro2 import MaestroDatasetSplit, FrameContextDataset, DynamicBatchIterableDataset2, custom_collate_fn, \
    custom_normalize_batch

if __name__ == '__main__':
    w.init(
        # set the wandb project where this run will be logged
        project="realtime-piano-transcription",

        # track hyperparameters and run metadata
        config={
            "hyper": {
                "learning_rate": 1e-5,
                "epochs": 4,
                "batch_size": 12,
                "num_workers": 2,
                "n_context": 128,  # Run-up of 32 frames
                "n_predict": 128, # Prediction of 32 frames
            },
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
            "architecture": ModelVersion.V2.value,
            "dataset": {
                "name": "MAESTRO",
                "split": MaestroSplitType.TRAIN.value,
                "size": 940,
                "stride": 128,
            }
        }
    )
    dataset_split = w.config['dataset']['split']
    dataset_size = w.config['dataset']['size']
    dataset_stride = w.config['dataset']['stride']
    n_context = w.config['hyper']['n_context']
    n_predict = w.config['hyper']['n_predict']
    batch_size = w.config['hyper']['batch_size']
    num_workers = w.config['hyper']['num_workers']
    epochs = w.config['hyper']['epochs']
    lr = w.config['hyper']['learning_rate']
    device = w.config['device']
    model_name = w.config['architecture']

    dataset = MaestroDatasetSplit(MaestroSplitType[dataset_split.upper()])

    print(f"Loaded metadata for {len(dataset.split.entries)} songs")
    wandb.config.update({"dataset.full_size": len(dataset.split.entries)})


    dataset.split.entries = dataset.split.entries[:dataset_size]
    dataset.split.df_entries = dataset.split.df_entries[:dataset_size]
    dataset2 = FrameContextDataset(dataset, n_context, n_predict, dataset_stride)
    wrapped_dataset = DynamicBatchIterableDataset2(dataset2, batch_size)
    data_loader = DataLoader(
        wrapped_dataset,
        batch_size=1,  # Let the collate_fn handle the final batching
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        prefetch_factor=(dataset2[0][0].shape[0] * 4) // (batch_size * num_workers),
        multiprocessing_context='spawn',
        pin_memory=True,
        pin_memory_device=device,
    )
    model = model_of(ModelVersion[model_name.upper()], n_predict, device)


    # initialize the weights
    def init_weights(m):
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.normal_(m.weight)
            m.bias.data.fill_(0.2)
        elif type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight)
            m.bias.data.fill_(0.2)


    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ## Pass dummy batch
    x, y = custom_normalize_batch(*next(iter(dataset2)))
    x = x[:batch_size].to(device)
    y = y[:batch_size].to(device)
    y_ = model.forward(x)
    loss = MSELoss()
    l = loss(y, y_)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initial batch loss: {l.item()}; {total_params=}")

    w.watch(model)
    model = model.to(device)

    try:
        for epoch in range(epochs):
            # Train
            with tqdm(total=len(data_loader)) as pbar:
                total_loss = 0
                for idx, (x, y) in enumerate(data_loader):
                    x, y = custom_normalize_batch(x, y)
                    optimizer.zero_grad()
                    output = model(x)
                    loss_val = loss(output, y)
                    loss_val.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss_val.item()
                    pbar.update(1)
                    w.log({'train_loss': loss_val.item(), 'epoch': epoch, 'batch': idx})
                    pbar.set_description(f'Epoch {epoch} - Loss-so-far: {loss_val.item()}')
                text = f'Epoch {epoch} - Loss: {total_loss}'
                pbar.set_description(text)
    except KeyboardInterrupt:
        print("Interrupted")

    w.finish()

    model_path = f'model_{hash(str(w.config))}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")