from torch import cuda
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.common import MaestroSplitType
from src.maestro2 import MaestroDatasetSplit, FrameContextDataset, DynamicBatchIterableDataset2, custom_collate_fn
from model import get_model
import mir_eval

dataset = MaestroDatasetSplit(MaestroSplitType.VALIDATION)
n_context = 21
n_predict = 3
dataset2 = FrameContextDataset(dataset, n_context, n_predict)

batch_size = 32
num_workers = 4
device = 'cuda' if cuda.is_available() else 'cpu'
wrapped_dataset = DynamicBatchIterableDataset2(dataset2, batch_size)
data_loader = DataLoader(
    wrapped_dataset,
    batch_size=1,  # Let the collate_fn handle the final batching
    collate_fn=custom_collate_fn,
    num_workers=num_workers,
    prefetch_factor=(dataset2[0][0].shape[0]*4) // (batch_size * num_workers),
    multiprocessing_context='spawn',
    pin_memory=True,
    pin_memory_device=device,
)


model = get_model(n_predict, device)

def mir_evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches):
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        est_intervals=est_intervals,
        est_pitches=est_pitches,
    )
    return precision, recall, f1

# Evaluate the model
model.eval()
for x, y in tqdm(data_loader, total=len(data_loader)):
    x = x.to(device)
    y = y.to(device)
    output = model(x)
    # Run mir_evaluate
    ref_intervals = y[:, :, 0].cpu().numpy()
    ref_pitches = y[:, :, 1].cpu().numpy()
    est_intervals = output[:, :, 0].cpu().detach().numpy()
    est_pitches = output[:, :, 1].cpu().detach().numpy()
    precision, recall, f1 = mir_evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches)
    print(precision, recall, f1)
    break
