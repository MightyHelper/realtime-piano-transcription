import mir_eval
import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from model import model_of, ModelVersion
from src.maestro2 import LOWEST_MIDI_NOTE
from src.common import MidiWrapper, midi_to_hz
from src.maestro2 import custom_normalize_batch
from src.common import MaestroSplitType
from src.maestro2 import MaestroDatasetSplit, FrameContextDataset, DynamicBatchIterableDataset2, custom_collate_fn
from matplotlib import pyplot as plt

if __name__ == '__main__':
    dataset = MaestroDatasetSplit(MaestroSplitType.TRAIN)
    print(len(dataset.split.entries))
    loader = FrameContextDataset(dataset, 32, 32, 32)
    iterable_loader = DynamicBatchIterableDataset2(loader, 4)
    train_loader = DataLoader(
        iterable_loader,
        batch_size=1,  # Let the collate_fn handle the final batching
        collate_fn=custom_collate_fn,
        num_workers=2,
        prefetch_factor=2,
        multiprocessing_context='spawn',
        pin_memory=True,
        pin_memory_device='cuda',
    )

    model = model_of(ModelVersion.V2, 4, 'cuda')
    model.load_state_dict(torch.load('model_6643467814334053095.pth', weights_only=True))
    model.eval()

    x, y = next(iter(train_loader))
    x, y = custom_normalize_batch(x, y)
    y_ = model.forward(x)

    slices = []

    for i in range(y_.shape[0]):
        xnp = x[i, 0].transpose(0, 1).cpu().detach().numpy()
        y_np = y_[i, 0].transpose(0, 1).cpu().detach().numpy()
        ynp = y[i, 0].transpose(0, 1).cpu().detach().numpy()
        midiw = MidiWrapper.from_piano_roll(y_np, note_offset=LOWEST_MIDI_NOTE)
        midiw2 = MidiWrapper.from_piano_roll(ynp, note_offset=LOWEST_MIDI_NOTE)
        pm1 = midiw.midi
        pm2 = midiw2.midi
        computed_mse = MSELoss()(y, y_)
        computed_mse.item()
        notes1 = [(note.start, note.end, midi_to_hz(note.pitch)) for note in pm1.instruments[0].notes]
        notes2 = [(note.start, note.end, midi_to_hz(note.pitch)) for note in pm2.instruments[0].notes]

        # Convert to the format required by mir_eval: intervals and pitches
        ref_intervals = np.array([[note[0], note[1]] for note in notes1]).reshape(-1, 2)
        ref_pitches = np.array([note[2] for note in notes1])
        est_intervals = np.array([[note[0], note[1]] for note in notes2]).reshape(-1, 2)
        est_pitches = np.array([note[2] for note in notes2])

        # Compute F1 score using mir_eval
        precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, est_intervals, est_pitches
        )

        print(f"Batch {i} Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        print(f"Computed MSE: {computed_mse}")
        items = [
            xnp,
            np.zeros_like(xnp[0:25, :]),
            y_np,
            np.ones_like(xnp[0:25, :]),
            ynp,
        ]
        for item in items:
            print(item.shape)
        # Save the piano roll as an image
        #disable interpolation
        vstack = np.vstack(items)
        plt.imsave(f"imgs/batch_{i}.png", vstack, cmap='gray')
        slices.append(vstack)

    plt.imsave("imgs/all.png", np.hstack(slices), cmap='gray')
    plt.imshow(np.hstack(slices), cmap='gray', interpolation='none')
    plt.show()
