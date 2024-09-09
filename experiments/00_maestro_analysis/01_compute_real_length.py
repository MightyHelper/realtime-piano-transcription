from functools import partial
from multiprocessing import Pool
import pandas as pd
from tqdm.auto import tqdm
from src.common import MaestroDataset, TimeKeeper

# The dataset contains a "length" column in seconds, but it's not accurate. We will compute the real length of each audio file in frames.

target_csv_path = "./maestro-v3.0.0-extended.csv"


def main():
    dataset = MaestroDataset()
    output = {}
    with Pool(32) as p:
        for audio_filename, min_duration in tqdm(
                p.imap_unordered(
                    partial(compute_duration, dataset), range(dataset.length())
                ),
                total=dataset.length()
        ):
            output[audio_filename] = min_duration

    df = pd.DataFrame.from_dict(output, orient='index', columns=['length'])
    df.to_csv(target_csv_path)
    print("Done")


def compute_duration(dataset, item):
    entry = dataset.load_index(item)
    audio = entry.load_audio
    midi = entry.load_midi
    mel = audio.compute_log_mel_spectrogram()
    roll = midi.get_piano_roll()
    mel_duration = mel.shape[1]
    roll_duration = roll.shape[1]
    min_duration = min(mel_duration, roll_duration)
    max_duration = max(mel_duration, roll_duration)
    if abs(mel_duration - roll_duration) > 0.1 * max_duration:
        raise ValueError(
            f"Mel and roll duration mismatch by {abs(mel_duration - roll_duration)}"
            f" frames ({TimeKeeper.frames_to_ms(abs(mel_duration - roll_duration))}ms)"
        )

    filename = entry.audio_filename
    del entry
    return filename, min_duration


if __name__ == "__main__":
    main()
