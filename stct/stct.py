import torch
import torch_dct


def sdct_torch(signals, frame_length, frame_step, window=torch.hamming_window):
    #Returns dct : Real-valued F-T domain DCT matrix/matrixes, a `[..., frame_length, n_frames]` tensor
    framed = signals.unfold(-1, frame_length, frame_step)
    if callable(window):
        window = window(frame_length).to(framed)
    if window is not None:
        framed = framed * window
    return torch_dct.dct(framed, norm="ortho").transpose(-1, -2)


def isdct_torch(dcts, *, frame_step, frame_length=None, window=torch.hamming_window):
    #inverse
    *_, frame_length2, n_frames = dcts.shape
    assert frame_length in {None, frame_length2}
    signals = torch_overlap_add(
        torch_dct.idct(dcts.transpose(-1, -2), norm="ortho").transpose(-1, -2),
        frame_step=frame_step,
    )
    if callable(window):
        window = window(frame_length2).to(signals)
    if window is not None:
        window_frames = window[:, None].expand(-1, n_frames)
        window_signal = torch_overlap_add(window_frames, frame_step=frame_step)
        signals = signals / window_signal
    return signals


def torch_overlap_add(framed, *, frame_step, frame_length=None):
    *rest, frame_length2, n_frames = framed.shape
    assert frame_length in {None, frame_length2}
    return torch.nn.functional.fold(
        framed.reshape(-1, frame_length2, n_frames),
        output_size=(((n_frames - 1) * frame_step + frame_length2), 1),
        kernel_size=(frame_length2, 1),
        stride=(frame_step, 1),
    ).reshape(*rest, -1)
