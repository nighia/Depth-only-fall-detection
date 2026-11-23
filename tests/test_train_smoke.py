import numpy as np
import pytest
torch = pytest.importorskip('torch')
from fall_detection.models.gru import DepthFeatureGRU
from fall_detection.training.train_gru import main


def test_gru_output_finite_after_scaling():
    assert torch.isfinite(DepthFeatureGRU()(torch.rand(2, 25, 14))).all()


def test_training_runs_and_saves_checkpoint(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 25, 14)).astype(np.float32); x[0, 2, 6] = np.nan
    y = rng.integers(0, 3, 60)
    npz = tmp_path / 'd.npz'
    np.savez(npz, X_train=x, y_train=y, X_val=x[:20], y_val=y[:20])
    out = tmp_path / 'm.pt'
    main(['--features', str(npz), '--epochs', '2', '--output', str(out),
          '--config', 'configs/experiment_gru.yaml'])
    ck = torch.load(out, weights_only=False)
    assert ck['classes'] == ['normal', 'hard_negative_motion', 'fall'] and 'scaler' in ck
