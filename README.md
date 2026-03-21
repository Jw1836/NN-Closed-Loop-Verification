# ReLU Lyapunov Function Certification

## Usage

### Train

```bash
python -u -m relu_vnn train --checkpoint-dir runs/bilen_50it --max-iterations 50 --epochs 600 --hidden-size 30 --device mps --max-workers 4 problems/bilinear_oscillator.py
```

### Verify

```bash
python -u -m relu_vnn verify  --device mps --max-workers 4 --hidden-size 30 --checkpoint runs/v2bilen/initial_train.pt problems/bilinear_oscillator.py
```