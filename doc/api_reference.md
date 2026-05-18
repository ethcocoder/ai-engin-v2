# API & CLI Script Reference

This specifies the core CLI entrypoints for evaluating the neural compression scripts.

## `src/train.py`
Standard training loop mapping utilizing PyTorch `DataLoader` wrappers specifically aimed at rapid testing (using low-res datasets like CIFAR). 

**Usage:**
```bash
python src/train.py --epochs 25 --batch_size 128 --latent_channels 4
```

## `src/train_hd.py`
Specialty loop explicitly designed to accept User-uploaded custom imagery dynamically scaling the `SpatialEncoder`. Used to aggressively overfit logic to verify HD transmission capacity.

**Usage:**
```bash
python src/train_hd.py --image_dir hd_images --epochs 100 --latent_channels 16
```
*(Requires a folder populated with raw photos)*

## `src/demo_hd.py`
The ultimate visual and console logging application. Scrapes the model path to push original user content through the encode/decode split protocol. Calculates mathematically explicit Byte array conversions tracking payload efficiencies.

**Usage:**
```bash
python src/demo_hd.py --model_path checkpoints/hd_compressor.pth --image_dir hd_images --latent_channels 16
```
