import torch
from torchvision.transforms.functional import to_pil_image
from src.model.aether_codec import AetherCodec
from src.train.dataset import ImageFolderDataset
from torchvision import transforms

def test_pipeline():
    print("Initializing AetherCodec-Elite...")
    model = AetherCodec()
    try:
        model.load_state_dict(torch.load('stage3_elite_final.pth', weights_only=True))
        print("Successfully loaded 'stage3_elite_final.pth'")
    except FileNotFoundError:
        print("Warning: stage3_elite_final.pth not found. Using untrained weights for demonstration.")
    
    model.eval()

    # Get a test image
    print("Loading test image from DIV2K...")
    try:
        dataset = ImageFolderDataset('dataset/DIV2K_train_HR', transform=transforms.ToTensor())
        test_image_tensor = dataset[0][0].unsqueeze(0)
    except Exception as e:
        print("Could not load DIV2K image. Creating a dummy 256x256 image for test.")
        test_image_tensor = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        print("\n--- SENDER DEVICE ---")
        print("Compressing image into math (latent tensor)...")
        y_math = model.encoder(test_image_tensor)
        
        y_payload, _ = model.y_quantizer(y_math, force_hard=True)
        
        num_elements = y_payload.numel()
        estimated_kilobytes = (num_elements * 1.5) / 8 / 1024
        print(f"📡 Math Payload Generated! Estimated Network Size: {estimated_kilobytes:.2f} KB")

        print("\n--- NETWORK TRANSMISSION ---")
        print("Sending Payload...")

        print("\n--- RECEIVER DEVICE ---")
        print("Receiving math payload and hallucinating image using GAN...")
        synthesized_image_tensor = model.decoder(y_payload)
        
    print("\nProcess Complete! Check output tensors.")
    # In colab, showing PIL image directly from script blocks sometimes fails, so we just print success.
    print(f"Original Image Shape: {test_image_tensor.shape}")
    print(f"Synthesized Image Shape: {synthesized_image_tensor.clamp(0, 1).shape}")

if __name__ == "__main__":
    test_pipeline()
