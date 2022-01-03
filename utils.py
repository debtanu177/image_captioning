import torch
import torchvvision.transforms as transforms
from PIL import Image


main_path = "/ssd_scratch/cvit/debtanu.gupta"


def print_examples(model, device, dataset):
	transform = transforms.Compose(
		[
			transforms.Resize((299, 299)),
			transforms.ToTensor(),
			transforms.normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		]
	)

	model.eval()
	test_img1 = transform(Image.open(os.path.join(main_path, "dog.jpg")).convert("RGB")).unsqueeze(0)
	print("Example 1 Correct: Dog on a beach by the ocean")
	print("Example 1 Output: " + " ".join(model.caption_image(test_img1.to(device), dataset.vocab)))


	test_img2 = transform(Image.open(os.path.join(main_path, "child.jpg")).convert("RGB")).unsqueeze(0)
	print("Example 2 Correct: Child holding red frisbee outdoors")
	print("Example 2 Output: " + " ".join(model.caption_image(test_img2.to(device), dataset.vocab)))


	test_img3 = transform(Image.open(os.path.join(main_path, "bus.png")).convert("RGB")).unsqueeze(0)
	print("Example 3 Correct: Bus driving by parked cars")
	print("Example 3 Output: " + " ".join(model.caption_image(test_img3.to(device), dataset.vocab)))


	test_img4 = transform(Image.open(os.path.join(main_path, "boat.png")).convert("RGB")).unsqueeze(0)
	print("Example 4 Correct: A small boat in the ocean")
	print("Example 4 Output: " + " ".join(model.caption_image(test_img4.to(device), dataset.vocab)))

	test_img5 = transform(Image.open(os.path.join(main_path, "horse.png")).convert("RGB")).unsqueeze(0)
	print("Example 5 Correct: A cowboy ridign a horse in the ocean")
	print("Example 5 Output: " + " ".join(model.caption_image(test_img5.to(device), dataset.vocab)))

	model.train()


def save_checkpoint(state, file_name="my_checkpoint.pth.tar"):
	print("=> Saving checkoint")
	torch.save(state, file_name)



def load_checkpoint(checkpoint, model, optimizer):
	print("=> Loading checkpoint")
	model.load_state_dict(checkpoint["state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer"])
	step = checkpoint["step"]
	return step


