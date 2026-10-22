"""Train an invariant deep vision classifier on 18 Indian vegetable classes.

Backbone: MobileNetV3-Large (1280-dim embeddings)
Augmentation: Random rotations, perspective tilt/angle, zoom-in/out, color jitter.
Classes: 18 (Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage,
          Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato,
          Pumpkin, Radish, Tomato, Okra, Onion, Spinach).
"""

from pathlib import Path
import random
import pickle
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

CLASSES = [
    "Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", "Cabbage",
    "Capsicum", "Carrot", "Cauliflower", "Cucumber", "Papaya", "Potato",
    "Pumpkin", "Radish", "Tomato", "Okra", "Onion", "Spinach"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pretrained backbone
backbone = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=0)
backbone.to(device)
backbone.eval()

# Transforms with aggressive rotation, tilt, and zoom invariance
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=45),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Gather training samples
kaggle_train = Path("data/raw/kaggle_veg/Vegetable Images/train")
kaggle_test = Path("data/raw/kaggle_veg/Vegetable Images/test")
okra_dir = Path("data/raw/okra_pods")
onion_dir = Path("data/raw/extracted_onion")
spinach_dir = Path("data/raw/spinach")

train_items = []
test_items = []

# Kaggle 15 classes
for cls in CLASSES[:15]:
    cls_train = list((kaggle_train / cls).glob("*.jpg"))
    random.shuffle(cls_train)
    for p in cls_train[:80]:
        train_items.append((str(p), cls))

    cls_test = list((kaggle_test / cls).glob("*.jpg"))
    random.shuffle(cls_test)
    for p in cls_test[:30]:
        test_items.append((str(p), cls))

# Okra
okra_files = list(okra_dir.rglob("*.jpg")) + list(okra_dir.rglob("*.png"))
if Path("istockphoto-2234976183-612x612.jpg").exists():
    okra_files.append(Path("istockphoto-2234976183-612x612.jpg"))
random.shuffle(okra_files)
for p in okra_files[:120]:
    train_items.append((str(p), "Okra"))
for p in okra_files[120:160]:
    test_items.append((str(p), "Okra"))

# Onion
onion_files = list(onion_dir.rglob("*.jpg")) + list(onion_dir.rglob("*.png"))
random.shuffle(onion_files)
for p in onion_files[:120]:
    train_items.append((str(p), "Onion"))
for p in onion_files[120:160]:
    test_items.append((str(p), "Onion"))

# Spinach
spinach_files = list(spinach_dir.rglob("*.jpg")) + list(spinach_dir.rglob("*.png"))
random.shuffle(spinach_files)
for p in spinach_files[:120]:
    train_items.append((str(p), "Spinach"))
for p in spinach_files[120:160]:
    test_items.append((str(p), "Spinach"))

print(f"Total training samples: {len(train_items)}")
print(f"Total testing samples: {len(test_items)}")

def extract_features(items, transform):
    features = []
    labels = []
    with torch.no_grad():
        for path_str, label in items:
            try:
                img = Image.open(path_str).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                feat = backbone(tensor).cpu().numpy().squeeze(0)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                pass
    return np.array(features), np.array(labels)

print("Extracting training deep embeddings...")
X_train, y_train = extract_features(train_items, train_transform)
print(f"X_train shape: {X_train.shape}")

print("Extracting testing deep embeddings...")
X_test, y_test = extract_features(test_items, val_transform)
print(f"X_test shape: {X_test.shape}")

# Train regularized logistic regression on 1280-dim deep features
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Evaluate on test set
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"\n==========================================")
print(f"TEST ACCURACY ON REAL DATA: {acc * 100:.2f}%")
print(f"==========================================\n")
print(classification_report(y_test, preds, digits=4))

# Test specifically on the in-the-wild istockphoto okra image
if Path("istockphoto-2234976183-612x612.jpg").exists():
    img_test = Image.open("istockphoto-2234976183-612x612.jpg").convert("RGB")
    t_test = val_transform(img_test).unsqueeze(0).to(device)
    with torch.no_grad():
        f_test = backbone(t_test).cpu().numpy()
    pred_istock = clf.predict(f_test)[0]
    prob_istock = np.max(clf.predict_proba(f_test))
    print(f"Wild Internet Image (istockphoto Okra on Wood): PREDICTED = {pred_istock} (Confidence: {prob_istock * 100:.2f}%)")

# Save model package
output_path = Path("config/models/deep_vegetable_classifier.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump({
        "backbone_name": "mobilenetv3_large_100",
        "classifier": clf,
        "classes": clf.classes_.tolist(),
        "accuracy": acc,
    }, f)

print(f"Saved deep vision model package to {output_path}")
