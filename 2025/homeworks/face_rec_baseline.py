# face_rec_baseline.py
# -*- coding: utf-8 -*-
import os, random, math, time, json, argparse, glob
from dataclasses import dataclass
from typing import List, Dict, Tuple
from PIL import Image

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

# -----------------------------
# УТИЛИТЫ: reproducibility
# -----------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Для полной детерминированности можно включить это, но иногда замедляет:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# -----------------------------
# ПАРСИНГ ДАТОСА
# Ожидаем структуру: root/person_id/*.jpg|png
# -----------------------------
@dataclass
class PersonIndex:
    label_to_paths: Dict[int, List[str]]
    label_to_name: Dict[int, str]
    paths: List[str]
    labels: List[int]
    n_classes: int

def build_index(root: str, valid_exts={".jpg", ".jpeg", ".png", ".bmp"}, quick_ratio: float = 1.0) -> PersonIndex:
    persons = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    label_to_paths, label_to_name = {}, {}
    all_paths, all_labels = [], []
    for lbl, name in enumerate(persons):
        folder = os.path.join(root, name)
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if os.path.splitext(f.lower())[1] in valid_exts]
        files = sorted(files)
        if not files:
            continue
        if quick_ratio < 1.0:
            k = max(1, int(len(files) * quick_ratio))
            files = random.sample(files, k)
        label_to_paths[lbl] = files
        label_to_name[lbl] = name
        all_paths.extend(files)
        all_labels.extend([lbl] * len(files))
    return PersonIndex(label_to_paths, label_to_name, all_paths, all_labels, len(label_to_paths))

# -----------------------------
# ТРАНСФОРМЫ
# -----------------------------
def make_transforms(img_size=128, aug=True):
    if aug:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.03),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    return tfm

# -----------------------------
# DATASETЫ
# -----------------------------
class SingleImageDataset(Dataset):
    """Обычный датасет картинок -> (image, label)"""
    def __init__(self, index: PersonIndex, transform):
        self.paths = index.paths
        self.labels = index.labels
        self.transform = transform

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        y = self.labels[i]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, y

class PairDataset(Dataset):
    """Онлайн-сэмплинг пар: (img1, img2, y_same)"""
    def __init__(self, index: PersonIndex, transform, pairs_per_epoch=20000, pos_fraction=0.5):
        self.index = index
        self.transform = transform
        self.pairs_per_epoch = pairs_per_epoch
        self.pos_fraction = pos_fraction
        # предварительно соберем удобные списки
        self.by_lbl = index.label_to_paths
        self.labels = list(self.by_lbl.keys())
        # подготовим кэш: на каждую эпоху создаём список спецификаций пар
        self.specs = self._resample_specs()

    def _resample_specs(self):
        specs = []
        n_pos = int(self.pairs_per_epoch * self.pos_fraction)
        n_neg = self.pairs_per_epoch - n_pos

        # positive
        for _ in range(n_pos):
            lbl = random.choice(self.labels)
            xs = self.by_lbl[lbl]
            if len(xs) < 2:  # редкий случай
                continue
            a, p = random.sample(xs, 2)
            specs.append((a, p, 1))

        # negative
        for _ in range(n_neg):
            la, ln = random.sample(self.labels, 2)
            a = random.choice(self.by_lbl[la])
            n = random.choice(self.by_lbl[ln])
            specs.append((a, n, 0))

        random.shuffle(specs)
        return specs

    def __len__(self): return len(self.specs)
    def __getitem__(self, i):
        a, b, y = self.specs[i]
        ia = self.transform(Image.open(a).convert("RGB"))
        ib = self.transform(Image.open(b).convert("RGB"))
        return ia, ib, torch.tensor([y], dtype=torch.float32)

    def on_epoch_end(self):
        self.specs = self._resample_specs()

# -----------------------------
# Сэмплер "m-per-class" для триплетов
# -----------------------------
class BalancedBatchSampler(Sampler[List[int]]):
    """
    В каждом батче K классов по M изображений = K*M = batch_size.
    """
    def __init__(self, labels: List[int], m_per_class=4, num_classes_per_batch=8, drop_last=True):
        self.labels = np.array(labels)
        self.m = m_per_class
        self.k = num_classes_per_batch
        self.drop_last = drop_last
        self.class_to_indices = {}
        for idx, y in enumerate(self.labels):
            self.class_to_indices.setdefault(int(y), []).append(idx)
        self.classes = list(self.class_to_indices.keys())

        # курсоры по классам
        self.ptrs = {c: 0 for c in self.classes}
        for c in self.classes:
            random.shuffle(self.class_to_indices[c])

        # посчитаем примерное число батчей
        # ограничение — по самому короткому классу
        min_len = min(len(v) for v in self.class_to_indices.values())
        self.num_batches = (min_len // self.m) * (len(self.classes) // self.k)

    def __len__(self):
        return self.num_batches if self.drop_last else self.num_batches + 1

    def __iter__(self):
        classes = self.classes[:]
        random.shuffle(classes)
        # «страница» по k классов
        for i in range(0, len(classes) - self.k + 1, self.k):
            chosen = classes[i:i+self.k]
            batch = []
            for c in chosen:
                # если не хватает — перетасуем и начнем заново
                if self.ptrs[c] + self.m > len(self.class_to_indices[c]):
                    random.shuffle(self.class_to_indices[c])
                    self.ptrs[c] = 0
                sl = self.class_to_indices[c][self.ptrs[c]: self.ptrs[c] + self.m]
                batch.extend(sl)
                self.ptrs[c] += self.m
            yield batch

# -----------------------------
# МОДЕЛИ
# -----------------------------
class SmallEncoder(nn.Module):
    """
    Маленькая CNN с нуля. На 128x128 работает быстро и стабильно.
    """
    def __init__(self, emb_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*8*8, 512), nn.ReLU(),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        # L2-нормировка эмбеддинга — полезна для триплетов и косинуса
        x = F.normalize(x, dim=1)
        return x

class PairClassifier(nn.Module):
    """
    Сиамская архитектура: два энкодера с общими весами → разностные признаки → логит.
    """
    def __init__(self, emb_dim=128):
        super().__init__()
        self.encoder = SmallEncoder(emb_dim=emb_dim)
        # Head на |a-b|, a*b, [a, b]
        in_dim = emb_dim*2 + emb_dim + emb_dim  # [a, b, |a-b|, a*b]
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)  # логит
        )

    def forward(self, xa, xb):
        ea = self.encoder(xa)
        eb = self.encoder(xb)
        feat = torch.cat([ea, eb, torch.abs(ea-eb), ea*eb], dim=1)
        logit = self.head(feat).squeeze(1)
        return logit

# -----------------------------
# ТРИПЛЕТ-ЛОСС (batch версия)
# -----------------------------
class BatchTripletLoss(nn.Module):
    """
    Полная batch-версия: все валидные тройки (A,P,N).
    - margin: отступ (alpha)
    - avg_over: 'violating' (усреднять по тройкам с L>0) или 'valid' (по всем валидным тройкам)
    """
    def __init__(self, margin: float = 0.2, avg_over: str = "violating"):
        super().__init__()
        assert avg_over in ("violating", "valid")
        self.margin = margin
        self.avg_over = avg_over

    @staticmethod
    def pairwise_dist2(x: torch.Tensor) -> torch.Tensor:
        # ||xi-xj||^2 = ||xi||^2 - 2 xi·xj + ||xj||^2  (стабильно и быстро)
        dot = x @ x.t()             # [B,B]
        sq = torch.diag(dot).unsqueeze(1)
        dist2 = torch.clamp(sq - 2*dot + sq.t(), min=0.0)
        return dist2

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: [B,D], labels: [B]
        B = embeddings.size(0)
        device = embeddings.device
        dist2 = self.pairwise_dist2(embeddings)  # [B,B]

        labels = labels.to(device)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)     # [B,B]
        pos_mask = same & (~torch.eye(B, dtype=torch.bool, device=device))
        neg_mask = ~same

        # (A,P) для положительных, (A,N) для отрицательных
        # расширим до [B,B,B]: для каждого A фиксируем P и N
        D_ap = dist2.unsqueeze(2)  # [B,B,1]
        D_an = dist2.unsqueeze(1)  # [B,1,B]

        pos_mask_3d = pos_mask.unsqueeze(2)  # [B,B,1]
        neg_mask_3d = neg_mask.unsqueeze(1)  # [B,1,B]

        # валидные тройки имеют pos & neg одновременно
        valid = pos_mask_3d & neg_mask_3d    # [B,B,B]

        # L = max( D_ap - D_an + margin, 0 )
        loss_mat = D_ap - D_an + self.margin
        loss_mat = torch.where(valid, loss_mat, torch.zeros_like(loss_mat))
        loss_mat = torch.clamp(loss_mat, min=0.0)

        if self.avg_over == "violating":
            num = (loss_mat > 0).sum().item()
        else:
            num = valid.sum().item()

        if num == 0:
            return torch.zeros((), device=device, dtype=embeddings.dtype)
        return loss_mat.sum() / num

# -----------------------------
# ОБУЧЕНИЕ: БИНАРНАЯ КЛАССИФИКАЦИЯ ПАР
# -----------------------------
def train_pairs(
    train_pairs: PairDataset, val_pairs: PairDataset,
    img_size=128, batch_size=64, epochs=15, lr=3e-4, weight_decay=1e-4,
    ckpt_path="pair_best.pt", device="cuda"
):
    model = PairClassifier(emb_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_pairs, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_pairs,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_auc = 0.0
    for epoch in range(epochs):
        model.train()
        train_losses = []
        y_true_all, y_score_all = [], []
        for a,b,y in tqdm(train_loader, desc=f"[Pairs] epoch {epoch}", leave=False):
            a,b,y = a.to(device), b.to(device), y.to(device).squeeze(1)
            logits = model(a,b)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())
            with torch.no_grad():
                y_true_all.append(y.detach().float().cpu())
                y_score_all.append(torch.sigmoid(logits.detach().cpu()))

        train_auc = roc_auc_score(torch.cat(y_true_all).numpy(), torch.cat(y_score_all).numpy())

        # валидация
        model.eval()
        y_true_all, y_score_all, val_losses = [], [], []
        with torch.no_grad():
            for a,b,y in val_loader:
                a,b,y = a.to(device), b.to(device), y.to(device).squeeze(1)
                logits = model(a,b)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                y_true_all.append(y.float().cpu())
                y_score_all.append(torch.sigmoid(logits.cpu()))
        val_auc = roc_auc_score(torch.cat(y_true_all).numpy(), torch.cat(y_score_all).numpy())

        scheduler.step(val_auc)
        print(f"Epoch {epoch:02d}  train_loss={np.mean(train_losses):.4f}  train_auc={train_auc:.4f}  val_loss={np.mean(val_losses):.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(),
                        "config": {"img_size": img_size}}, ckpt_path)
            print(f"  ✓ New best AUC={best_auc:.4f} → saved {ckpt_path}")

        # ресемплим пары для след. эпохи (важно!)
        train_pairs.on_epoch_end()
        val_pairs.on_epoch_end()

    return best_auc

# -----------------------------
# ОБУЧЕНИЕ: ТРИПЛЕТЫ
# -----------------------------
def evaluate_auc_from_encoder(encoder: nn.Module, val_index: PersonIndex, transform, device="cuda", pairs_eval=20000):
    """
    Оцениваем AUC по парам, созданным на лету из валидационного индекса.
    Скорами берём -euclidean_distance (или -||a-b||^2).
    """
    encoder.eval()
    # предвычислим эмбеддинги всех валидационных картинок
    all_paths = val_index.paths
    all_labels = val_index.labels
    embs, ys = [], []
    with torch.no_grad():
        for p, y in tqdm(list(zip(all_paths, all_labels)), desc="[Eval] encode", leave=False):
            x = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            e = encoder(x)
            embs.append(e.squeeze(0).cpu().numpy())
            ys.append(y)
    embs = np.stack(embs)     # [N,D]
    ys   = np.array(ys)       # [N]

    # сэмплим пары
    rng = np.random.default_rng(123)
    N = len(ys)
    pos_scores, pos_labels = [], []
    neg_scores, neg_labels = [], []
    # половина позитивов, половина негативов
    half = pairs_eval // 2

    # позитивы
    by_lbl = {}
    for i, y in enumerate(ys):
        by_lbl.setdefault(y, []).append(i)
    pos_cnt = 0
    while pos_cnt < half:
        lbl = rng.choice(list(by_lbl.keys()))
        if len(by_lbl[lbl]) < 2: 
            continue
        i, j = rng.choice(by_lbl[lbl], size=2, replace=False)
        d2 = np.sum((embs[i]-embs[j])**2)
        pos_scores.append(-d2); pos_labels.append(1)
        pos_cnt += 1

    # негативы
    neg_cnt = 0
    while neg_cnt < half:
        i, j = rng.integers(0, N, size=2)
        if ys[i] == ys[j]:
            continue
        d2 = np.sum((embs[i]-embs[j])**2)
        neg_scores.append(-d2); neg_labels.append(0)
        neg_cnt += 1

    scores = np.array(pos_scores + neg_scores)
    labels = np.array(pos_labels + neg_labels)
    return roc_auc_score(labels, scores)

def train_triplets(
    train_index: PersonIndex, val_index: PersonIndex,
    img_size=128, batch_size=64, m_per_class=4, lr=3e-4, weight_decay=1e-4,
    epochs=20, margin=0.2, avg_over="violating", ckpt_path="triplet_best.pt", device="cuda"
):
    assert batch_size % m_per_class == 0, "batch_size должен быть кратен m_per_class"
    k_classes = batch_size // m_per_class

    transform_train = make_transforms(img_size, aug=True)
    transform_val   = make_transforms(img_size, aug=False)

    train_ds = SingleImageDataset(train_index, transform_train)
    val_ds   = SingleImageDataset(val_index,   transform_val)

    sampler = BalancedBatchSampler(train_index.labels, m_per_class=m_per_class, num_classes_per_batch=k_classes)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=4, pin_memory=True)
    # для валидации просто пройдём по всему датасету (эмбеддинги и пары считаем вне DataLoader)

    encoder = SmallEncoder(emb_dim=128).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    criterion = BatchTripletLoss(margin=margin, avg_over=avg_over)

    best_auc = 0.0
    for epoch in range(epochs):
        encoder.train()
        losses = []
        viol_frac_log = []

        for imgs, labels in tqdm(train_loader, desc=f"[Triplet] epoch {epoch}", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            emb = encoder(imgs)
            # лосс
            loss = criterion(emb, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # небольшая метрика: доля нарушающих троек
            with torch.no_grad():
                # посчитаем грубо через положительные элементы loss_mat
                # (повторная прогонка pairwise для мониторинга; на практике можно оптимизировать)
                dist2 = BatchTripletLoss.pairwise_dist2(emb)
                same = labels.unsqueeze(0) == labels.unsqueeze(1)
                pos_mask = same & (~torch.eye(emb.size(0), dtype=torch.bool, device=emb.device))
                neg_mask = ~same
                L = torch.clamp(dist2.unsqueeze(2) - dist2.unsqueeze(1) + criterion.margin, min=0.0)
                valid = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
                num_viol = (L[valid] > 0).float().mean().item() if valid.any() else 0.0
                viol_frac_log.append(num_viol)

            losses.append(loss.item())

        # оценим AUC на валидации по парам
        val_auc = evaluate_auc_from_encoder(encoder, val_index, transform_val, device=device, pairs_eval=20000)
        scheduler.step(val_auc)
        print(f"Epoch {epoch:02d}  train_loss={np.mean(losses):.4f}  viol_frac~{np.mean(viol_frac_log):.3f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"encoder": encoder.state_dict(),
                        "config": {"img_size": img_size}}, ckpt_path)
            print(f"  ✓ New best AUC={best_auc:.4f} → saved {ckpt_path}")

    return best_auc

# -----------------------------
# СПЛИТ ТРЕЙН/ВАЛ
# -----------------------------
def split_index(index: PersonIndex, val_fraction=0.2) -> Tuple[PersonIndex, PersonIndex]:
    train_l2p, val_l2p = {}, {}
    l2n = index.label_to_name
    for lbl, paths in index.label_to_paths.items():
        if len(paths) < 3:
            # очень маленькие классы полностью в train (или как решишь)
            train_l2p[lbl] = paths
            continue
        k_val = max(1, int(len(paths)*val_fraction))
        v = random.sample(paths, k_val)
        t = [p for p in paths if p not in v]
        train_l2p[lbl] = t
        val_l2p[lbl]   = v

    def pack(l2p):
        paths, labels = [], []
        for lbl, ps in l2p.items():
            for p in ps:
                paths.append(p)
                labels.append(lbl)
        return PersonIndex(l2p, l2n, paths, labels, len(l2p))

    return pack(train_l2p), pack(val_l2p)

# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Папка датасета: root/person/*.jpg")
    ap.add_argument("--mode", type=str, choices=["pairs", "triplet"], default="pairs")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--m_per_class", type=int, default=4)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--quick_debug", type=float, default=1.0, help="например 0.1 = 10% картинок на класс")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # индекс + сплит
    full_index = build_index(args.data_root, quick_ratio=args.quick_debug)
    train_index, val_index = split_index(full_index, val_fraction=args.val_fraction)
    print(f"Classes: train={train_index.n_classes}  val={val_index.n_classes}")

    best_auc = train_triplets(
        train_index, val_index,
        img_size=args.img_size,
        batch_size=args.batch_size,
        m_per_class=args.m_per_class,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        margin=args.margin,
        avg_over="violating",
        ckpt_path="triplet_best.pt",
        device=device
    )
    print(f"[TRIPLET] Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
