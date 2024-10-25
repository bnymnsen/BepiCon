import torch

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torchmetrics import Metric
from torchmetrics.functional import pairwise_cosine_similarity



class Silhouette(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("silhouette_score", default=torch.tensor(0.0))
        self.add_state("count", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Calculates the silhouette score using `sklearn.metrics.silhouette_score`
        Args:
            preds: Embeddings of the model. Shape: [B, D]
            target: Target labels. Shape: [B]
        Returns:
            None
        """
        
        self.silhouette_score += silhouette_score(preds.cpu().numpy(), target.cpu().numpy())
        self.count += 1

    def compute(self) -> torch.Tensor:
        return self.silhouette_score / self.count
    

class InterAndIntraClassSimilarity(Metric):
    def __init__(self):
        super().__init__()

        self.add_state("intra_class_similarity", default=torch.tensor(0.0))
        self.add_state("inter_class_similarity", default=torch.tensor(0.0))
        self.add_state("count", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Calculates the inter and intra class similarity using pairwise cosine similarity
        Args:
            preds: Embeddings of the model. Shape: [B, D]
            target: Target labels. Shape: [B]
        Returns:
            None
        """

        cos_sim = pairwise_cosine_similarity(preds, zero_diagonal=True).detach().cpu()

        # Create a boolean mask for intra-class and inter-class pairs
        same_label_mask = target.cpu().unsqueeze(1) == target.cpu().unsqueeze(0)
        # Create a mask for the upper triangle (excluding diagonal)
        upper_triangle_mask = torch.triu(torch.ones_like(same_label_mask), diagonal=1)

        # Combine masks
        intra_class_mask = same_label_mask & upper_triangle_mask
        inter_class_mask = (~same_label_mask) & upper_triangle_mask

        intra_class_sim = cos_sim[intra_class_mask]
        inter_class_sim = cos_sim[inter_class_mask]

        self.intra_class_similarity += intra_class_sim.mean()
        self.inter_class_similarity += inter_class_sim.mean()
        self.count += 1

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inter_class_similarity / self.count, self.intra_class_similarity / self.count
    

class KNNAccuracy(Metric):
    def __init__(self, k: int = 20):
        self.k = k
        super().__init__()
        self.add_state("accuracy", default=torch.tensor(0.0))
        self.add_state("count", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Calculates the accuracy of the KNN classifier
        Args:
            preds: Embeddings of the model. Shape: [B, D]
            target: Target labels. Shape: [B]
        Returns:
            None
        """
        preds = preds.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        preds = KNeighborsClassifier(self.k, metric="cosine").fit(preds, target).predict(preds)
        self.accuracy += accuracy_score(target, preds)
        self.count += 1

    def compute(self) -> torch.Tensor:
        return self.accuracy / self.count
    

class AverageMeter:
    def __init__(self):
        self.values = torch.tensor(0.0)
        self.total = torch.tensor(0)
    
    def update(self, value: torch.Tensor | float) -> None:
        self.values += value
        self.total += 1
    
    def avg(self) -> torch.Tensor:
        return self.values / self.total
    
    def reset(self) -> None:
        self.values = torch.tensor(0.0)
        self.total = torch.tensor(0)