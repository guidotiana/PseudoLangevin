import torch
import torch.nn.functional as F



def generate_kspin_data(P, K, d, pflip, one_hot_encode_labels=False, seed=0, **kwargs):
    generator = torch.Generator().manual_seed(seed)
    
    ref_vectors = (2 * torch.randint(low=0, high=2, size=(K,d), generator=generator) - 1).float()
    
    labels = torch.randint(low=0, high=K, size=(P,), generator=generator)
    vectors = ref_vectors[labels]
    flips = torch.rand(size=(P,d), generator=generator) <= pflip
    vectors[flips] *= -1

    if one_hot_encode_labels:
        labels = F.one_hot(labels, num_classes=K)

    return ref_vectors, vectors, labels



""" ########## """
""" OPERATIONS """
""" ########## """

def compute_kspin_similarity(vectors_a, vectors_b=None):
    if vectors_b is None:
        similarity = torch.ones(len(vectors_a), len(vectors_a))
        for row in range(len(vectors_a)):
            similarity[row, row+1:] = (vectors_a[row] == vectors_a[row+1:]).float().mean(axis=1)
            similarity[row, :row] = similarity[:row, row]
    else:
        similarity = torch.ones(len(vectors_a), len(vectors_b))
        for row in range(len(vectors_a)):
            similarity[row, :] = (vectors_a[row] == vectors_b).float().mean(axis=1)
    return similarity

#Alternative to similarity: alignment = 2*similarity-1
def compute_kspin_alignment(vectors_a, vectors_b=None):
    similarity = compute_kspin_similarity(vectors_a, vectors_b=None)
    return 2. * similarity - 1.

def compute_kspin_miscounts(ref_vectors, vectors, labels, fraction=True):
    if labels.ndim == 2:
        K = labels.shape[1]
        labels = labels.argmax(axis=-1)
    else:
        K = labels.max().item()+1

    miscounts = []
    for label in range(K):
        mask = labels==label
        similarity = compute_kspin_similarity(ref_vectors, vectors[mask])
        miscount = (similarity.argmax(axis=0) != label).sum().item()
        if fraction:
            miscount /= mask.sum().item()
        miscounts.append(miscount)

    return torch.tensor(miscounts)
