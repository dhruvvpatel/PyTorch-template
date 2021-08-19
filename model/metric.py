# :: dhruv :: #


import torch

def accuracy(pred, gt):
    with torch.no_grad():
        best_pred = torch.argmax(pred, dim=1)
    
        assert best_pred.shape[0] == len(gt)
        correct = 0
        correct += torch.sum(best_pred == gt).item()

    return correct / len(gt)


def top_k_acc(pred, gt, k=3):
    with torch.no_grad():
        best_pred = torch.topk(pred, k, dim=1)[1]
        assert best_pred.shape[0] == len(gt)
        correct = 0
        for i in range(k):
            correct += torch.sum(best_pred[:,i] == gt).item()

    return correct / len(gt)

