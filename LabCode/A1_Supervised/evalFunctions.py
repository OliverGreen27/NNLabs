import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    T = 0
    for i in range(len(LPred)):
        if LPred[i] == LTrue[i]:
            T += 1
            
    acc = T/len(LPred)
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    countClassMax = max(len(set(LPred)), len(set(LTrue)))
    cM = np.zeros((countClassMax, countClassMax),int)
    
    for i in range(len(LPred)):
        cM[LPred[i], LTrue[i]] += 1
            
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    count = 0
    for i in range(len(cM)):
        count += cM[i, i]
        
    acc = count / np.sum(cM)
    # ============================================
    
    return acc
