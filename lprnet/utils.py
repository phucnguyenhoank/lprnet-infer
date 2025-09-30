import string
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Build mappings
# -------------------------
CHARS = list(string.digits + string.ascii_uppercase)
CHARS.append('-')

char_to_idx = {ch: i for i, ch in enumerate(CHARS)}
idx_to_char = {i: ch for i, ch in enumerate(CHARS)}

blank_idx = char_to_idx['-']   # CTC blank

# -------------------------
# Greedy decoder
# -------------------------
def greedy_decode(logits, idx_to_char, blank_idx):
    """
    logits: [T, N, C] tensor (log probs or raw logits)
    idx_to_char: dict mapping int -> char
    blank_idx: index of blank symbol
    """
    preds = logits.argmax(2).permute(1, 0)   # [N, T]

    results = []
    for pred in preds:
        string = ""
        prev = None
        for p in pred.cpu().numpy():
            if p != prev and p != blank_idx:   # collapse repeats + remove blank
                string += idx_to_char[p]
            prev = p
        results.append(string)
    return results

# -------------------------
# Predict single image
# -------------------------
def predict_image(model, image):
    """
    model : trained model
    image : tensor [3, 24, 94] (CHW, normalized to [0,1])
    """
    # model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(DEVICE)        # [1, 3, 24, 94]
        logits = model(image)                        # [N, C, T]
        logits = logits.permute(2, 0, 1)             # [T, N, C]

        preds = greedy_decode(logits, idx_to_char, blank_idx)
        return preds[0]
