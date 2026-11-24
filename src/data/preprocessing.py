def pad_or_truncate(sequence, max_length, pad_id):
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    
    if len(sequence) < max_length:
        return sequence + [pad_id] * (max_length - len(sequence))
    else:
        return sequence[:max_length]

