import torch


@torch.no_grad()
def prep_for_answers(outputs, targets, names=None):
    outputs = outputs.cpu().detach().tolist()
    targets = targets.cpu().detach().tolist()


    if names is not None:
        if isinstance(names[0], list):
            names = [x for y in names for x in y]
        assert len(outputs) == len(targets) == len(names), f"len(outputs) == len(targets) == len(names). got {len(outputs)} == {len(targets)} == {len(names)}"
        return list(zip(names, outputs, targets))
    else:
        assert len(outputs) == len(targets), f"len(outputs) == len(targets). got {len(outputs)} == {len(targets)}"
        return list(zip(outputs, targets))