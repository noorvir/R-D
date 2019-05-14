import torch


def match_loss():
    # compute loss for pixels that should be similar
    pass


def non_match_loss():
    # compute loss for pixels that need to be pushed apart
    pass


def triplet_loss():
    pass


def correspondence_loss(output, clist, margin=0.5, hard_negative=True):

    loss = 0
    means = []

    # TODO: incorporate object matches. Currently matches on object are considered both
    #       during minization of within cluster variance as well as when maximizing difference
    #       between cluster means. This leads to conflicting training signal.
    for mval_tuple in clist:
        matches_idx = mval_tuple[0]
        non_matches_idx = mval_tuple[1]
        obj_matches_idx = mval_tuple[2]

        match_vals = output.index_select(idx=matches_idx)
        non_match_vals = output.index_select(idx=non_matches_idx)
        obj_match_vals = output.index_select(idx=obj_matches_idx)

        # minimise variance
        # maximise distance between means of these classes

        var_matches = torch.var(match_vals)
        var_non_matches = torch.var(non_match_vals)
        var_obj_matches = torch.var(obj_match_vals)

        loss += var_matches + var_non_matches + var_obj_matches
        means.append(torch.mean(match_vals))

    for i, mean in enumerate(means):
        loss += torch.sum(mean - means[i:]).pow(2)
