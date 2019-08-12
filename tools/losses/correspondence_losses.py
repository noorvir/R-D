import torch
from dataloaders.correspondence import find_correspondences


def match_loss():
    # compute loss for pixels that should be similar
    pass


def non_match_loss():
    # compute loss for pixels that need to be pushed apart
    pass


def triplet_loss():

    # foreach batch
    #   clist = find_correspondences(mat_masks)
    #
    #   foreach material in clist
    #       let num
    #       sample
    #       triplet_losses = (matches_a_descriptors - matches_b_descriptors).pow(2) - (matches_a_descriptors - non_matches_b_descriptors).pow(2) + alpha
    #         triplet_loss = 1.0 / num_non_matches * torch.clamp(triplet_losses, min=0).sum()
    #
    #
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


def cluster_loss(output, mat_masks, obj_masks, dtypes):

    # Find correspondences
    # evaluate triplet loss at correspondences

    total_loss = 0

    # iterate over batch (?)
    for mat_mask, obj_mask in zip(mat_masks, obj_masks):
        clist = find_correspondences(mat_mask, obj_mask, dtypes)

        variances_list = []
        means_list = []
        similar_non_matches_loss_list = []

        for material in clist:

            # 1. Get matches and non-matches
            match_idx = material[0]
            non_match_idx = material[1]
            obj_match_idx = material[2]

            matches = output[:, :, match_idx[:, 0], match_idx[:, 1]]
            non_matches = output[:, :, non_match_idx[:, 0], non_match_idx[:, 1]]
            obj_matches = output[:, :, obj_match_idx[:, 0], obj_match_idx[:, 1]]

            # 2. Compute mean and variance for each cluster
            variances_list.append(torch.var(matches))
            means_list.append(torch.mean(matches))

            # 3. Compute (dis)similarity in pixel intensity
            mshape = matches.shape          # (N, C, num_matches)
            nmshape = non_matches.shape
            rand_idx = torch.randint(0, nmshape[-1], (mshape[-1],)).type(dtypes.long)
            similar_non_matches = non_matches[:, :, rand_idx]
            similar_non_matches_loss = torch.mean((matches - similar_non_matches).pow(2))
            similar_non_matches_loss_list.append(similar_non_matches_loss)

        variances_tensor = torch.tensor(variances_list).type(dtypes.float)
        similarity_tensor = torch.tensor(similar_non_matches_loss_list).type(dtypes.float)

        mean_loss = 0
        var_loss = torch.mean(variances_tensor)
        similarity_loss = torch.mean(similarity_tensor)

        # Compute loss b/w each mean and all the rest.
        for i in range(len(means_list)):
            means_tensor = torch.tensor(means_list[:i] + means_list[i + 1:]).type(dtypes.float)
            mean_loss += torch.mean((means_list[i] - means_tensor).pow(2))

        total_loss += (var_loss + 1/similarity_loss + 1/mean_loss)

    return total_loss
